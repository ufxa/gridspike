"""phase 3 tests for the ``experiments`` runner: manifest, run_one, analyze.

Coverage of the deliverables' contracts:

* the dry-run manifest has the expected Compact total (500), no ``run_id``
  collisions, and byte-stable ``config_hash`` across two independent builds;
* ``run_one`` is idempotent (a second call on a finished run SKIPs without
  training);
* a real tiny end-to-end run (2 rounds via ``--limit-rounds``, K=4, --days 1
  data, one clean arm + one attacked+defended arm) writes a valid RunResult with
  populated final metrics and non-null provenance, and the attacked+defended run
  rejects the attacker in at least one round;
* ``analyze`` produces the full statistical battery on synthetic RunResults with
  a planted significant difference and detects it (Wilcoxon+Holm, Friedman,
  Cliff's delta, BCa CIs, McNemar).

Kept fast (< ~90 s): K=4, 2 rounds, a shrunk CNN-LSTM, one shared generated
federation, and synthetic RunResults for the statistics (no training).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from experiments import analyze as A
from experiments.manifest import (
    COMPACT_TIER1_SEEDS,
    COMPACT_TIER2_SEEDS,
    battery_runs,
    block_counts,
    config_hash,
    make_config,
)
from experiments.run_one import run_line

from fedgrid.metrics.runresult import RunResult, load_run, save_run

_TINY_MODEL = {"conv_channels": [8, 16], "lstm_hidden": 16, "lstm_layers": 1}


# --------------------------------------------------------------------------- #
# Manifest.
# --------------------------------------------------------------------------- #


def test_compact_battery_total_and_blocks() -> None:
    runs = battery_runs()
    counts = block_counts(runs)
    assert counts == {
        "t1a": 60, "t1b": 200, "t2a": 72, "t2b": 48,
        "t2c": 72, "t2d": 12, "t2e": 36, "TOTAL": 500,
    }
    # ~540 with the documented 8% re-run margin (amendment 2026-07-28e/h).
    assert 500 <= counts["TOTAL"] * 1.08 <= 545


def test_full_standard_reproduces_920() -> None:
    # Tier-1 = 10, Tier-2 = 5 reproduces the §3.9 base (920; ~995 with margin).
    assert block_counts(battery_runs(10, 5))["TOTAL"] == 920


def test_run_ids_unique_and_fs_safe() -> None:
    runs = battery_runs()
    ids = [r.run_id for r in runs]
    assert len(set(ids)) == len(ids)  # battery_runs also asserts this internally
    import re

    assert all(re.fullmatch(r"[A-Za-z0-9_.\-]+", rid) for rid in ids)


def test_config_hash_stable_across_two_builds() -> None:
    a = {r.run_id: r.config_hash for r in battery_runs()}
    b = {r.run_id: r.config_hash for r in battery_runs()}
    assert a == b
    # and every stored hash is the sha256 of its own canonical config
    assert all(r.config_hash == config_hash(r.config) for r in battery_runs())


def test_manifest_line_schema() -> None:
    spec = battery_runs()[0]
    entry = json.loads(spec.to_json_line())
    assert set(entry) == {"run_id", "arm", "model", "seed", "tier", "block",
                          "config", "config_hash"}
    assert entry["config_hash"] == config_hash(entry["config"])


def test_attack_params_reach_the_attack_constructor() -> None:
    """``attack.params`` is the only lever on attacker strength; it must arrive.

    The magnitude sweep (block T4mag) exists because a cosine gate cannot see
    magnitude, so ``scale`` has to travel manifest -> config -> constructor intact.
    """
    from experiments.build import make_attacks

    cfg = make_config(model="cnn_lstm", arm="pipeline_krum", seed=3,
                      attack="adaptive_cosine", attack_fraction=0.3,
                      attack_params={"scale": 5.0, "target_score": 0.4})
    assert cfg["attack"]["params"] == {"scale": 5.0, "target_score": 0.4}
    attacks = make_attacks(cfg)
    assert set(attacks) == {7, 8, 9}  # the faulty tail at f = 3/10
    atk = attacks[9]
    assert type(atk).__name__ == "AdaptiveCosine"
    assert atk.scale == 5.0
    assert atk.target_score == 0.4
    assert atk.collude_seed == 3 and atk.n_peers == 10  # the colluding coalition


def test_attack_defaults_unchanged_without_params() -> None:
    """No params -> the historical defaults, so every past run stays valid."""
    from experiments.build import make_attacks

    def one(attack: str) -> object:
        cfg = make_config(model="cnn_lstm", arm="no_defense", seed=0,
                          attack=attack, attack_fraction=0.3)
        assert cfg["attack"]["params"] == {}
        return make_attacks(cfg)[9]

    adaptive = one("adaptive_cosine")
    assert adaptive.scale == 1.0 and adaptive.target_score == 0.1
    assert one("sign_flip").scale == 1.0
    assert one("gaussian_noise").sigma == 1.0
    assert one("label_flip").frac == 1.0


def test_magnitude_sweep_grid_and_metadata() -> None:
    from experiments.gen_magnitude_sweep import ARMS, SCALES, magnitude_runs

    runs = magnitude_runs()
    assert len(runs) == 144  # 4 scales x 4 arms x 3 models x 3 seeds
    assert len({r.run_id for r in runs}) == 144
    assert {r.block for r in runs} == {"t4mag"} and {r.tier for r in runs} == {3}
    assert {r.arm for r in runs} == set(ARMS)
    assert {r.model for r in runs} == {"cnn_lstm", "snn", "ssm"}
    assert {r.seed for r in runs} == {0, 1, 2}
    for r in runs:
        entry = json.loads(r.to_json_line())
        assert set(entry) == {"run_id", "arm", "model", "seed", "tier", "block",
                              "config", "config_hash"}
        atk = entry["config"]["attack"]
        assert atk["type"] == "adaptive_cosine" and atk["fraction"] == 0.3
        assert atk["params"]["scale"] in SCALES
        assert entry["config_hash"] == config_hash(entry["config"])
        assert r.run_id.startswith("t4mag-") and r.run_id.endswith(f"-s{r.seed}")
    # Every scale is present the same number of times (a full factorial grid).
    per_scale = {s: sum(1 for r in runs if r.config["attack"]["params"]["scale"] == s)
                 for s in SCALES}
    assert set(per_scale.values()) == {36}


def test_magnitude_sweep_scale_one_is_the_historical_operating_point() -> None:
    """scale=1.0 reproduces the T1b adaptive cells: only ``attack.params`` differs."""
    from experiments.gen_magnitude_sweep import magnitude_runs

    battery = {(r.arm, r.model, r.seed): r for r in battery_runs()
               if r.block == "t1b" and r.config["attack"]["type"] == "adaptive_cosine"}
    checked = 0
    for r in magnitude_runs():
        if r.config["attack"]["params"]["scale"] != 1.0:
            continue
        twin = battery.get((r.arm, r.model, r.seed))
        if twin is None:  # t4mag runs 3 seeds; t1b's ssm lines live in a separate manifest
            continue
        a = {k: v for k, v in r.config.items() if k != "attack"}
        b = {k: v for k, v in twin.config.items() if k != "attack"}
        assert a == b
        assert r.config["attack"]["params"] == {"scale": 1.0}
        assert twin.config["attack"]["params"] == {}
        checked += 1
    assert checked >= 24


def test_dry_run_writes_manifest(tmp_path: Path) -> None:
    from experiments.run_all import write_manifest

    out = tmp_path / "manifest.jsonl"
    n = write_manifest(out, COMPACT_TIER1_SEEDS, COMPACT_TIER2_SEEDS)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert n == len(lines) == 500
    assert all(set(json.loads(x)) == {"run_id", "arm", "model", "seed", "tier",
                                      "block", "config", "config_hash"} for x in lines)


# --------------------------------------------------------------------------- #
# End-to-end run_one + idempotency (real training on a shared tiny federation).
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tiny_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One K=4, 1-day federation generated once and shared by the E2E tests."""
    from fedgrid.simulation.generate import main as generate_main

    data_dir = tmp_path_factory.mktemp("k4data")
    rc = generate_main(["--k", "4", "--days", "1", "--seed", "0", "--out", str(data_dir)])
    assert rc == 0
    assert (data_dir / "manifest.json").exists()
    return data_dir


def _entry(arm: str, **cfg_kwargs: object) -> dict[str, object]:
    cfg = make_config(model="cnn_lstm", arm=arm, seed=0, k=4, days=1, rounds=2,
                      batch_size=16, lr=5e-2, model_kwargs=_TINY_MODEL, **cfg_kwargs)
    rid = f"e2e-{arm}"
    return {"run_id": rid, "arm": arm, "model": "cnn_lstm", "seed": 0,
            "tier": 1, "block": "e2e", "config": cfg, "config_hash": config_hash(cfg)}


def test_e2e_clean_run_and_idempotency(tiny_data_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "runs"
    entry = _entry("fedavg")
    logs: list[str] = []

    code, path = run_line(entry, data_dir=tiny_data_dir, out_dir=out_dir,
                          limit_rounds=2, log=logs.append)
    assert code == 0
    assert path.exists()
    result = load_run(path)
    # Populated final metrics.
    for key in ("acc", "macro_f1", "per_class_f1", "confusion"):
        assert key in result.final
    assert isinstance(result.final["macro_f1"], float)
    assert 0.0 <= result.final["acc"] <= 1.0
    assert len(result.rounds) == 2  # limit-rounds honoured
    # Non-null provenance.
    assert isinstance(result.git_commit, str) and result.git_commit
    assert "py" in result.env_fingerprint
    # Per-sample correctness attached for the H1a McNemar.
    assert result.final["n_test"] == len(result.final["correct"])

    # Idempotent: a second call SKIPs without retraining.
    logs.clear()
    code2, path2 = run_line(entry, data_dir=tiny_data_dir, out_dir=out_dir,
                            limit_rounds=2, log=logs.append)
    assert code2 == 0 and path2 == path
    assert any(msg == f"SKIP {entry['run_id']}" for msg in logs)
    assert not any(msg.startswith("[run_one] START") for msg in logs)


def test_e2e_attacked_defended_rejects_attacker(tiny_data_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "runs"
    # 1 attacker at K=4 (fraction 0.25 -> cid 3), M=4 f=1, tau=0.5.
    entry = _entry("pipeline_krum", attack="sign_flip", attack_fraction=0.25,
                   m=4, f=1, threshold=0.5)
    code, path = run_line(entry, data_dir=tiny_data_dir, out_dir=out_dir, limit_rounds=2)
    assert code == 0

    result = load_run(path)
    assert isinstance(result.final["macro_f1"], float)
    # The sign-flipping attacker (cid 3) is rejected by the fidelity gate in >=1 round.
    attacker = 3
    rejected_somewhere = any(attacker in ev["consensus"]["rejected"] for ev in result.rounds)
    assert rejected_somewhere, "attacked+defended run never rejected the attacker"
    # And the committing PBFT round has the exact 2M(M-1) message count for M=4.
    assert any(ev["consensus"]["n_messages"] == 24 for ev in result.rounds)


def test_run_one_atomic_no_tmp_left(tiny_data_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "runs"
    entry = _entry("fedavg")
    run_line(entry, data_dir=tiny_data_dir, out_dir=out_dir, limit_rounds=2)
    assert not list(out_dir.glob("*.tmp"))  # os.replace leaves no partial file


# --------------------------------------------------------------------------- #
# analyze: full battery on synthetic RunResults with a planted difference.
# --------------------------------------------------------------------------- #


def _make_run(runs_dir: Path, run_id: str, arm: str, model: str, seed: int,
              macro_f1: float, attack: str | None = None,
              correct: list[int] | None = None) -> None:
    inner = {
        "attack": {"type": attack, "fraction": 0.3, "params": {}} if attack else None,
        "federation": {"k": 10, "alpha": 0.5, "days": 30, "seed": seed, "iid": False},
    }
    manifest = {"run_id": run_id, "arm": arm, "model": model, "seed": seed,
                "tier": 1, "block": "syn", "config": inner}
    final: dict[str, object] = {"acc": macro_f1, "macro_f1": macro_f1}
    if correct is not None:
        final["correct"] = correct
        final["n_test"] = len(correct)
    rr = RunResult(method=arm, seed=seed, config={"manifest": manifest},
                   rounds=[], final=final, energy=None, latency=None,
                   git_commit="deadbeef", env_fingerprint="py-test")
    save_run(rr, runs_dir / f"{run_id}.json")


@pytest.fixture
def synthetic_runs(tmp_path: Path) -> Path:
    """A runs/ dir with a planted pipeline>no_defense effect and an SNN/CNN discordance."""
    runs = tmp_path / "runs"
    runs.mkdir()
    n_seeds = 10
    # Attack-grid cell (cnn_lstm, sign_flip): three arms, consistent ordering so
    # Wilcoxon (pipeline vs no_defense) and Friedman both fire.
    for s in range(n_seeds):
        _make_run(runs, f"nd-{s}", "no_defense", "cnn_lstm", s, 0.50 + 0.004 * s, "sign_flip")
        _make_run(runs, f"ka-{s}", "krum_alone", "cnn_lstm", s, 0.60 + 0.004 * s, "sign_flip")
        _make_run(runs, f"pk-{s}", "pipeline_krum", "cnn_lstm", s, 0.72 + 0.004 * s, "sign_flip")
    # Clean parity arms with a planted McNemar discordance (b=60, c=10 per seed).
    both_right = [1] * 100
    snn_only = [1] * 60          # snn right, cnn wrong  -> b
    cnn_only = [0] * 10          # snn wrong, cnn right  -> c (snn side)
    both_wrong = [0] * 30
    cnn_correct = [1] * 100 + [0] * 60 + [1] * 10 + [0] * 30
    snn_correct = both_right + snn_only + cnn_only + both_wrong
    for s in range(3):
        _make_run(runs, f"cnn-{s}", "fedavg", "cnn_lstm", s, 0.88, None, cnn_correct)
        _make_run(runs, f"snn-{s}", "fedavg", "snn", s, 0.88, None, snn_correct)
    return runs


def test_analyze_full_battery_detects_planted_effect(synthetic_runs: Path) -> None:
    summary = A.analyze_dir(synthetic_runs, n_resamples=2000)

    # Per-arm summary with BCa CIs present for every arm.
    assert summary["per_arm"]
    for s in summary["per_arm"].values():
        assert s["bca_ci_low"] <= s["mean_macro_f1"] <= s["bca_ci_high"]

    # Wilcoxon + Holm: pipeline_krum beats no_defense, detected and large effect.
    pk = next(e for e in summary["pairwise_wilcoxon_holm"]
              if e["arm"] == "pipeline_krum" and e["model"] == "cnn_lstm")
    assert pk["mean_diff"] > 0
    assert pk["p_holm"] < 0.05  # 10 all-positive pairs -> exact p far below 0.05
    assert pk["cliffs_delta"] == 1.0 and pk["magnitude"] == "large"

    # Friedman across the three defense arms fires with a low p-value.
    fr = next(f for f in summary["friedman_nemenyi"]
              if f["model"] == "cnn_lstm" and f["attack"] == "sign_flip")
    assert len(fr["arms"]) == 3
    assert fr["p"] < 0.05

    # McNemar detects the SNN/CNN-LSTM discordance (pooled b=180, c=30).
    mc = summary["mcnemar_parity"]
    assert mc["available"] is True
    assert mc["pooled"]["b"] == 180 and mc["pooled"]["c"] == 30
    assert mc["pooled"]["p"] < 0.05

    # The report renders without error.
    assert "statistical battery" in A.format_report(summary)


def test_analyze_wilcoxon_floor_documented() -> None:
    # The amendment's "5 seeds = Wilcoxon floor": 5 all-positive pairs cannot reach
    # a two-sided p < 0.05 (min is 0.0625); the one-sided p (0.03125) can.
    w = A.wilcoxon_signed_rank([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])
    assert w["p_two"] == pytest.approx(0.0625)
    assert w["p_greater"] == pytest.approx(0.03125)
