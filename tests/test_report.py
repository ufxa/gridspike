"""phase 5 tests for the results pipeline: generated tables, summary, figures.

What is pinned here:

* all **twenty-one** contracted table files are written, as bodies only (no
  float, no caption -- those live in the section files);
* each file still matches **its stub's documented schema**: the same column
  specification, the same header cells, and the same row labels, so the
  manuscript's ``\\input`` keeps working after any regeneration;
* every body row has exactly the header's column count -- a silent column drift
  would compile into a mangled table;
* a cell the runs cannot support renders ``\\TODOnum`` -- never ``0``, never a
  plausible-looking value -- and the marker count *falls* as real runs land,
  while a cell no battery could ever fill renders ``--`` instead;
* ``summary.json`` is written and round-trips through JSON;
* all ten figures render to a temp dir as real PDFs, including the placeholder
  path for a figure whose measurement has not run, and every one of them is
  saved at one journal column so the manuscript scales none of them (see
  ``_NOTE_WIDENED_FLOOR`` for the one partial-battery exception).

Synthetic RunResults only -- no training, no staged data.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
from experiments import report as R

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# --------------------------------------------------------------------------- #
# The stub contract (paper/OPEN_QUESTIONS_PAPER.md sec 4.2), transcribed from the
# pristine stubs. The generator must keep reproducing exactly this.
# --------------------------------------------------------------------------- #

STUB_SCHEMA: dict[str, tuple[str, tuple[str, ...]]] = {
    "tab_dataset.tex": (
        "l l r r l r r",
        ("Client", "Feeder", r"PV (\%)", "Windows", "Train/Val/Test",
         r"\cls{normal} (\%)", "Classes")),
    # The worst-feeder-day column was dropped: a single extreme feeder-day is an
    # anecdote, and the held *rate* over every feeder-day already carries it.
    "tab_nonconvergence.tex": (
        "l r r r",
        ("Feeder", "Feeder-days", "Held steps", r"Held rate (\%)")),
    "tab_reference_frame.tex": (
        "l l l r c r r",
        ("", "Arm", "Model", r"\macroF{}", r"BCa 95\% CI", "Accuracy",
         "Rounds to target")),
    "tab_attack_defense.tex": (
        "l l r r r r r",
        ("Defense arm", "Model", "Clean", "Sign-flip", "Gaussian",
         r"Label-flip$^{\dagger}$", "Adaptive")),
    "tab_defense_decomposition.tex": (
        "l l r r r r c",
        ("Defense rung", "Model", r"\macroF{}", r"$\Delta$ vs none",
         "Attacker adm.", "Honest FRR", "Ledger")),
    "tab_adaptive_evasion.tex": (
        "l l l r r r r",
        ("Aggregator", "Attack", "Model", "Mean score", "Admission", "Honest FRR",
         r"$\Delta$\macroF{}")),
    "tab_ldp_curve.tex": (
        "c c r r r r r r",
        (r"$\tau$", r"$\sigma$", r"$\varepsilon$", "Honest FRR", "Attacker adm.",
         r"\macroF{} CNN-LSTM", r"\macroF{} Spiking", r"\macroF{} Mamba")),
    # The T3tau block (540 runs, 41% of the battery) had no reader until
    # 2026-07-30. tab_tau_sweep takes its 216 attack runs -- the admission
    # threshold swept against the defense-aware attacker -- and
    # tab_ldp_tau_extended takes its 324 LDP runs, carrying the sigma x tau grid
    # past T2c's tau=0.7 up to a fully open gate at tau=1.0.
    "tab_tau_sweep.tex": (
        "c l r r r r r",
        (r"$\tau$", "Model", "Honest FRR", "Attacker adm.", "Attacker score",
         r"\macroF{} Krum", r"\macroF{} geo-med")),
    "tab_ldp_tau_extended.tex": (
        "c c r r r r r r",
        (r"$\tau$", r"$\sigma$", r"$\varepsilon$", "Honest FRR", "Attacker adm.",
         r"\macroF{} CNN-LSTM", r"\macroF{} Spiking", r"\macroF{} Mamba")),
    "tab_alpha_sweep.tex": (
        "l l r r r r r r",
        (r"$\alpha$", "Arm", "CNN clean", "CNN adaptive", "Spiking clean",
         "Spiking adaptive", "Mamba clean", "Mamba adaptive")),
    # Ledger dropped (it is constant over a committee-size sweep -- every rung
    # here runs PBFT -- and tab_defense_decomposition is where it discriminates);
    # the third model takes its place.
    # The macro-F1 columns are measured *under the adaptive attacker* -- T2d
    # sweeps the committee on the pipeline-Krum arm only -- so the header carries
    # the dagger that names the condition rather than leaving it to the caption.
    "tab_consensus_overhead.tex": (
        "c c r r r r r r",
        ("$M$", "$f$", "Messages/round", r"Sim.\ p50 (ms)", r"Sim.\ p95 (ms)",
         r"\macroF{}$^{\dagger}$ CNN", r"\macroF{}$^{\dagger}$ SNN",
         r"\macroF{}$^{\dagger}$ Mamba")),
    "tab_energy.tex": (
        "l l l r r r r",
        ("Model", "Coding", "Precision", "SynOps / MACs per inf.", "Spike rate",
         r"Est.\ \uJ{}/inf.", r"Ratio $\rho$")),
    # fp32 only, so there is no precision axis left to tabulate: torch's dynamic
    # INT8 kernels reject the recurrent and state-space packed parameters, so an
    # INT8 *latency* row would have to be invented. The INT8 estimate that can be
    # computed honestly is the energy one, and it stays in tab_energy.
    "tab_latency.tex": (
        "l l r r r r r",
        ("Model", "Device", "Mean (ms)", "p50 (ms)", "p95 (ms)", "p99 (ms)",
         r"Share of 4\,ms env.\ (\%)")),
    "tab_per_class.tex": (
        "l r r r r r",
        ("Class", "Support", r"F\textsubscript{1} CNN", "Recall CNN",
         r"F\textsubscript{1} SNN", "Recall SNN")),
    # p99 dropped: latency is a property of the deployed model, not of the
    # coding x T training grid, and tab_latency reports it per device.
    "tab_coding_tsteps.tex": (
        "l c r r r r",
        ("Coding", "$T$", r"\macroF{} clean", r"\macroF{} adaptive", "SynOps/inf.",
         r"Est.\ \uJ{}/inf.")),
    # The Delta column is a *paired* difference over the seeds the row shares
    # with T1a FedAvg, which is not the five-seed mean in the absolute column; the
    # subtrahend now has a column of its own so the arithmetic reconciles on the
    # page instead of looking like six broken cells out of nine.
    "tab_fedprox_mu.tex": (
        "l l r c r r",
        (r"$\mu$", "Model", r"\macroF{}", r"BCa 95\% CI", "FedAvg (paired)",
         r"$\Delta$ vs FedAvg")),
    # Round 3 (2026-07-31). The reviewers asked for the reference frame without
    # window leakage: t1a_temporal / t1b_temporal re-run the whole Tier-1 grid
    # with each client's windows cut chronologically instead of at random, so
    # every cell has a matched twin in T1a/T1b and the difference is the split
    # alone. tab_temporal_attack deliberately repeats tab_attack_defense's shape
    # so the two read side by side.
    "tab_temporal_frame.tex": (
        "l l r r r r",
        ("Arm", "Model", r"\macroF{} random", r"\macroF{} temporal", r"$\Delta$",
         r"$\Delta$ (\%)")),
    "tab_temporal_attack.tex": (
        "l l r r r r r",
        ("Defense arm", "Model", "Clean", "Sign-flip", "Gaussian",
         r"Label-flip$^{\dagger}$", "Adaptive")),
    # t4mag: the adaptive attacker aims inside the honest cone, so the cosine
    # gate has no direction to reject it on and magnitude is the only axis left.
    "tab_magnitude.tex": (
        "c l r r r r r r",
        ("Scale", "Arm", r"\macroF{} CNN-LSTM", r"\macroF{} Spiking",
         r"\macroF{} Mamba", r"Atk.\ score", r"Atk.\ adm.", "Honest FRR")),
    "tab_stats.tex": (
        r"l p{0.19\textwidth} l r r r r l c c",
        ("Family", "Comparison", "Test", "Statistic", "$p$", "Holm $p$", "Effect",
         "Band", r"BCa 95\% CI", "$n$")),
    # The evidence column wraps: as a bare `l' the widest cell was already
    # 45.8pt overfull, and the power-floor clause _family() appends took it to
    # 188pt. Both text columns are p{} so the table stays inside the margin.
    "tab_hypotheses.tex": (
        r"l p{0.36\textwidth} p{0.34\textwidth} l",
        ("", "Pre-registered claim", "Evidence", "Verdict")),
}

#: Row labels the manuscript's prose refers to; they must survive regeneration.
STUB_ROW_LABELS: dict[str, tuple[str, ...]] = {
    # Three models per arm (CNN-LSTM / Spiking / Mamba), so each arm spans three
    # battery blocks instead of two: B1--B3 rather than B1/B2.
    "tab_reference_frame.tex": ("B1--B3", "B4--B6", "B7--B9", "B10--B12",
                                "B13--B15", "B16--B18"),
    "tab_attack_defense.tex": ("No defense", "Krum alone", "Geo-median alone",
                               "Pipeline + Krum", "Pipeline + geo-median"),
    "tab_defense_decomposition.tex": ("No defense", "Fidelity gate only",
                                      "Krum alone", "Geo-median alone",
                                      "Fidelity + PBFT + Krum",
                                      "Fidelity + PBFT + geo-median"),
    "tab_hypotheses.tex": ("H1a", "H1b", "H1", "H2", "H3a", "H3b", "H3", "H4",
                           "H5a", "H5b", "H5"),
    "tab_nonconvergence.tex": ("IEEE 13", "IEEE 34", "IEEE 37", "IEEE 123", "EU LV"),
    "tab_per_class.tex": (r"\cls{normal}", r"\cls{fault\_lg}", r"\cls{swell}",
                          "Macro"),
}


# --------------------------------------------------------------------------- #
# Synthetic RunResults.
# --------------------------------------------------------------------------- #

_N_TEST = 40
_N_CLASSES = 8


def _confusion(macro_f1: float) -> list[list[int]]:
    """A confusion matrix whose row sums are a fixed, plausible support vector."""
    support = [26, 2, 2, 2, 2, 2, 2, 2]
    hit = max(1, int(round(macro_f1 * 2)))
    rows = []
    for i, n in enumerate(support):
        row = [0] * _N_CLASSES
        row[i] = min(n, hit if i else n)
        row[(i + 1) % _N_CLASSES] += n - row[i]
        rows.append(row)
    return rows


def write_run(runs_dir: Path, *, block: str, arm: str, model: str, seed: int,
              macro_f1: float, attack: str | None = None, defense: str | None = None,
              alpha: float = 0.5, sigma: float | None = None, tau: float | None = None,
              coding: str = "delta", t_steps: int = 8, mu: float | None = None,
              m: int = 10, f: int = 3, rounds: int = 4, energy: bool = False,
              attackers: tuple[int, ...] = (7, 8, 9), k: int = 10,
              agg: str = "krum", suffix: str = "", config_hash: str | None = None,
              diverged: bool = False, energy_synops: float = 70000.0,
              scale: float | None = None, split_mode: str | None = None) -> Path:
    """One synthetic RunResult in exactly the on-disk battery schema.

    ``config_hash`` defaults to a digest of the run id, so two runs are distinct
    unless a test deliberately gives them the same hash (the real T3tau manifest
    emitted several cells in exact triplicate, and the generator deduplicates on
    this field). ``diverged`` writes the abandoned-run marker the engine leaves
    behind: ``final.diverged`` with a placeholder macro-F1 of zero.

    ``scale`` is the T4mag attacker-magnitude multiplier, written where the
    battery writes it (``attack.params.scale``); ``split_mode="temporal"`` marks
    a run of the leakage-free re-run, written into the federation block. Both
    default to absent, which is what every pre-round-3 block carries.
    """
    tags = [t for t in (f"atk{attack}" if attack else None,
                        f"sg{sigma}" if sigma is not None else None,
                        f"t{tau}" if tau is not None else None,
                        f"sc{scale}" if scale is not None else None,
                        suffix or None) if t]
    run_id = "-".join([block, arm, model, *tags, f"s{seed}"])
    defense_blk: dict[str, Any] | None = None
    if defense == "pipeline":
        defense_blk = {"kind": "pipeline", "m": m, "f": f,
                       "threshold": tau if tau is not None else 0.5,
                       "net_ms": "utility_wan", "agg": agg}
    elif defense == "fidelity_only":
        defense_blk = {"kind": "fidelity_only",
                       "threshold": tau if tau is not None else 0.5}
    cfg: dict[str, Any] = {
        "model": ({"name": model, "kwargs": {}, "coding": coding, "t_steps": t_steps}
                  if model == "snn" else {"name": model, "kwargs": {}}),
        "federation": ({"k": k, "alpha": alpha, "days": 30, "seed": seed,
                        "iid": False, "split_mode": split_mode} if split_mode
                       else {"k": k, "alpha": alpha, "days": 30, "seed": seed,
                             "iid": False}),
        "training": {"rounds": rounds, "local_epochs": 1, "batch_size": 64,
                     "lr": 1e-3, "optimizer": "adam", "class_weighting": "effective_num"},
        "mode": "centralized" if arm == "centralized" else "federated",
        "strategy": ({"name": "fedprox", "mu": mu} if mu is not None
                     else {"name": "fedavg"}),
        "defense": defense_blk,
        "attack": ({"type": attack, "fraction": 0.3,
                    "params": ({"scale": scale} if scale is not None else {})}
                   if attack else None),
        "ldp": ({"clip": 1.0, "sigma": sigma, "seed": seed} if sigma is not None
                else None),
        "precision": "fp32",
        "seed": seed,
    }
    round_events = []
    for r in range(1, rounds + 1):
        gate = defense_blk is not None
        rejected = sorted({*attackers, 6}) if gate else []
        clients = [{
            "cid": c, "n_samples": 100, "loss": 1.0,
            "fidelity": (0.9 if c in attackers else 0.4) if gate else 0.0,
            "accepted": c not in rejected,
            "attack": attack if (attack and c in attackers) else None,
        } for c in range(k)]
        round_events.append({
            "v": 1, "run_id": run_id, "round": r, "clients": clients,
            "global": {"loss": 1.0, "acc": 0.9,
                       "macro_f1": macro_f1 * r / rounds},
            "consensus": ({"n_messages": 2 * m * (m - 1), "latency_ms": 120.0 + r,
                           "rejected": rejected} if gate
                          else {"n_messages": 0, "latency_ms": 0.0, "rejected": []}),
            "energy": {"synops": 1.0e6, "ratio": 0.05},
            "ts": "2026-07-28T00:00:00Z",
        })
    payload: dict[str, Any] = {
        "method": arm,
        "seed": seed,
        "config": {"engine": {"rounds": rounds, "strategy": arm},
                   "manifest": {"arm": arm, "block": block, "config": cfg,
                                "config_hash": config_hash or hashlib.sha256(
                                    run_id.encode()).hexdigest(),
                                "model": model,
                                "run_id": run_id, "seed": seed,
                                "tier": 1 if block.startswith("t1") else 2}},
        "rounds": round_events,
        "final": ({"acc": 0.0, "macro_f1": 0.0, "diverged": True,
                   "note": "diverged (NaN fidelity score) under strong attack"}
                  if diverged else
                  {"acc": 0.9, "macro_f1": macro_f1, "loss": 1.0, "n_test": _N_TEST,
                   "split": "global_test", "confusion": _confusion(macro_f1),
                   "per_class_f1": [round(macro_f1, 4)] * _N_CLASSES,
                   "correct": [(i + seed) % 3 != 0 for i in range(_N_TEST)]}),
        "energy": ({"synops_per_inference": energy_synops, "spike_rate": 0.11,
                    "ann_cnn_lstm_macs": 2014208.0, "snn_uJ": 0.063,
                    "ann_uJ": 9.265, "ratio": 0.0068,
                    "snn_dense_macs_per_inference": 901120.0,
                    "snn_internal_ratio": 0.076, "n_samples": _N_TEST}
                   if energy else None),
        "latency": None,
        "git_commit": "0" * 40,
        "env_fingerprint": "py3.14 | torch test | cuda none | cpu",
    }
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def populated_runs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A partial battery: T1a complete for two arms, T1b partial, nothing else."""
    runs = tmp_path_factory.mktemp("runs_partial")
    for seed in range(5):
        for model, base in (("cnn_lstm", 0.60), ("snn", 0.16)):
            write_run(runs, block="t1a", arm="fedavg", model=model, seed=seed,
                      macro_f1=base + 0.01 * seed, energy=(model == "snn"))
            write_run(runs, block="t1a", arm="centralized", model=model, seed=seed,
                      macro_f1=base + 0.05 + 0.01 * seed, energy=(model == "snn"))
            write_run(runs, block="t1a", arm="pipeline_krum", model=model, seed=seed,
                      macro_f1=base - 0.10 + 0.01 * seed, defense="pipeline")
            write_run(runs, block="t1b", arm="no_defense", model=model, seed=seed,
                      macro_f1=base - 0.20 + 0.01 * seed, attack="sign_flip")
            write_run(runs, block="t1b", arm="pipeline_krum", model=model, seed=seed,
                      macro_f1=base - 0.05 + 0.01 * seed, attack="sign_flip",
                      defense="pipeline")
    return runs


@pytest.fixture(scope="module")
def built(populated_runs: Path, tmp_path_factory: pytest.TempPathFactory
          ) -> tuple[Path, dict[str, Any]]:
    tables = tmp_path_factory.mktemp("tables")
    summary = R.write_report(populated_runs, tables, data_dir=None, resamples=200)
    return tables, summary


# --------------------------------------------------------------------------- #
# Helpers over a generated file.
# --------------------------------------------------------------------------- #


def _colspec(text: str) -> str:
    m = re.search(r"\\begin\{tabular\}\{@\{\}(.*?)@\{\}\}", text, re.S)
    assert m, "generated table has no @{}...@{} tabular preamble"
    return " ".join(m.group(1).split())


def _rows(text: str) -> list[str]:
    """Every ``\\\\``-terminated row between \\toprule and \\bottomrule."""
    body = text.split("\\toprule", 1)[1].split("\\bottomrule", 1)[0]
    out = []
    for line in body.splitlines():
        line = line.strip()
        if line.endswith("\\\\"):
            out.append(line[:-2].strip())
    return out


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.split("&")]


def _labelled_block(text: str, label: str) -> list[list[str]]:
    """The rows of one label-spanning block.

    A row label (arm, scale, defense rung) is printed on the block's *first* row
    only, so a lookup keyed on column 0 finds one row of three and a lookup
    keyed on the model finds the last block's row rather than this one's.
    """
    out: list[list[str]] = []
    started = False
    for row in _rows(text)[1:]:
        cells = _cells(row)
        if cells[0] == label:
            started = True
        elif cells[0] and started:
            break
        if started:
            out.append(cells)
    assert out, f"no block labelled {label!r}"
    return out


# --------------------------------------------------------------------------- #
# 1. All twenty-one tables, matching the stub contract.
# --------------------------------------------------------------------------- #


def test_all_contracted_tables_are_written(built: tuple[Path, dict[str, Any]]) -> None:
    tables, _ = built
    written = {p.name for p in tables.glob("*.tex")}
    assert written == set(STUB_SCHEMA), (
        f"missing {set(STUB_SCHEMA) - written}, unexpected {written - set(STUB_SCHEMA)}")
    # Sixteen until 2026-07-30; tab_tau_sweep and tab_ldp_tau_extended gave the
    # 540-run t3tau block its first reader. Twenty-one on 2026-07-31, when round
    # 3 landed the leakage-free *_temporal re-run of Tier 1 and the t4mag
    # attacker-magnitude sweep.
    assert len(STUB_SCHEMA) == 21


@pytest.mark.parametrize("name", sorted(STUB_SCHEMA))
def test_table_matches_stub_schema(built: tuple[Path, dict[str, Any]],
                                   name: str) -> None:
    tables, _ = built
    text = (tables / name).read_text(encoding="utf-8")
    colspec, header = STUB_SCHEMA[name]
    assert _colspec(text) == colspec, f"{name}: column specification drifted"
    assert _cells(_rows(text)[0]) == list(header), f"{name}: header cells drifted"


@pytest.mark.parametrize("name", sorted(STUB_SCHEMA))
def test_table_is_a_body_not_a_float(built: tuple[Path, dict[str, Any]],
                                     name: str) -> None:
    """The float and the caption live in the section files; regeneration must not
    touch prose, so a generated file carries neither."""
    text = (tables_of(built) / name).read_text(encoding="utf-8")
    assert "\\begin{table" not in text
    assert "\\caption" not in text
    assert "\\label" not in text
    assert text.count("\\begin{tabular}") == 1
    assert text.count("\\end{tabular}") == 1
    assert "\\toprule" in text and "\\bottomrule" in text


def tables_of(built: tuple[Path, dict[str, Any]]) -> Path:
    return built[0]


@pytest.mark.parametrize("name", sorted(STUB_SCHEMA))
def test_every_row_has_the_header_column_count(built: tuple[Path, dict[str, Any]],
                                               name: str) -> None:
    text = (tables_of(built) / name).read_text(encoding="utf-8")
    n = len(STUB_SCHEMA[name][1])
    for row in _rows(text):
        assert len(_cells(row)) == n, f"{name}: row has {len(_cells(row))} cells: {row}"


@pytest.mark.parametrize("name", sorted(STUB_ROW_LABELS))
def test_row_labels_survive_regeneration(built: tuple[Path, dict[str, Any]],
                                         name: str) -> None:
    text = (tables_of(built) / name).read_text(encoding="utf-8")
    for label in STUB_ROW_LABELS[name]:
        assert label in text, f"{name}: lost row label {label!r}"


# --------------------------------------------------------------------------- #
# 2. Missing data renders as a loud marker, never a zero.
# --------------------------------------------------------------------------- #


#: Leading columns of each table that are row *labels* (client id, sigma, tau,
#: committee size ...) rather than results; they are numeric by design.
LABEL_COLUMNS: dict[str, int] = {
    "tab_dataset.tex": 2, "tab_nonconvergence.tex": 1, "tab_reference_frame.tex": 3,
    "tab_attack_defense.tex": 2, "tab_defense_decomposition.tex": 2,
    "tab_adaptive_evasion.tex": 3, "tab_ldp_curve.tex": 2, "tab_alpha_sweep.tex": 2,
    # tab_latency lost its Precision column, so only Model and Device are labels.
    "tab_consensus_overhead.tex": 2, "tab_energy.tex": 3, "tab_latency.tex": 2,
    "tab_per_class.tex": 1, "tab_coding_tsteps.tex": 2, "tab_fedprox_mu.tex": 2,
    "tab_stats.tex": 3, "tab_hypotheses.tex": 3,
    "tab_tau_sweep.tex": 2, "tab_ldp_tau_extended.tex": 2,
    "tab_temporal_frame.tex": 2, "tab_temporal_attack.tex": 2,
    "tab_magnitude.tex": 2,
}


def _value_cells(name: str, row: str) -> list[str]:
    """The result cells of a row, with the label columns stripped off."""
    return _cells(row)[LABEL_COLUMNS[name]:]


def test_missing_arms_render_todonum_not_zeros(built: tuple[Path, dict[str, Any]]
                                               ) -> None:
    tables, _ = built
    # Nothing in the fixture populates the LDP grid or the latency measurement.
    for name in ("tab_ldp_curve.tex", "tab_latency.tex"):
        text = (tables / name).read_text(encoding="utf-8")
        for row in _rows(text)[1:]:
            numeric = [c for c in _value_cells(name, row)
                       if re.fullmatch(r"[-+]?\d+(\.\d+)?", c)]
            assert not numeric, f"{name}: fabricated numeric cell(s) {numeric} in {row}"
            assert "\\TODOnum" in row, f"{name}: empty row without a marker: {row}"


def test_absent_cell_is_a_marker_where_a_present_cell_is_a_number(
        built: tuple[Path, dict[str, Any]]) -> None:
    """Same column, two arms: the one with runs shows a value, the other a marker.

    The two kinds of empty cell are distinct and must not be confused. A cell the
    battery *could* fill but has not yet run renders ``\\TODOnum``; a cell that no
    battery can ever fill renders ``--``. Both are checked here, because a table
    that printed ``--`` for a merely-missing run would quietly retire the hole.
    """
    text = (tables_of(built) / "tab_attack_defense.tex").read_text(encoding="utf-8")
    # Each arm spans three model rows; the arm label sits on the first (CNN-LSTM).
    rows = {_cells(r)[0]: _cells(r) for r in _rows(text) if _cells(r)[0]}
    # Sign-flip column: no-defense has runs, krum-alone does not.
    assert re.match(r"0\.\d+", rows["No defense"][3]), "sign-flip cell should be a value"
    assert rows["Krum alone"][3].startswith("\\TODOnum")
    # Clean column: "krum alone" is defined only under attack, so a clean
    # krum-alone cell is structurally inapplicable rather than merely unrun.
    assert rows["Krum alone"][2] == "--"
    assert rows["No defense"][2] != "--"  # ...whereas clean no-defense is a real arm


def test_empty_runs_dir_still_writes_every_table_with_no_invented_values(
        tmp_path: Path) -> None:
    runs = tmp_path / "empty"
    runs.mkdir()
    tables = tmp_path / "tables_empty"
    summary = R.write_report(runs, tables, data_dir=None, resamples=50)
    assert {p.name for p in tables.glob("*.tex")} == set(STUB_SCHEMA)
    assert summary["meta"]["n_runs"] == 0
    for name in STUB_SCHEMA:
        text = (tables / name).read_text(encoding="utf-8")
        for row in _rows(text)[1:]:
            for cell in _value_cells(name, row):
                assert not re.fullmatch(r"[-+$\\]*0(\.0+)?", cell), (
                    f"{name}: silent zero in {row}")


def test_todo_count_falls_as_runs_land(tmp_path: Path) -> None:
    empty = tmp_path / "runs_none"
    empty.mkdir()
    few = tmp_path / "runs_some"
    for seed in range(3):
        for model, base in (("cnn_lstm", 0.6), ("snn", 0.2)):
            write_run(few, block="t1a", arm="fedavg", model=model, seed=seed,
                      macro_f1=base, energy=(model == "snn"))
            write_run(few, block="t1a", arm="centralized", model=model, seed=seed,
                      macro_f1=base + 0.05)
    before = R.write_report(empty, tmp_path / "t0", data_dir=None, resamples=50)
    after = R.write_report(few, tmp_path / "t1", data_dir=None, resamples=50)
    assert after["tables"]["todo_total"] < before["tables"]["todo_total"]


def test_short_seed_counts_are_marked(tmp_path: Path) -> None:
    """A cell built from fewer seeds than the block plans carries its realised n."""
    runs = tmp_path / "runs_short"
    for seed in range(2):  # T1a plans five
        write_run(runs, block="t1a", arm="fedavg", model="cnn_lstm", seed=seed,
                  macro_f1=0.5 + 0.01 * seed)
    tables = tmp_path / "tables_short"
    R.write_report(runs, tables, data_dir=None, resamples=50)
    text = (tables / "tab_reference_frame.tex").read_text(encoding="utf-8")
    assert "$^{2}$" in text, "under-powered cell is not flagged with its realised n"


# --------------------------------------------------------------------------- #
# 3. Real values actually reach the tables.
# --------------------------------------------------------------------------- #


def test_planted_values_reach_the_reference_frame(built: tuple[Path, dict[str, Any]]
                                                  ) -> None:
    text = (tables_of(built) / "tab_reference_frame.tex").read_text(encoding="utf-8")
    # FedAvg CNN-LSTM was planted at 0.60..0.64 -> mean 0.620.
    assert "0.620" in text
    # Centralized ceiling at 0.65..0.69 -> mean 0.670.
    assert "0.670" in text


def test_energy_table_recomputes_the_int8_rows(built: tuple[Path, dict[str, Any]]
                                               ) -> None:
    from fedgrid.metrics.energy import E_AC_INT8_PJ

    text = (tables_of(built) / "tab_energy.tex").read_text(encoding="utf-8")
    expected = R.sig(70000.0 * E_AC_INT8_PJ * 1e-6, 3)
    assert expected in text, "INT8 spiking energy not recomputed from the counters"
    assert "81\\,315" not in text  # sanity: not the fp32 SynOps in an INT8 slot


def test_gate_statistics_come_from_the_consensus_partition(
        built: tuple[Path, dict[str, Any]]) -> None:
    """Honest FRR / attacker admission are measured, not assumed."""
    text = (tables_of(built) / "tab_adaptive_evasion.tex").read_text(encoding="utf-8")
    rows = [_cells(r) for r in _rows(text)]
    krum_signflip = next(r for r in rows if r[1] == "Sign-flip" and r[2] == "CNN-LSTM")
    # The fixture rejects the three attackers plus honest client 6 every round.
    assert krum_signflip[4] == "0.000", "attacker admission should be zero"
    assert krum_signflip[5] == "0.143", "honest FRR should be 1 of 7 honest clients"


def test_no_gate_arm_reports_structural_dashes_not_constants(
        built: tuple[Path, dict[str, Any]]) -> None:
    """An arm with no admission gate has no *measured* admission rate.

    "adm. 1.000 / FRR 0.000" on an ungated rung is that arm's definition
    restated, not an observation, and printed as a number it invites comparison
    with the measured rates one row down. Structural, so ``--``.
    """
    text = (tables_of(built) / "tab_defense_decomposition.tex").read_text(
        encoding="utf-8")
    no_defense = next(_cells(r) for r in _rows(text) if _cells(r)[0] == "No defense")
    assert no_defense[4] == "--", "ungated arm must not print a constant admission"
    assert no_defense[5] == "--", "ungated arm must not print a constant FRR"
    assert no_defense[6] == "--", "an arm with no ledger must read as inapplicable"


def test_ledger_column_claims_no_verification_the_battery_never_ran(
        built: tuple[Path, dict[str, Any]]) -> None:
    """``Ledger.verify()`` is never called in the battery, so nothing may tick.

    The pipeline arms do build a hash chain, but no RunResult records a
    verification outcome; a hard-coded checkmark asserted an end-to-end result
    that does not exist in the data.
    """
    text = (tables_of(built) / "tab_defense_decomposition.tex").read_text(
        encoding="utf-8")
    assert "\\checkmark" not in text, (
        "the ledger column asserts a verification the RunResults do not carry")
    for row in _rows(text)[1:]:
        assert _cells(row)[6] == "--"


def test_energy_rate_row_is_a_single_unroll_length(tmp_path: Path) -> None:
    """SynOps scale with T, so a rate row pooled over T describes no model.

    The T2e coding ablation runs rate coding at T in {4, 8, 16}. Averaging their
    SynOps produced a row (999,481 in the shipped battery) that belongs to none
    of the three trained models and whose energy ratio was ~17% too high.
    """
    runs = tmp_path / "runs_energy_T"
    for t_steps, synops in ((4, 400_000.0), (8, 800_000.0), (16, 1_600_000.0)):
        for seed in range(3):
            write_run(runs, block="t2e", arm="fedavg", model="snn", seed=seed,
                      macro_f1=0.2, coding="rate", t_steps=t_steps, energy=True,
                      energy_synops=synops, suffix=f"ts{t_steps}")
    tables = tmp_path / "tables_energy_T"
    R.write_report(runs, tables, data_dir=None, resamples=50)
    text = (tables / "tab_energy.tex").read_text(encoding="utf-8")
    rate = next(_cells(r) for r in _rows(text)
                if _cells(r)[0] == "Spiking" and _cells(r)[1].startswith("rate"))
    assert rate[3] == R.thousands(800_000.0), (
        f"rate row is not the T=8 model: {rate[3]}")
    assert R.thousands(933_333.0) not in text, "the mean over T leaked into a cell"
    assert "T{=}8" in rate[1], "the row does not declare which T it is"


def test_diverged_runs_are_marked_not_averaged_as_zero(tmp_path: Path) -> None:
    """A divergence placeholder of 0.0 is a marker; no mean may swallow it."""
    runs = tmp_path / "runs_div"
    for seed in range(3):
        write_run(runs, block="t3tau", arm="pipeline_krum", model="ssm", seed=seed,
                  macro_f1=0.30, attack="adaptive_cosine", defense="pipeline",
                  sigma=1.0, tau=0.8, diverged=(seed == 0))
    (row_ok, *_rest) = [r for r in R.load_rows(runs) if not r.diverged]
    assert row_ok.macro_f1 == 0.30
    assert sum(1 for r in R.load_rows(runs) if r.diverged) == 1
    tables = tmp_path / "tables_div"
    summary = R.write_report(runs, tables, data_dir=None, resamples=50)
    text = (tables / "tab_ldp_tau_extended.tex").read_text(encoding="utf-8")
    cell = next(_cells(r)[7] for r in _rows(text)[1:]
                if _cells(r)[1] == "1.0" and "\\TODOnum" not in _cells(r)[7])
    assert cell.startswith("0.300"), f"the diverged zero dragged the mean: {cell}"
    assert "div." in cell, "the divergence is not disclosed in the cell"
    assert summary["block_inventory"]["t3tau"]["n_diverged"] == 1


def test_extended_ldp_grid_deduplicates_the_triplicated_manifest(tmp_path: Path
                                                                 ) -> None:
    """T3tau emitted each LDP cell once per source T2c threshold; n must not treble."""
    runs = tmp_path / "runs_dupe"
    for source_tau in ("t0p3", "t0p5", "t0p7"):
        for seed in range(3):
            write_run(runs, block="t3tau", arm="pipeline_krum", model="cnn_lstm",
                      seed=seed, macro_f1=0.05, attack="adaptive_cosine",
                      defense="pipeline", sigma=0.25, tau=0.8, suffix=source_tau,
                      config_hash=f"{seed:064d}")
    assert len(R.load_rows(runs)) == 9, "fixture should plant nine files"
    tables = tmp_path / "tables_dupe"
    summary = R.write_report(runs, tables, data_dir=None, resamples=50)
    assert summary["block_inventory"]["t3tau"]["n_runs"] == 9
    assert summary["block_inventory"]["t3tau"]["n_distinct_configs"] == 3
    text = (tables / "tab_ldp_tau_extended.tex").read_text(encoding="utf-8")
    cell = next(_cells(r)[5] for r in _rows(text)
                if _cells(r)[0] == "0.8" and _cells(r)[1] == "0.25")
    # Three distinct seeds is the block's plan, so no short-n superscript at all;
    # counting the nine files would have marked the cell as over-complete instead.
    assert cell == "0.050", f"deduplication did not happen: {cell}"


# --------------------------------------------------------------------------- #
# 3b. Round 3: the leakage-free frame and the magnitude sweep.
# --------------------------------------------------------------------------- #


def test_temporal_frame_reads_one_block_per_column(tmp_path: Path) -> None:
    """The whole point of the table is that the two columns are two blocks.

    Every table above ``tab_temporal_frame`` reads T1a, whose windows were split
    at random; the temporal column must come from ``t1a_temporal`` and from
    nowhere else, or the table silently reports the leakage it exists to remove.
    """
    runs = tmp_path / "runs_temporal"
    for seed in range(5):
        write_run(runs, block="t1a", arm="fedavg", model="cnn_lstm", seed=seed,
                  macro_f1=0.60)
        write_run(runs, block="t1a_temporal", arm="fedavg", model="cnn_lstm",
                  seed=seed, macro_f1=0.50, split_mode="temporal", suffix="tmp")
    tables = tmp_path / "tables_temporal"
    R.write_report(runs, tables, data_dir=None, resamples=50)
    text = (tables / "tab_temporal_frame.tex").read_text(encoding="utf-8")
    row = next(_cells(r) for r in _rows(text) if _cells(r)[1] == "CNN-LSTM"
               and _cells(r)[0] == "FedAvg")
    assert row[2].startswith("0.600"), f"random column is not T1a: {row[2]}"
    assert row[3].startswith("0.500"), f"temporal column is not T1a_temporal: {row[3]}"
    # Delta is the difference of the two printed columns, so it reconciles on the
    # page; a reader checking 0.500 - 0.600 must land on exactly this cell.
    assert row[4] == "$-$0.100", f"delta does not reconcile: {row[4]}"
    assert row[5] == "$-$16.7", f"relative delta is wrong: {row[5]}"
    # The temporal runs must not leak back into the random-split reference frame.
    frame = (tables / "tab_reference_frame.tex").read_text(encoding="utf-8")
    fedavg = next(_cells(r) for r in _rows(frame) if _cells(r)[1] == "FedAvg")
    assert fedavg[3].startswith("0.600"), (
        f"t1a_temporal contaminated the random-split frame: {fedavg[3]}")


def test_a_delta_that_rounds_to_zero_prints_unsigned(tmp_path: Path) -> None:
    """``-0.000`` reads as a typesetting fault, not as a direction."""
    runs = tmp_path / "runs_zero_delta"
    for seed in range(5):
        write_run(runs, block="t1a", arm="fedavg", model="snn", seed=seed,
                  macro_f1=0.2000)
        write_run(runs, block="t1a_temporal", arm="fedavg", model="snn", seed=seed,
                  macro_f1=0.2001, split_mode="temporal", suffix="tmp")
    tables = tmp_path / "tables_zero_delta"
    R.write_report(runs, tables, data_dir=None, resamples=50)
    text = (tables / "tab_temporal_frame.tex").read_text(encoding="utf-8")
    assert "$-$0.000" not in text and "$+$0.000" not in text
    row = next(c for c in _labelled_block(text, "FedAvg") if c[1] == "Spiking")
    assert row[4] == "0.000", f"a rounded-zero delta kept its sign: {row[4]}"


def test_magnitude_table_separates_the_attacker_scales(tmp_path: Path) -> None:
    """The sweep's axis is ``attack.params.scale``; pooling it reports no setting.

    Nothing else in the battery varies the Byzantine update magnitude, so the
    axis has no other carrier: read it wrong and all four rows become the same
    average of four different attacks.
    """
    runs = tmp_path / "runs_mag"
    planted = {1.0: 0.60, 2.0: 0.50, 5.0: 0.20, 10.0: 0.10}
    for scale, f1 in planted.items():
        for seed in range(3):
            write_run(runs, block="t4mag", arm="no_defense", model="cnn_lstm",
                      seed=seed, macro_f1=f1, attack="adaptive_cosine", scale=scale)
            write_run(runs, block="t4mag", arm="pipeline_krum", model="cnn_lstm",
                      seed=seed, macro_f1=f1 / 2, attack="adaptive_cosine",
                      defense="pipeline", scale=scale)
    assert {r.atk_scale for r in R.load_rows(runs)} == set(planted)
    tables = tmp_path / "tables_mag"
    R.write_report(runs, tables, data_dir=None, resamples=50)
    text = (tables / "tab_magnitude.tex").read_text(encoding="utf-8")
    for scale, f1 in planted.items():
        block = _labelled_block(text, f"{scale:.0f}$\\times$")
        no_def = next(c for c in block if c[1] == "No defense")
        assert no_def[2].startswith(f"{f1:.3f}"), (
            f"scale {scale} pooled with its neighbours: {no_def[2]}")


def test_ungated_arms_in_the_magnitude_table_report_dashes(tmp_path: Path) -> None:
    """An arm with no admission gate has no measured admission, at any magnitude.

    Same rule as ``tab_defense_decomposition``: 1.000/0.000 on an ungated arm is
    that arm's definition restated, and printed as a number it invites the reader
    to compare it against the measured rates two rows down.
    """
    runs = tmp_path / "runs_mag_gate"
    for seed in range(3):
        for arm, defense in (("no_defense", None), ("geomed_alone", None),
                             ("pipeline_krum", "pipeline")):
            write_run(runs, block="t4mag", arm=arm, model="cnn_lstm", seed=seed,
                      macro_f1=0.3, attack="adaptive_cosine", defense=defense,
                      scale=5.0)
    tables = tmp_path / "tables_mag_gate"
    R.write_report(runs, tables, data_dir=None, resamples=50)
    text = (tables / "tab_magnitude.tex").read_text(encoding="utf-8")
    # Scoped to the block the fixture populates: an *empty* row also reads `--',
    # so a lookup that fell through to an unplanted scale would pass vacuously.
    rows = {c[1]: c for c in _labelled_block(text, "5$\\times$")}
    for arm in ("No defense", "Geo-median alone"):
        assert rows[arm][5:] == ["--", "--", "--"], f"{arm} printed a gate constant"
    # ...whereas the gated arm reports what the consensus partition actually did:
    # the fixture rejects the three attackers plus honest client 6 every round.
    # The gate pools the three models, so a one-model fixture keeps its 3/9
    # short-pool superscript -- that disclosure is the point, hence startswith.
    assert rows["Pipeline + Krum"][6].startswith("0.000"), "admission not measured"
    assert rows["Pipeline + Krum"][7].startswith("0.143"), "honest FRR not measured"


def test_temporal_attack_cell_carries_its_random_split_delta(tmp_path: Path) -> None:
    """Each cell is the temporal mean with the change against its T1b twin."""
    runs = tmp_path / "runs_temporal_atk"
    for seed in range(5):
        write_run(runs, block="t1b", arm="no_defense", model="ssm", seed=seed,
                  macro_f1=0.40, attack="gaussian_noise")
        write_run(runs, block="t1b_temporal", arm="no_defense", model="ssm",
                  seed=seed, macro_f1=0.25, attack="gaussian_noise",
                  split_mode="temporal", suffix="tmp")
    tables = tmp_path / "tables_temporal_atk"
    R.write_report(runs, tables, data_dir=None, resamples=50)
    text = (tables / "tab_temporal_attack.tex").read_text(encoding="utf-8")
    row = next(_cells(r) for r in _rows(text) if _cells(r)[1] == "Mamba"
               and "\\TODOnum" not in _cells(r)[4])
    assert row[4] == "0.250\\,($-$0.150)", f"cell lost its random-split delta: {row[4]}"
    # An arm evaluated under attack only keeps its structural dash in the clean
    # column here exactly as it does in the random-split table.
    krum = next(_cells(r) for r in _rows(text) if _cells(r)[0] == "Krum alone")
    assert krum[2] == "--"


# --------------------------------------------------------------------------- #
# 4. summary.json.
# --------------------------------------------------------------------------- #


def test_summary_json_is_written_and_round_trips(populated_runs: Path,
                                                 tmp_path: Path) -> None:
    tables = tmp_path / "tables_rt"
    target = tmp_path / "nested" / "summary.json"
    summary = R.write_report(populated_runs, tables, data_dir=None, resamples=100,
                             summary_path=target)
    assert target.exists()
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(R._jsonable(summary), sort_keys=True))
    assert json.loads(json.dumps(on_disk)) == on_disk  # nothing unserialisable
    for key in ("meta", "coverage", "per_arm", "hypotheses", "tables"):
        assert key in on_disk
    assert on_disk["meta"]["n_runs"] == 50
    assert on_disk["tables"]["todo_total"] == sum(
        on_disk["tables"]["todo_markers"].values())
    assert set(on_disk["hypotheses"]["verdicts"]) >= {"H1a", "H1b", "H2", "H5"}


def test_summary_defaults_into_the_runs_dir(tmp_path: Path) -> None:
    runs = tmp_path / "runs_default"
    write_run(runs, block="t1a", arm="fedavg", model="snn", seed=0, macro_f1=0.2)
    R.write_report(runs, tmp_path / "tab_default", data_dir=None, resamples=50)
    assert (runs / "summary.json").exists()


def test_summary_json_is_not_mistaken_for_a_run(tmp_path: Path) -> None:
    """Writing summary.json into runs/ must not add a phantom run on reload."""
    runs = tmp_path / "runs_reload"
    write_run(runs, block="t1a", arm="fedavg", model="snn", seed=0, macro_f1=0.2)
    R.write_report(runs, tmp_path / "tab_reload", data_dir=None, resamples=50)
    assert len(R.load_rows(runs)) == 1


def test_undecidable_hypothesis_has_no_verdict(built: tuple[Path, dict[str, Any]]
                                               ) -> None:
    """A family with no runs must read as a hole, not as 'not supported'."""
    _tables, summary = built
    assert summary["hypotheses"]["verdicts"]["H4"]["supported"] is None
    text = (tables_of(built) / "tab_hypotheses.tex").read_text(encoding="utf-8")
    h4 = next(_cells(r) for r in _rows(text) if _cells(r)[0] == "H4")
    assert h4[3].startswith("\\TODOnum")


# --------------------------------------------------------------------------- #
# 4b. The pre-registered protocol, as registered.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def three_model_runs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """T1a + a full T1b defense grid over all *three* battery models."""
    runs = tmp_path_factory.mktemp("runs_three")
    bases = {"cnn_lstm": 0.60, "snn": 0.16, "ssm": 0.75}
    arms = ("no_defense", "krum_alone", "geomed_alone", "pipeline_krum",
            "pipeline_geomed")
    for seed in range(5):
        for model, base in bases.items():
            write_run(runs, block="t1a", arm="fedavg", model=model, seed=seed,
                      macro_f1=base + 0.01 * seed, energy=(model == "snn"))
            for offset, arm in enumerate(arms):
                write_run(runs, block="t1a", arm=arm, model=model, seed=seed,
                          macro_f1=base - 0.10 + 0.01 * seed,
                          defense="pipeline" if arm.startswith("pipeline") else None)
                for attack in ("sign_flip", "gaussian_noise", "label_flip",
                               "adaptive_cosine"):
                    write_run(runs, block="t1b", arm=arm, model=model, seed=seed,
                              macro_f1=base - 0.20 + 0.01 * offset + 0.005 * seed,
                              attack=attack,
                              defense="pipeline" if arm.startswith("pipeline")
                              else None)
    return runs


def test_every_hypothesis_family_counts_all_three_models(three_model_runs: Path,
                                                         tmp_path: Path) -> None:
    """The Mamba is a battery model, so it is a member of every family.

    H2 iterated only CNN-LSTM and Spiking, which under-counted the family at 16
    cells instead of 24 and quietly dropped a third of the evidence.
    """
    summary = R.write_report(three_model_runs, tmp_path / "tab_3m", data_dir=None,
                             resamples=50)
    h = summary["hypotheses"]
    assert len(h["h2_wilcoxon"]) == 24, "H2 is 2 pipelines x 3 models x 4 attacks"
    assert len(h["h3"]["evasion"]) == 6 and len(h["h3"]["damage"]) == 6
    for family in (h["h2_wilcoxon"], h["h3"]["evasion"], h["h3"]["damage"],
                   h["h5"]["trend"], h["h5"]["differential"]):
        assert any("Mamba" in c["comparison"] for c in family), (
            f"the Mamba never reached {family[0]['family']}")
    assert len(h["h5"]["trend"]) == 6 and len(h["h5"]["differential"]) == 3


def test_friedman_blocks_are_the_preregistered_seed_means(three_model_runs: Path,
                                                          tmp_path: Path) -> None:
    """HYPOTHESES.md fixes blocks = (attack x model) seed-means, not seed rows.

    Blocking per seed row multiplies the block count by the seed count, which
    treats five repetitions of one cell as five independent blocks: Q inflates and
    the Nemenyi critical difference shrinks by sqrt(5). The per-seed variant is
    kept, explicitly labelled a sensitivity analysis.
    """
    summary = R.write_report(three_model_runs, tmp_path / "tab_fr", data_dir=None,
                             resamples=50)
    fr = summary["hypotheses"]["h2_friedman"]
    assert fr["unit"] == "seed_mean"
    assert fr["n"] == 12, "3 models x 4 attacks = 12 pre-registered blocks"
    sens = fr["sensitivity"]
    assert sens["unit"] == "seed" and sens["n"] == 60
    # The pseudo-replicated variant is the one with the smaller critical
    # difference; that is exactly why it must not be the headline.
    assert sens["cd"] < fr["cd"]
    text = (tmp_path / "tab_fr" / "tab_stats.tex").read_text(encoding="utf-8")
    assert "[pre-registered]" in text and "[sensitivity]" in text


def test_evidence_string_says_support_over_decided(three_model_runs: Path,
                                                   tmp_path: Path) -> None:
    """"0/16 cells decided (16 planned)" read as if 16 cells were undecided.

    The H2 family is power-limited at this seed budget, so the cell must also
    carry the undecidability clause: Section 7 calls the family undecidable, and
    a bare "24/24 cells decided" beside that prose reads as a contradiction.
    """
    R.write_report(three_model_runs, tmp_path / "tab_ev", data_dir=None, resamples=50)
    text = (tmp_path / "tab_ev" / "tab_hypotheses.tex").read_text(encoding="utf-8")
    h2 = next(_cells(r) for r in _rows(text) if _cells(r)[0] == "H2")
    assert re.match(r"\d+/24 support \(24/24 cells decided[;)]", h2[2]), h2[2]
    assert "family undecidable at this seed budget" in h2[2], h2[2]
    # H5a is not power-limited, so it must NOT claim undecidability.
    h5a = next(_cells(r) for r in _rows(text) if _cells(r)[0] == "H5a")
    assert "undecidable" not in h5a[2], h5a[2]


def test_power_ceiling_is_declared_where_holm_cannot_reach_alpha(
        three_model_runs: Path, tmp_path: Path) -> None:
    """An n=5 one-sided Wilcoxon in a 24-cell Holm family floors at 0.75.

    No arrangement of the data could have made a single H2 cell significant, so
    the tables must say so rather than let "not supported" read as a refutation.
    """
    summary = R.write_report(three_model_runs, tmp_path / "tab_pc", data_dir=None,
                             resamples=50)
    ceiling = R.power_ceiling(
        [R.StatRow(**c) for c in summary["hypotheses"]["h2_wilcoxon"]])
    assert ceiling is not None
    assert ceiling["m"] == 24 and ceiling["n_max"] == 5
    assert ceiling["p_min"] == pytest.approx(0.03125)
    assert ceiling["holm_floor"] == pytest.approx(0.75)
    assert ceiling["decidable"] is False
    for name in ("tab_hypotheses.tex", "tab_stats.tex"):
        text = (tmp_path / "tab_pc" / name).read_text(encoding="utf-8")
        assert "Power ceiling" in text, f"{name} does not disclose the floor"


def test_stats_table_shows_both_ends_of_a_multi_cell_family(three_model_runs: Path,
                                                            tmp_path: Path) -> None:
    """Summarising only the worst cell hides every cell that went the right way."""
    R.write_report(three_model_runs, tmp_path / "tab_be", data_dir=None, resamples=50)
    text = (tmp_path / "tab_be" / "tab_stats.tex").read_text(encoding="utf-8")
    assert "[worst cell]" in text and "[best cell]" in text


def test_fedprox_delta_reconciles_with_its_own_paired_baseline(tmp_path: Path
                                                               ) -> None:
    """Delta is paired over shared seeds; the subtrahend must be on the page.

    T2e runs three seeds and T1a five, so the Delta column never equalled the
    difference of the two absolute macro-F1 columns beside it -- six of nine
    cells did not reconcile until the paired baseline got a column.
    """
    runs = tmp_path / "runs_fedprox"
    for seed in range(5):
        write_run(runs, block="t1a", arm="fedavg", model="cnn_lstm", seed=seed,
                  macro_f1=0.50 + 0.02 * seed)          # 5 seeds: mean 0.54
    for seed in range(3):
        write_run(runs, block="t2e", arm="fedprox", model="cnn_lstm", seed=seed,
                  macro_f1=0.60, mu=0.1)                 # 3 seeds: mean 0.60
    tables = tmp_path / "tables_fedprox"
    R.write_report(runs, tables, data_dir=None, resamples=50)
    text = (tables / "tab_fedprox_mu.tex").read_text(encoding="utf-8")
    row = next(_cells(r) for r in _rows(text) if _cells(r)[0] == "0.1")
    # Seeds 0-2 of FedAvg mean 0.52, not the 0.54 of all five.
    assert row[4].startswith("0.5200"), f"paired baseline not printed: {row[4]}"
    assert row[5] == "$+$0.080", f"delta is not 0.60 - 0.52: {row[5]}"


# --------------------------------------------------------------------------- #
# 5. Figures.
# --------------------------------------------------------------------------- #


def test_all_ten_figures_render(populated_runs: Path, tmp_path: Path) -> None:
    figures = pytest.importorskip("experiments.figures")
    out = tmp_path / "figures"
    status = figures.render_all(populated_runs, out, data_dir=None, resamples=50)
    assert set(status) == set(figures.FIGURES)
    assert len(status) == 10
    for name in status:
        pdf = out / f"{name}.pdf"
        assert pdf.exists(), f"{name} did not render"
        assert pdf.read_bytes()[:5] == b"%PDF-", f"{name} is not a PDF"
        assert pdf.stat().st_size > 1000, f"{name} is suspiciously small"


def test_figure_with_no_measurement_is_a_loud_placeholder(populated_runs: Path,
                                                          tmp_path: Path) -> None:
    figures = pytest.importorskip("experiments.figures")
    out = tmp_path / "figures_ph"
    status = figures.render_all(populated_runs, out, data_dir=None, resamples=50)
    # No RunResult in the fixture carries a latency block.
    assert status["fig_latency_goose"] is False
    assert status["fig_rounds"] is True


#: Figures whose natural width in *this deliberately partial* battery is set by
#: the loud "no runs yet for ..." note rather than by the plot, and so needs a
#: wider floor here than the manuscript ever sees.
#:
#: ``_note`` centres the annotation on the figure, but a panel stack's tight box
#: is not centred on the figure -- the rotated two-line y-labels protrude to the
#: left. In ``fig_rounds`` the body occupies x in [-0.21, 2.76] in while the note,
#: centred, occupies [0.12, 2.96]; their union is 3.17 in against a 3.07 in
#: column, so ``_save`` can no longer pad the figure out to exactly one column and
#: the saved PDF is ~4% wide. The note itself fits (2.84 in) -- it is the offset
#: that overflows. Three model panels rather than two lengthened the missing-arm
#: list enough to expose it. With a complete battery no arm is missing, no note is
#: drawn, and the shipped figure is one column exactly (paper/figures/fig_rounds.pdf
#: measures 221.17 pt == COL_W). Reported upstream, not worked around here.
_NOTE_WIDENED_FLOOR: dict[str, float] = {"fig_rounds": 0.95}


def test_every_figure_is_saved_at_one_column_width(populated_runs: Path,
                                                   tmp_path: Path) -> None:
    """Uniform natural width == uniform effective type size in the manuscript."""
    figures = pytest.importorskip("experiments.figures")
    out = tmp_path / "figures_w"
    figures.render_all(populated_runs, out, data_dir=None, resamples=50)
    for name in figures.FIGURES:
        scale = figures.column_scale(out / f"{name}.pdf")
        assert scale is not None
        floor = _NOTE_WIDENED_FLOOR.get(name, 0.97)
        # Nothing may ever be *enlarged*: that is always a design error, never a
        # side effect of an annotation, so the upper bound stays tight for all ten.
        assert floor <= scale <= 1.03, f"{name} would be rescaled by {scale:.3f}x"


def test_placeholder_figures_are_exactly_one_column(tmp_path: Path) -> None:
    """With nothing to annotate, every figure is one column on the nose.

    This is the half of the width contract the manuscript actually depends on --
    the shipped figures are rendered from a complete battery, so none of them
    carries a missing-data note -- and it keeps the guard tight for the figure
    whose floor is relaxed above.
    """
    figures = pytest.importorskip("experiments.figures")
    runs = tmp_path / "runs_none"
    runs.mkdir()
    out = tmp_path / "figures_ph_w"
    figures.render_all(runs, out, data_dir=None, resamples=50)
    for name in figures.FIGURES:
        scale = figures.column_scale(out / f"{name}.pdf")
        assert scale is not None
        assert 0.97 <= scale <= 1.03, f"{name} would be rescaled by {scale:.3f}x"


def test_figures_survive_an_empty_runs_dir(tmp_path: Path) -> None:
    figures = pytest.importorskip("experiments.figures")
    runs = tmp_path / "runs_none"
    runs.mkdir()
    out = tmp_path / "figures_none"
    status = figures.render_all(runs, out, data_dir=None, resamples=50)
    assert len(status) == 10
    assert not any(status.values()), "no data must mean no figure claims data"
    for name in status:
        assert (out / f"{name}.pdf").read_bytes()[:5] == b"%PDF-"


# --------------------------------------------------------------------------- #
# 6. Loading contract.
# --------------------------------------------------------------------------- #


def test_load_rows_skips_non_runresult_json(tmp_path: Path) -> None:
    runs = tmp_path / "runs_mixed"
    write_run(runs, block="t1a", arm="fedavg", model="snn", seed=0, macro_f1=0.2)
    (runs / "fleet_state.json").write_text('{"workers": []}', encoding="utf-8")
    (runs / "broken.json").write_text("{not json", encoding="utf-8")
    rows = R.load_rows(runs)
    assert len(rows) == 1
    assert rows[0].arm == "fedavg"


def test_row_axes_are_parsed_from_the_manifest(tmp_path: Path) -> None:
    runs = tmp_path / "runs_axes"
    write_run(runs, block="t2c", arm="pipeline_krum", model="cnn_lstm", seed=1,
              macro_f1=0.3, attack="adaptive_cosine", defense="pipeline",
              sigma=0.5, tau=0.7, alpha=0.1)
    (row,) = R.load_rows(runs)
    assert (row.block, row.arm, row.model, row.seed) == ("t2c", "pipeline_krum",
                                                         "cnn_lstm", 1)
    assert (row.attack, row.sigma, row.tau, row.alpha) == ("adaptive_cosine", 0.5,
                                                           0.7, 0.1)
    assert row.has_gate and row.defense_kind == "pipeline"
    gate = row.gate_stats()
    assert gate is not None
    assert gate["attacker_admission"] == 0.0
    assert gate["n_attacker"] == 3 * len(row.rounds)


def test_gate_stats_can_skip_the_first_round(tmp_path: Path) -> None:
    runs = tmp_path / "runs_skip"
    write_run(runs, block="t1b", arm="pipeline_krum", model="snn", seed=0,
              macro_f1=0.2, attack="adaptive_cosine", defense="pipeline", rounds=4)
    (row,) = R.load_rows(runs)
    full = row.gate_stats()
    skipped = row.gate_stats(skip_first_round=True)
    assert full is not None and skipped is not None
    assert skipped["n_attacker"] == full["n_attacker"] - 3
