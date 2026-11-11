"""phase 4 data-staging resolution: one --data-dir root holding many federations.

Pure path-resolution tests for :func:`experiments.build.find_federation_dir` -- no OpenDSS,
no torch, no generated windows. Guards the seam that lets a fleet worker carry all 14 battery
federations under a single ``--data-dir`` and load each run's own without regenerating.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.build import find_federation_dir


def _write_manifest(d: Path, k: int, alpha: float, days: int, seed: int) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(
        json.dumps({"config": {"k": k, "alpha": alpha, "days": days, "seed": seed}}),
        encoding="utf-8",
    )


def _cfg(k: int, alpha: float, days: int, seed: int, iid: bool = False) -> dict:
    return {"federation": {"k": k, "alpha": alpha, "days": days, "seed": seed, "iid": iid}}


def test_single_federation_dir(tmp_path: Path) -> None:
    _write_manifest(tmp_path, 10, 0.5, 30, 0)
    assert find_federation_dir(_cfg(10, 0.5, 30, 0), tmp_path) == tmp_path


def test_multi_federation_root_picks_matching_subdir(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "fed_a", 10, 0.5, 30, 0)
    _write_manifest(tmp_path / "fed_b", 10, 0.1, 30, 1)
    _write_manifest(tmp_path / "fed_c", 10, 100.0, 30, 2)  # the IID (alpha=100) corner
    assert find_federation_dir(_cfg(10, 0.5, 30, 0), tmp_path) == tmp_path / "fed_a"
    assert find_federation_dir(_cfg(10, 0.1, 30, 1), tmp_path) == tmp_path / "fed_b"
    assert find_federation_dir(_cfg(10, 100.0, 30, 2, iid=True), tmp_path) == tmp_path / "fed_c"


def test_no_match_returns_none(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "fed_a", 10, 0.5, 30, 0)
    assert find_federation_dir(_cfg(10, 0.5, 30, 9), tmp_path) is None  # seed not staged


def test_none_data_dir_returns_none() -> None:
    assert find_federation_dir(_cfg(10, 0.5, 30, 0), None) is None


def test_alpha_is_matched_exactly(tmp_path: Path) -> None:
    # alpha 0.5 vs 1.0 must not cross-match (the sweep depends on it).
    _write_manifest(tmp_path / "fed_a", 10, 0.5, 30, 0)
    _write_manifest(tmp_path / "fed_b", 10, 1.0, 30, 0)
    assert find_federation_dir(_cfg(10, 1.0, 30, 0), tmp_path) == tmp_path / "fed_b"


def test_split_mode_resolves_to_the_same_staged_federation(tmp_path: Path) -> None:
    """The temporal battery re-cuts the SAME windows: no new data dir, no regeneration."""
    _write_manifest(tmp_path / "fed_a", 10, 0.5, 30, 0)
    cfg = _cfg(10, 0.5, 30, 0)
    cfg["federation"]["split_mode"] = "temporal"
    assert find_federation_dir(cfg, tmp_path) == tmp_path / "fed_a"


def test_stage_data_ignores_split_mode_when_identifying_a_federation(tmp_path: Path) -> None:
    """``fed_<hash>`` ids stay byte-stable: a temporal manifest stages nothing new."""
    from experiments.stage_data import distinct_federations, federation_id

    fed = {"k": 10, "alpha": 0.5, "days": 30, "seed": 0, "iid": False}
    assert federation_id({**fed, "split_mode": "temporal"}) == federation_id(fed)

    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps({"config": {"federation": fed}}) + "\n"
        + json.dumps({"config": {"federation": {**fed, "split_mode": "temporal"}}}) + "\n",
        encoding="utf-8",
    )
    feds = distinct_federations(manifest)
    assert feds == [fed]  # one federation, and split_mode is not carried into staging
