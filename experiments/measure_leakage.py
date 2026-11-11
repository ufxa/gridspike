"""Measure the train/test window-overlap leakage of the random split.

What "leakage" means here
-------------------------
Each client is a continuous simulated time series cut into sliding windows:
``WINDOW_T = 64`` one-minute samples (:data:`fedgrid.data.WINDOW_T`) taken every
``STRIDE = 16`` samples (:data:`fedgrid.simulation.partition.STRIDE`), i.e. 75 %
overlap between neighbouring windows, 87 windows per simulated day. The battery's
default protocol (``federation.split_mode = "random"``) then splits those windows
with a *seeded permutation* -- :func:`fedgrid.data._split_indices`, 70/15/15 --
so a held-out test window and a training window can be three quarters the same
samples. Two windows of the same client-day share samples iff their indices differ
by less than ``WINDOW_T / STRIDE = 4``; a permutation split makes no attempt to
avoid that, so almost every test window has a near-copy of itself in the train set
and the reported macro-F1 is partly a memorisation score, not a generalisation one.

Why the number matters
----------------------
It is the size of the objection that motivated the leakage-free re-run: the paper
reports both the random-split battery (blocks ``t1a``/``t1b``) and the
chronological re-run (``t1a_temporal``/``t1b_temporal``,
:func:`fedgrid.data.apply_temporal_split` -- oldest 70 % train, newest 15 % test,
with a four-window purge at each block boundary). This script turns "the random
split leaks" from an assertion into a measured, re-runnable figure, and prints the
same statistic under the temporal split as the control (it must be 0).

    python -m experiments.measure_leakage
    python -m experiments.measure_leakage --split random --json leakage.json

What it reuses, and what it computes here
-----------------------------------------
The federation, the window geometry and *both* splitters are the framework's own:
the data is loaded with :func:`fedgrid.data.load_federation` (persisted ``.npz``
windows and persisted permutation indices -- the exact index sets the battery runs
trained on) and the chronological control is produced by
:func:`fedgrid.data.apply_split_mode`. Nothing is re-derived or re-seeded.

The *interval arithmetic is implemented in this module* (there is no overlap
helper in the library to reuse): window index ``i`` of a client is mapped to
``(day = i // 87, start = (i % 87) * STRIDE)`` and two windows are counted as
overlapping when they belong to the same client-day and their ``[start,
start + WINDOW_T)`` sample intervals intersect. Days are treated as separate
blocks because they are separate simulations (client-day ``d`` is calendar day
``3 + 12 d``, :func:`fedgrid.simulation.partition.sim_calendar_day`) and each day
is windowed independently, so no window ever straddles a day boundary. That
mapping is not assumed: ``--verify-geometry`` (on by default) checks it against
the persisted window tensors themselves, asserting that consecutive windows share
their last/first ``WINDOW_T - STRIDE`` samples exactly within a day and do *not*
at every day boundary.

Deterministic (no RNG is drawn here), CPU-only, numpy-only arithmetic.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from fedgrid.data import Federation

__all__ = [
    "client_overlap",
    "measure_federation",
    "verify_geometry",
    "windows_per_day",
]

#: ``--split both`` runs the random split and the chronological control in one go;
#: the contrast is the point of the measurement, so it is the default.
SPLIT_CHOICES: tuple[str, ...] = ("random", "temporal", "both")


# --------------------------------------------------------------------------- #
# Window geometry.
# --------------------------------------------------------------------------- #


def windows_per_day(n_windows: int, days: int) -> int:
    """Windows cut from one simulated day (87 for the 1440-sample default day).

    Derived from the client's own window count rather than hard-coded, so a
    federation generated with a different day length still measures correctly.
    """
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days}")
    if n_windows % days:
        raise ValueError(
            f"{n_windows} windows do not divide into {days} equal days; the "
            "day-block mapping this measurement relies on would be wrong"
        )
    return n_windows // days


def _day_and_start(n: int, wpd: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Window index -> (client-day, start sample within that day)."""
    idx = np.arange(n, dtype=np.int64)
    return idx // wpd, (idx % wpd) * stride


def verify_geometry(windows: np.ndarray, wpd: int, stride: int) -> dict[str, Any]:
    """Check the (day, start) mapping against the persisted window tensors.

    Consecutive windows *within* a day must satisfy ``w[i][:, stride:] ==
    w[i + 1][:, :stride - WINDOW_T]`` exactly (they are copies of the same
    samples), and that identity must fail at every day boundary. Returns the
    counts plus ``ok``; no tolerance is used -- the comparison is bit-exact.
    """
    n = int(windows.shape[0])
    same = np.all(windows[:-1, :, stride:] == windows[1:, :, :-stride], axis=(1, 2))
    expected = ((np.arange(n - 1, dtype=np.int64) + 1) % wpd) != 0
    return {
        "pairs_checked": int(n - 1),
        "contiguous_pairs": int(same.sum()),
        "expected_contiguous_pairs": int(expected.sum()),
        "day_boundaries": int((~expected).sum()),
        "ok": bool(np.array_equal(same, expected)),
    }


# --------------------------------------------------------------------------- #
# The measurement.
# --------------------------------------------------------------------------- #


def client_overlap(
    n: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    wpd: int,
    window_t: int,
    stride: int,
) -> dict[str, Any]:
    """Overlap statistics for one client's test windows against its train windows.

    For every test window, every train window of the same client-day whose sample
    interval intersects it is counted. Returns

    ``leak_fraction``
        share of test windows overlapping at least one train window -- the headline.
    ``mean_overlapping_train_windows``
        mean count of such train windows per test window (over *all* test windows).
    ``mean_max_overlap_fraction``
        mean over test windows of the largest single-train-window overlap / ``L``.
    ``mean_union_coverage``
        mean share of a test window's ``L`` samples that appear in *some* train
        window -- the strictest reading of "how much of this window was seen".
    """
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    day, start = _day_and_start(n, wpd, stride)

    is_train = np.zeros(n, dtype=bool)
    is_train[train_idx] = True

    n_test = int(test_idx.size)
    if n_test == 0:
        raise ValueError("client has no test windows")
    n_over = np.zeros(n_test, dtype=np.int64)      # overlapping train windows
    max_ov = np.zeros(n_test, dtype=np.int64)      # largest overlap, in samples
    covered = np.zeros((n_test, window_t), dtype=bool)  # union coverage mask

    # Only |i - j| * stride < window_t can intersect, so the candidate set is the
    # +-max_delta index neighbourhood; every candidate is still tested with the
    # explicit interval intersection below rather than trusted by construction.
    max_delta = (window_t - 1) // stride
    t_day, t_start = day[test_idx], start[test_idx]
    for delta in range(-max_delta, max_delta + 1):
        j = test_idx + delta
        ok = (j >= 0) & (j < n)
        jc = np.clip(j, 0, n - 1)
        ok &= is_train[jc] & (day[jc] == t_day)
        if not ok.any():
            continue
        j_start = start[jc]
        # [a, a+L) vs [b, b+L): intersection length, clipped at zero.
        ov = np.minimum(t_start + window_t, j_start + window_t) - np.maximum(t_start, j_start)
        hit = ok & (ov > 0)
        if not hit.any():
            continue
        n_over += hit
        max_ov = np.maximum(max_ov, np.where(hit, ov, 0))
        # Within a day, start_j - start_i is exactly delta * stride, so the
        # covered slice of the test window is the same for every hit at this delta.
        shift = delta * stride
        if not np.all(j_start[hit] - t_start[hit] == shift):
            raise AssertionError("same-day windows are not on the expected stride lattice")
        covered[hit, max(0, shift):min(window_t, window_t + shift)] = True

    leaking = n_over > 0
    return {
        "n_windows": int(n),
        "n_train": int(train_idx.size),
        "n_test": n_test,
        "n_test_leaking": int(leaking.sum()),
        "leak_fraction": float(leaking.mean()),
        "mean_overlapping_train_windows": float(n_over.mean()),
        "max_overlapping_train_windows": int(n_over.max()),
        "mean_max_overlap_fraction": float(max_ov.mean() / window_t),
        "mean_union_coverage": float(covered.mean()),
    }


def measure_federation(
    fed: Federation,
    window_t: int,
    stride: int,
    verify: bool = True,
) -> dict[str, Any]:
    """Run :func:`client_overlap` over every client and pool the counts.

    ``fed`` must already carry the split under test (``load_federation(dir,
    split_mode)`` / :func:`fedgrid.data.apply_split_mode` have run); this function
    only reads ``client.train.indices`` and ``client.test.indices``.
    """
    per_client: list[dict[str, Any]] = []
    geometry: list[dict[str, Any]] = []
    days = int(fed.days) or int(fed.clients[0].meta.days)
    for client in fed.clients:
        n = len(client.labels)
        wpd = windows_per_day(n, days)
        if verify:
            g = verify_geometry(client.windows, wpd, stride)
            g["cid"] = int(client.meta.cid)
            geometry.append(g)
            if not g["ok"]:
                raise AssertionError(
                    f"client {client.meta.cid}: persisted windows do not match the "
                    f"stride-{stride} / {wpd}-per-day geometry this measurement assumes"
                )
        stats = client_overlap(
            n,
            np.asarray(client.train.indices, dtype=np.int64),
            np.asarray(client.test.indices, dtype=np.int64),
            wpd,
            window_t,
            stride,
        )
        stats["cid"] = int(client.meta.cid)
        stats["windows_per_day"] = int(wpd)
        per_client.append(stats)

    fracs = np.array([c["leak_fraction"] for c in per_client], dtype=np.float64)
    n_test = np.array([c["n_test"] for c in per_client], dtype=np.int64)
    n_leak = np.array([c["n_test_leaking"] for c in per_client], dtype=np.int64)
    # Pooled = over the union of the client test sets, i.e. the federation's global
    # test set (Federation.global_test), which is what every reported number scores.
    weights = n_test / n_test.sum()
    pooled = {
        "n_test": int(n_test.sum()),
        "n_test_leaking": int(n_leak.sum()),
        "leak_fraction": float(n_leak.sum() / n_test.sum()),
        "mean_overlapping_train_windows": float(
            np.average([c["mean_overlapping_train_windows"] for c in per_client], weights=weights)
        ),
        "mean_max_overlap_fraction": float(
            np.average([c["mean_max_overlap_fraction"] for c in per_client], weights=weights)
        ),
        "mean_union_coverage": float(
            np.average([c["mean_union_coverage"] for c in per_client], weights=weights)
        ),
        "per_client_leak_fraction_min": float(fracs.min()),
        "per_client_leak_fraction_median": float(np.median(fracs)),
        "per_client_leak_fraction_max": float(fracs.max()),
    }
    return {"pooled": pooled, "per_client": per_client, "geometry": geometry}


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #


def _resolve_federation(
    k: int, alpha: float, days: int, seed: int, data_dir: Path | None, split_mode: str
) -> tuple[Federation, str]:
    """Load the pre-generated federation for (k, alpha, days, seed), else build it.

    Same resolution rule as a real run (:func:`experiments.build.find_federation_dir`):
    ``data_dir`` may be one generated federation or a root of ``<fed_id>/`` subdirs.
    Generating in-process needs OpenDSS and takes minutes; the staged battery data
    is the fast path and is what the reported figures use.
    """
    from experiments.build import find_federation_dir

    config = {"federation": {"k": k, "alpha": alpha, "days": days, "seed": seed}}
    found = find_federation_dir(config, data_dir)
    if found is not None:
        from fedgrid.data import load_federation

        return load_federation(found, split_mode=split_mode), str(found)

    from fedgrid.data import apply_split_mode
    from fedgrid.simulation.partition import make_federation

    print(f"[leakage] no staged federation under {data_dir}; generating in-process "
          f"(k={k}, alpha={alpha}, days={days}, seed={seed}) -- this runs OpenDSS")
    fed = make_federation(k=k, alpha=alpha, days=days, seed=seed)
    return apply_split_mode(fed, split_mode), "<generated in-process>"


def _print_report(split_mode: str, report: dict[str, Any]) -> None:
    p = report["pooled"]
    print(f"\n=== split_mode = {split_mode} ===")
    print(f"  pooled test windows overlapping >=1 train window : "
          f"{p['n_test_leaking']}/{p['n_test']} = {p['leak_fraction']:.4%}")
    print(f"  per-client leak fraction  min / median / max      : "
          f"{p['per_client_leak_fraction_min']:.4%} / "
          f"{p['per_client_leak_fraction_median']:.4%} / "
          f"{p['per_client_leak_fraction_max']:.4%}")
    print(f"  mean overlapping train windows per test window    : "
          f"{p['mean_overlapping_train_windows']:.4f}")
    print(f"  mean max single-window overlap (samples / L)      : "
          f"{p['mean_max_overlap_fraction']:.4%}")
    print(f"  mean union coverage of a test window by train     : "
          f"{p['mean_union_coverage']:.4%}")
    print("  per client:")
    print("    cid   n_test  leaking   frac      mean_ovl  union_cov")
    for c in report["per_client"]:
        print(f"    {c['cid']:>3}  {c['n_test']:>7}  {c['n_test_leaking']:>7}  "
              f"{c['leak_fraction']:>7.4%}  {c['mean_overlapping_train_windows']:>8.4f}  "
              f"{c['mean_union_coverage']:>8.4%}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m experiments.measure_leakage",
        description="Measure how many test windows overlap a train window in samples.",
    )
    p.add_argument("--k", type=int, default=10, help="clients (default 10, the t1a frame)")
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha (default 0.5)")
    p.add_argument("--days", type=int, default=30, help="simulated days per client (default 30)")
    p.add_argument("--seed", type=int, default=0, help="federation seed (default 0)")
    p.add_argument("--split", choices=SPLIT_CHOICES, default="both",
                   help="which protocol to measure (default both: random + the control)")
    p.add_argument("--data-dir", type=Path, default=Path("data_battery"),
                   help="staged federation dir or root of <fed_id>/ subdirs "
                        "(default data_battery); generated in-process if no match")
    p.add_argument("--gap", type=int, default=None,
                   help="windows purged at each temporal block boundary "
                        "(default fedgrid.data.TEMPORAL_SPLIT_GAP = 4)")
    p.add_argument("--json", type=Path, default=None,
                   help="write the full report (pooled + per-client) to this path")
    p.add_argument("--no-verify-geometry", dest="verify", action="store_false",
                   help="skip the bit-exact check of the day/stride mapping against "
                        "the persisted windows")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Heavy imports after argparse so --help stays instant.
    from fedgrid.data import TEMPORAL_SPLIT_GAP, WINDOW_T
    from fedgrid.simulation.partition import STRIDE

    gap = TEMPORAL_SPLIT_GAP if args.gap is None else int(args.gap)
    modes = ("random", "temporal") if args.split == "both" else (args.split,)
    t0 = time.perf_counter()

    reports: dict[str, Any] = {}
    source = ""
    for mode in modes:
        fed, source = _resolve_federation(
            args.k, args.alpha, args.days, args.seed, args.data_dir, mode
        )
        if mode == "temporal" and gap != TEMPORAL_SPLIT_GAP:
            from fedgrid.data import apply_temporal_split

            apply_temporal_split(fed, gap=gap)
        reports[mode] = measure_federation(fed, WINDOW_T, STRIDE, verify=args.verify)
        _print_report(mode, reports[mode])

    elapsed = time.perf_counter() - t0
    wpd = reports[modes[0]]["per_client"][0]["windows_per_day"]
    payload = {
        "federation": {"k": args.k, "alpha": args.alpha, "days": args.days, "seed": args.seed},
        "source": source,
        "geometry": {
            "window_t": int(WINDOW_T),
            "stride": int(STRIDE),
            "windows_per_day": int(wpd),
            "samples_covered_per_day": int((wpd - 1) * STRIDE + WINDOW_T),
            "overlap_index_radius": int((WINDOW_T - 1) // STRIDE),
            "neighbour_overlap_fraction": float((WINDOW_T - STRIDE) / WINDOW_T),
            "window_t_source": "fedgrid.data.WINDOW_T",
            "stride_source": "fedgrid.simulation.partition.STRIDE",
        },
        "temporal_split_gap": int(gap),
        "splits": reports,
        "runtime_seconds": round(elapsed, 2),
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    if "random" in reports and "temporal" in reports:
        r = reports["random"]["pooled"]["leak_fraction"]
        t = reports["temporal"]["pooled"]["leak_fraction"]
        print(f"\nrandom split leaks {r:.4%} of test windows; "
              f"temporal control leaks {t:.4%}")
    print(f"done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
