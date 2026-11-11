"""Dataset generation CLI: materialise a federation to ``.npz`` files. Usage::

    python -m fedgrid.simulation.generate --k 10 --days 30 --seed 0 --out data/

Defaults are smoke-sized (``--days 1``); the large-scale run happens on the
server with ``--days 30``. Output: one ``client_XX.npz`` per client (raw
windows, labels, split indices, normaliser statistics, Dirichlet weights) plus
``manifest.json`` (config, per-client SHA-256 of the window bytes, class
histogram, throughput, engine/library versions). ``fedgrid.data.load_federation``
rebuilds the exact in-process :class:`~fedgrid.data.Federation` from this
output (identical splits and normalisation - no recomputation).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m fedgrid.simulation.generate",
        description="Generate the FedGrid federation dataset (OpenDSS co-simulation).",
    )
    p.add_argument("--k", type=int, default=10, help="number of clients (default 10)")
    p.add_argument("--days", type=int, default=1,
                   help="simulated days per client (default 1 = smoke; paper uses 30)")
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha (default 0.5)")
    p.add_argument("--seed", type=int, default=0, help="federation seed (default 0)")
    p.add_argument("--out", type=Path, default=Path("data"),
                   help="output directory (default data/)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Heavy imports after argparse so `--help` stays instant.
    import numpy as np

    import fedgrid
    from fedgrid.simulation.partition import make_federation

    # Absolute path up front: the OpenDSS engine may change the process cwd
    # during feeder compiles (restored by FeederSim, but belt and braces here).
    out: Path = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    fed = make_federation(k=args.k, alpha=args.alpha, days=args.days, seed=args.seed)
    elapsed = time.perf_counter() - t0

    clients_meta = []
    total_windows = 0
    for c in fed.clients:
        fname = f"client_{c.meta.cid:02d}.npz"
        np.savez_compressed(
            out / fname,
            windows=c.windows,
            labels=c.labels,
            class_weights=c.meta.class_weights,
            train_idx=np.asarray(c.train.indices, dtype=np.int64),
            val_idx=np.asarray(c.val.indices, dtype=np.int64),
            test_idx=np.asarray(c.test.indices, dtype=np.int64),
            norm_mean=c.normalizer.mean,
            norm_std=c.normalizer.std,
        )
        total_windows += len(c)
        clients_meta.append({
            "cid": c.meta.cid,
            "file": fname,
            "feeder": {
                "name": c.meta.feeder.name,
                "load_shape_seed": c.meta.feeder.load_shape_seed,
                "pv_penetration": c.meta.feeder.pv_penetration,
            },
            "split_seed": c.meta.split_seed,
            "n_windows": len(c),
            "n_nonconverged": c.meta.n_nonconverged,
            "n_steps": c.meta.n_steps,
            "sha256_windows": hashlib.sha256(c.windows.tobytes()).hexdigest(),
        })

    hist = fed.label_histogram()
    normal_share = float(hist[:, 0].sum() / hist.sum())
    wpm = total_windows / (elapsed / 60.0) if elapsed > 0 else float("inf")
    nonconv = sum(cm["n_nonconverged"] for cm in clients_meta)
    steps = sum(cm["n_steps"] for cm in clients_meta)
    manifest = {
        "config": {"k": args.k, "days": args.days, "alpha": args.alpha, "seed": args.seed},
        "fedgrid_version": fedgrid.__version__,
        "engine": _engine_version(),
        "clients": clients_meta,
        "label_histogram": hist.tolist(),
        "normal_share": normal_share,
        "total_windows": total_windows,
        # Solver non-convergence held-sample accounting (policy in docs/REALISM.md).
        "nonconvergence": {
            "held_steps": nonconv,
            "total_steps": steps,
            "rate": (nonconv / steps) if steps else 0.0,
        },
        "elapsed_s": round(elapsed, 2),
        "windows_per_min": round(wpm, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rate = (nonconv / steps) if steps else 0.0
    print(f"wrote {args.k} clients x {args.days} day(s) -> {out} | "
          f"{total_windows} windows in {elapsed:.1f}s ({wpm:.0f} windows/min) | "
          f"NORMAL share {normal_share:.1%} | "
          f"solver-held {nonconv}/{steps} steps ({rate:.3%})")
    return 0


def _engine_version() -> str:
    import opendssdirect as dss

    return str(dss.Basic.Version()).split(";")[0].strip()


if __name__ == "__main__":
    raise SystemExit(main())
