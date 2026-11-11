"""Pre-generate every distinct battery federation into ``<root>/<fed_id>/`` subdirs.

The phase 4 fleet loads pre-generated data (workers never run OpenDSS in-process). The
battery's 500 runs draw on only a handful of distinct federations -- each defined solely by
``(k, alpha, days, seed)`` -- so we generate each once here and stage them under one root that
every worker carries. ``build.find_federation_dir`` then resolves each run to its subdir by
matching the generator manifest, so no federation is regenerated per run.

Usage (run where OpenDSS is available, e.g. the server WSL):
    python -m experiments.stage_data --manifest experiments/manifest.jsonl --out data_battery
Idempotent: a federation whose ``manifest.json`` already exists is skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Federation-block keys that are a *view* of the generated data, not part of it.
# ``split_mode`` re-cuts the persisted windows at load time (fedgrid.data.
# apply_split_mode), so the temporal battery reuses the very same staged
# directories: stripping it here keeps every existing fed_<hash> id byte-stable and
# stops `stage_data --manifest manifest_temporal.jsonl` from re-running OpenDSS for
# data that is already on disk.
_VIEW_ONLY_KEYS: tuple[str, ...] = ("split_mode",)


def _data_block(fed: dict[str, Any]) -> dict[str, Any]:
    """The part of a ``federation`` block that actually determines the generated data."""
    return {k: v for k, v in fed.items() if k not in _VIEW_ONLY_KEYS}


def distinct_federations(manifest_path: Path) -> list[dict[str, Any]]:
    """The unique ``federation`` blocks across the manifest, in first-seen order."""
    seen: dict[str, dict[str, Any]] = {}
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fed = _data_block(json.loads(line)["config"]["federation"])
            seen.setdefault(json.dumps(fed, sort_keys=True), fed)
    return list(seen.values())


def federation_id(fed: dict[str, Any]) -> str:
    """Stable, filesystem-safe id for a federation (content hash, not an index)."""
    canonical = json.dumps(_data_block(fed), sort_keys=True).encode()
    digest = hashlib.sha256(canonical).hexdigest()[:10]
    return f"fed_{digest}"


def stage(manifest_path: Path, root: Path, dry_run: bool = False) -> int:
    feds = distinct_federations(manifest_path)
    root.mkdir(parents=True, exist_ok=True)
    print(f"{len(feds)} distinct federations to stage into {root}")
    total_windows = 0
    for i, fed in enumerate(feds, 1):
        dest = root / federation_id(fed)
        tag = (f"k={fed['k']} alpha={fed['alpha']} days={fed['days']} "
               f"seed={fed['seed']} iid={fed.get('iid', False)}")
        if (dest / "manifest.json").exists():
            print(f"[{i}/{len(feds)}] SKIP {dest.name}  ({tag})")
            continue
        print(f"[{i}/{len(feds)}] GEN  {dest.name}  ({tag})")
        if dry_run:
            continue
        cmd = [
            sys.executable, "-m", "fedgrid.simulation.generate",
            "--k", str(fed["k"]), "--days", str(fed["days"]),
            "--seed", str(fed["seed"]), "--alpha", str(fed["alpha"]),
            "--out", str(dest),
        ]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"FAILED to generate {dest} (rc={rc})", file=sys.stderr)
            return rc
        try:
            m = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
            total_windows += int(m.get("n_windows", m.get("windows", 0)) or 0)
        except (OSError, ValueError):
            pass
    print(f"staging complete: {len(feds)} federations under {root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("experiments/manifest.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data_battery"))
    p.add_argument("--dry-run", action="store_true", help="list federations without generating")
    args = p.parse_args(argv)
    return stage(args.manifest, args.out, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
