"""RunResult: the single serialisation format every experiment writes.

``git_commit`` and ``env_fingerprint`` are mandatory per contract amendment
2026-07-28d §4 — every quoted paper number must be traceable to the exact code
and machine that produced it. :func:`load_run` still reads pre-amendment files
(missing fields load as ``"unknown"``), which is the amendment's declared
backward compatibility.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "RunResult",
    "current_env_fingerprint",
    "current_git_commit",
    "load_run",
    "save_run",
]


@dataclass
class RunResult:
    """Everything one experiment run produced, in one serialisable record."""

    method: str
    seed: int
    config: dict
    rounds: list[dict]  # one dict per FL round (see api RoundEvent fields)
    final: dict  # acc, macro-F1, per-class F1, confusion matrix
    energy: dict | None
    latency: dict | None
    git_commit: str  # amendment 2026-07-28d §4
    env_fingerprint: str  # torch/CUDA versions + GPU name (see helper below)


def save_run(r: RunResult, path: Path) -> None:
    """Persist a run to ``path``; the on-disk form must round-trip via :func:`load_run`.

    Plain JSON, sorted keys — byte-stable for the same record, which the
    same-seed determinism test hashes (VALIDATION_STANDARD.md §3.7).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dataclasses.asdict(r), indent=2, sort_keys=True)
    path.write_text(payload, encoding="utf-8")


def load_run(path: Path) -> RunResult:
    """Load a run previously written by :func:`save_run`.

    Tolerates pre-amendment files that lack ``git_commit``/``env_fingerprint``
    (loaded as ``"unknown"``); unrecognised extra keys are ignored so newer
    files stay readable by older code.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    kwargs = {f.name: data[f.name] for f in dataclasses.fields(RunResult) if f.name in data}
    kwargs.setdefault("git_commit", "unknown")
    kwargs.setdefault("env_fingerprint", "unknown")
    return RunResult(**kwargs)


def current_git_commit(repo_root: str | Path = ".") -> str:
    """HEAD commit hash of ``repo_root``, or ``"unknown"``.

    Tolerant by design: missing git binary, not a repository, or a repository
    with zero commits (this repo's state at phase 2) all yield ``"unknown"``
    rather than raising — provenance capture must never kill an experiment.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = out.stdout.strip()
    return commit if out.returncode == 0 and commit else "unknown"


def current_env_fingerprint() -> str:
    """One-line environment fingerprint: python | torch | CUDA | GPU name.

    Example: ``py3.14.3 | torch 2.11.0+cu128 | cuda 12.8 | NVIDIA GeForce RTX
    3060``; the GPU field reads ``cpu`` when CUDA is unavailable. Torch is
    imported lazily so provenance capture works even in torch-less tooling
    contexts (reported as ``torch unavailable``).
    """
    import platform

    py = f"py{platform.python_version()}"
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        return f"{py} | torch unavailable"
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        cuda = torch.version.cuda or "unknown"
    else:
        gpu = "cpu"
        cuda = "none"
    return f"{py} | torch {torch.__version__} | cuda {cuda} | {gpu}"
