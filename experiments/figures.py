"""RunResults -> the manuscript's figures, as vector PDFs.

One command renders every figure the manuscript needs into ``paper/figures/``::

    python -m experiments.figures --runs-dir runs_snn_fixed runs runs_x3 \
        runs_ssm_x3 runs_gap_x3

Design rules, applied uniformly so they read as one system:

* **Form follows the data's job.** Magnitude over a grid is a heatmap on a single
  sequential hue; identity across series is the fixed categorical order; a
  distribution is a box with its own points, never a bar of means.
* **Colour is assigned, never cycled.** The categorical slots are used in a fixed
  order and validated for protanopia/deuteranopia separation; scatter-like forms
  (where any two marks can touch) carry at most three of them.
* **No dual axes, no chartjunk.** Hairline recessive grid, two spines, thin
  marks, a legend whenever more than one series is drawn, selective direct
  labels rather than a number on every point.
* **Sized for one journal column** (222 pt) at 8 pt type. Every PDF is saved at
  exactly that width, so ``width=\\linewidth`` neither enlarges nor shrinks it and
  all carry the same effective type size in the compiled manuscript.

**Missing data is never invented.** A series with no runs is skipped and named in
a visible note on the figure; a figure with no data at all still renders, as a
loud red "no data" panel, so a stale asset can never be mistaken for a result.

Owner: results pipeline (phase 5). Companion: :mod:`experiments.report`.
"""

from __future__ import annotations

import argparse
import math
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments import analyze as A
from experiments import report as R

__all__ = ["FIGURES", "render_all", "style"]

# --------------------------------------------------------------------------- #
# Design tokens. Categorical order validated (adjacent CVD dE 9.1, normal-vision
# 19.6, OKLab x100) -- do not re-order; the order is the safety mechanism.
# --------------------------------------------------------------------------- #

SERIES: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                           "#e87ba4", "#008300", "#4a3aa7", "#e34948")
#: Scatter / small-multiple forms, where any two marks can sit side by side,
#: are capped at the three slots that clear the all-pairs floors.
SERIES_ALLPAIRS: tuple[str, ...] = SERIES[:3]
#: One-hue sequential ramp (light -> dark) for magnitude.
SEQUENTIAL: tuple[str, ...] = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                               "#256abf", "#184f95", "#0d366b")
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
CRITICAL = "#d03b3b"

#: One elsarticle 3p two-column text column is 222.0 pt = 3.072 in; every figure
#: is designed and saved at exactly that width so `width=\linewidth` is a no-op.
COL_W = 222.0 / 72.27
COL_W_PT = 222.0
GOLDEN = 0.62


def style() -> None:
    """Apply the shared rcParams; identical typography across every figure."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.labelcolor": INK,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.0,
        "legend.handletextpad": 0.5,
        "pdf.fonttype": 42,  # embed TrueType, not Type 3 (publisher requirement)
        "ps.fonttype": 42,
        # Saving goes through _save(), which pins the natural width to one column
        # so LaTeX scales every figure by the same factor.
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
    })


#: rcParams scaled when a figure cannot fit a column at its design size.
_SCALABLE = ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize",
             "ytick.labelsize", "legend.fontsize", "lines.linewidth",
             "lines.markersize", "axes.linewidth", "grid.linewidth")


def _scale_rc(factor: float) -> dict[str, Any]:
    """Multiply the type/mark scale by ``factor``; returns the previous values."""
    import matplotlib as mpl

    previous = {k: mpl.rcParams[k] for k in _SCALABLE}
    for key, value in previous.items():
        mpl.rcParams[key] = float(value) * factor
    return previous


def _save(fig: Any, path: Path) -> None:
    """Save with the natural width pinned to one column.

    ``\\includegraphics[width=\\linewidth]`` rescales whatever it is handed, so a
    figure whose tight bounding box is narrower than a column would be *enlarged*
    and one that is wider would be shrunk -- and they would end up with
    different effective type sizes. Padding the box out to exactly one column
    makes that scale factor 1.0. A figure too wide to fit is a design problem, not
    a save problem: :func:`column_scale` reports it and the CLI prints it.
    """
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    pad = 0.02
    x0, x1 = bb.x0 - pad, bb.x1 + pad
    if (x1 - x0) < COL_W:
        centre = 0.5 * (x0 + x1)
        x0, x1 = centre - COL_W / 2.0, centre + COL_W / 2.0
    fig.savefig(path, bbox_inches=Bbox.from_extents(x0, bb.y0 - pad, x1, bb.y1 + pad))


#: Characters of 6.5 pt DejaVu Sans that fit across one 222 pt column.
_NOTE_WRAP = 52


def _note(fig: Any, text: str, y: float = -0.06) -> None:
    """A visible, non-decorative note naming what is missing from the figure.

    Wrapped to the column: an unwrapped note is the one thing that can push a
    figure's natural width past a column and shrink the whole plot with it.
    """
    if not text:
        return
    fig.text(0.5, y, textwrap.fill(text, _NOTE_WRAP), ha="center", va="top",
             fontsize=6.5, color=CRITICAL, style="italic")


def _no_data(path: Path, message: str, width: float = COL_W) -> bool:
    """Render a loud placeholder so an absent measurement can never look like one."""
    import matplotlib.pyplot as plt

    body = textwrap.fill(message, 52)
    n_lines = body.count("\n") + 1
    fig, ax = plt.subplots(figsize=(width, 0.62 + 0.13 * n_lines))
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0.005, 0.02), 0.99, 0.96, transform=ax.transAxes,
                               fill=False, edgecolor=CRITICAL, linewidth=1.2))
    ax.text(0.5, 0.84, "NO DATA", transform=ax.transAxes, ha="center", va="top",
            fontsize=11, fontweight="bold", color=CRITICAL)
    ax.text(0.5, 0.52, body, transform=ax.transAxes, ha="center", va="top",
            fontsize=6.5, color=CRITICAL, style="italic")
    _save(fig, path)
    plt.close(fig)
    return False


def _finish(fig: Any, path: Path, missing: Sequence[str] = (),
            prefix: str = "no runs yet for", y: float = -0.06) -> bool:
    import matplotlib.pyplot as plt

    if missing:
        _note(fig, f"{prefix}: " + "; ".join(sorted(set(missing))), y=y)
    _save(fig, path)
    plt.close(fig)
    return True


def _shared_legend(fig: Any, ax: Any, ncol: int = 2, y: float = -0.02) -> None:
    """One legend for the whole figure, below the panels -- never over the data.

    ``handlelength`` is deliberately long: several figures separate series by
    dash pattern as well as hue, and a short handle renders solid and dashed
    identically, which would put the identity back on colour alone.
    """
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 2:
        fig.legend(handles, labels, ncol=ncol, loc="upper center",
                   bbox_to_anchor=(0.5, y), fontsize=6.5, handlelength=2.2,
                   columnspacing=0.7, handletextpad=0.4)


def _boxplot(ax: Any, data: Any, **kw: Any) -> Any:
    """Horizontal boxplot across the Matplotlib 3.11 ``vert``/``orientation`` rename."""
    import matplotlib as mpl

    if tuple(int(p) for p in mpl.__version__.split(".")[:2]) >= (3, 11):
        return ax.boxplot(data, orientation="horizontal", **kw)
    return ax.boxplot(data, vert=False, **kw)


def _band(ax: Any, x: Sequence[float], lo: Sequence[float], hi: Sequence[float],
          color: str) -> None:
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)


def _legend(ax: Any, ncol: int = 2, loc: str = "best", **kw: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 2:  # a single series is named by the caption, not a box
        ax.legend(handles, labels, ncol=ncol, loc=loc, **kw)


# --------------------------------------------------------------------------- #
# 1. Per-client label histogram (magnitude over client x class -> heatmap).
# --------------------------------------------------------------------------- #


def fig_label_histogram(ctx: Ctx, path: Path) -> bool:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm

    wanted = [(0.1, r"$\alpha=0.1$"), (0.5, r"$\alpha=0.5$"),
              (1.0, r"$\alpha=1.0$"), (100.0, "near-uniform")]
    panels = []
    missing = []
    for alpha, label in wanted:
        fed = next((f for f in ctx.feds
                    if R._get(f, "config", "alpha") == alpha
                    and R._get(f, "config", "seed") == 0), None)
        hist = (fed or {}).get("label_histogram")
        if not hist:
            missing.append(label)
            continue
        panels.append((label, np.asarray(hist, dtype=float)))
    if not panels:
        return _no_data(path, "no staged federation manifests with a label histogram")

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)
    ncol = 2 if len(panels) > 1 else 1
    nrow = math.ceil(len(panels) / ncol)
    # Slightly under a column: the class tick labels and the colourbar live
    # outside the axes grid, and _save() pins the saved width to COL_W exactly.
    fig, axes = plt.subplots(nrow, ncol, figsize=(COL_W * 0.93, 1.35 * nrow + 0.55),
                             squeeze=False)
    norm = PowerNorm(gamma=0.35, vmin=0.0, vmax=1.0)
    for ax, (label, hist) in zip(axes.ravel(), panels, strict=False):
        share = hist / np.maximum(hist.sum(axis=1, keepdims=True), 1.0)
        ax.imshow(share.T, cmap=cmap, norm=norm, aspect="auto",
                  interpolation="nearest")
        ax.set_title(label, fontsize=7.5, color=INK, pad=3)
        ax.set_xticks(range(share.shape[0]))
        ax.set_xticklabels(range(share.shape[0]), fontsize=5.5)
        ax.set_yticks(range(share.shape[1]))
        ax.set_yticklabels(CLASS_SHORT[:share.shape[1]], fontsize=5.5)
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("client", fontsize=7)
    fig.subplots_adjust(hspace=0.55, wspace=0.45)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      ax=axes.ravel().tolist(), fraction=0.028, pad=0.02,
                      ticks=[0.0, 0.05, 0.25, 0.6, 1.0])
    cb.set_label("share within client", fontsize=6)
    cb.ax.tick_params(labelsize=5.5, length=0)
    cb.outline.set_visible(False)
    return _finish(fig, path, missing)


CLASS_SHORT = ("NORMAL", "LG", "LL", "LLG", "3PH", "OVLD", "SAG", "SWELL")


# --------------------------------------------------------------------------- #
# 2. Example windows (small multiples; three phase voltages + one cross-feeder
#    reference in muted ink, so the fourth line is not a fourth hue).
# --------------------------------------------------------------------------- #


def fig_example_windows(ctx: Ctx, path: Path) -> bool:
    import matplotlib.pyplot as plt

    fed_dir = ctx.reference_federation_dir()
    if fed_dir is None:
        return _no_data(path, "no staged reference federation on disk")
    ref_cid, ref_name = ctx.client_for_feeder(fed_dir, "ieee13")
    alt_cid, alt_name = ctx.client_for_feeder(fed_dir, "ieee37")
    if ref_cid is None:
        return _no_data(path, "reference federation has no IEEE 13 client")

    windows, labels = _load_windows(fed_dir, ref_cid)
    if windows is None:
        return _no_data(path, f"cannot read client_{ref_cid:02d}.npz")
    alt_w, alt_l = (_load_windows(fed_dir, alt_cid) if alt_cid is not None
                    else (None, None))

    fig, axes = plt.subplots(4, 2, figsize=(COL_W, 4.0), sharex=True)
    missing: list[str] = []
    for cls in range(8):
        ax = axes.ravel()[cls]
        idx = np.flatnonzero(labels == cls)
        if idx.size == 0:
            missing.append(CLASS_SHORT[cls])
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{CLASS_SHORT[cls]}: absent", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6, color=CRITICAL)
            continue
        w = windows[idx[0]]
        t = np.arange(w.shape[1])
        for ch, (name, colour) in enumerate(zip(("V$_a$", "V$_b$", "V$_c$"),
                                                SERIES_ALLPAIRS, strict=True)):
            ax.plot(t, w[ch], color=colour, linewidth=0.9,
                    label=name if cls == 0 else None)
        if alt_w is not None and alt_l is not None:
            alt_idx = np.flatnonzero(alt_l == cls)
            if alt_idx.size:
                ax.plot(t, alt_w[alt_idx[0]][0], color=MUTED, linewidth=0.7,
                        linestyle="--",
                        label=f"V$_a$, {alt_name}" if cls == 0 else None)
        ax.set_title(CLASS_SHORT[cls], fontsize=6.5, color=INK, pad=2)
        ax.tick_params(labelsize=5.5)
        ax.set_yticks([])
    for ax in axes[-1]:
        ax.set_xlabel("sample within window", fontsize=6.5)
    axes[0, 0].set_ylabel(f"pu ({ref_name})", fontsize=6.5)
    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.045),
               fontsize=6.5)
    fig.subplots_adjust(hspace=0.55, wspace=0.15)
    return _finish(fig, path, missing, prefix="classes absent from this client",
                   y=-0.005)


def _load_windows(fed_dir: Path, cid: int | None) -> tuple[Any, Any]:
    if cid is None:
        return None, None
    npz = fed_dir / f"client_{cid:02d}.npz"
    if not npz.exists():
        return None, None
    try:
        with np.load(npz) as z:
            return z["windows"], z["labels"]
    except (OSError, ValueError, KeyError):
        return None, None


# --------------------------------------------------------------------------- #
# 3. Round trajectories (change over time; one panel per model).
# --------------------------------------------------------------------------- #

#: Five arms on three all-pairs-safe hues: the *family* is the hue (ceiling /
#: plain averaging / robust pipeline) and the variant within a family is the
#: dash. Five distinct hues would fail the all-pairs colour-vision floor, and a
#: line chart is a form where any two series can sit side by side.
_FRAME_ARMS: tuple[tuple[str, str, int, str], ...] = (
    ("centralized", "Centralized", 0, "-"),
    ("fedavg", "FedAvg", 1, "-"),
    ("fedprox", "FedProx", 1, "--"),
    ("pipeline_krum", "Pipeline+Krum", 2, "-"),
    ("pipeline_geomed", "Pipeline+geomed", 2, "--"),
)


def fig_rounds(ctx: Ctx, path: Path) -> bool:
    import matplotlib.pyplot as plt

    models = ("cnn_lstm", "snn", "ssm")
    fig, axes = plt.subplots(len(models), 1, figsize=(COL_W, COL_W * 1.72),
                             sharex=True)
    missing: list[str] = []
    drawn = 0
    for ax, model in zip(axes, models, strict=True):
        for arm, label, slot, dash in _FRAME_ARMS:
            colour = SERIES_ALLPAIRS[slot]
            curves = [r.val_curve() for r in ctx.bank.select(block="t1a", arm=arm,
                                                             model=model)]
            curves = [c for c in curves if c]
            if not curves:
                missing.append(f"{label} ({R.MODEL_LABEL[model]})")
                continue
            n = min(len(c) for c in curves)
            mat = np.asarray([c[:n] for c in curves], dtype=float)
            x = np.arange(1, n + 1)
            mean = mat.mean(axis=0)
            ax.plot(x, mean, color=colour, linestyle=dash, label=label)
            if mat.shape[0] >= 2:
                lo, hi = zip(*(R.bca(list(mat[:, i]), ctx.resamples) or
                               (mean[i], mean[i]) for i in range(n)), strict=True)
                _band(ax, x, lo, hi, colour)
            drawn += 1
        ax.set_ylabel(f"validation macro-F$_1$\n({R.MODEL_LABEL[model]})", fontsize=7)
        ax.set_ylim(bottom=0.0)
    if not drawn:
        plt.close(fig)
        return _no_data(path, "no T1a runs with a round trajectory")
    axes[-1].set_xlabel("federated round")
    fig.subplots_adjust(hspace=0.12)
    _shared_legend(fig, axes[0], ncol=2)
    return _finish(fig, path, missing, y=-0.14)


# --------------------------------------------------------------------------- #
# 4. Accuracy-energy plane (scatter -> at most three categorical slots).
# --------------------------------------------------------------------------- #


def fig_energy_accuracy(ctx: Ctx, path: Path) -> bool:
    import matplotlib.pyplot as plt

    from fedgrid.metrics.energy import energy_uj

    bank = ctx.bank
    series: list[tuple[str, list[float], list[float], str, str]] = []
    missing: list[str] = []
    specs = (
        ("Spiking (delta)", dict(block="t1a", arm="fedavg", model="snn"), "delta"),
        ("Spiking (rate)", dict(block="t2e", arm="fedavg", model="snn"), "rate"),
    )
    for (label, kw, coding), colour in zip(specs, SERIES_ALLPAIRS, strict=False):
        rows = [r for r in bank.select(attack=None, **kw)
                if r.coding == coding and r.energy]
        uj = [float(energy_uj(r.energy["synops_per_inference"],
                              r.energy["ann_cnn_lstm_macs"])["snn_uJ"])
              for r in rows if R._finite(r.energy.get("synops_per_inference"))]
        f1 = [float(r.macro_f1) for r in rows if R._finite(r.macro_f1)]
        if uj and f1 and len(uj) == len(f1):
            series.append((label, uj, f1, colour, "o"))
        else:
            missing.append(label)
    cnn_rows = [r for r in bank.select(block="t1a", arm="fedavg", model="cnn_lstm")]
    macs = next((float(r.energy["ann_cnn_lstm_macs"]) for r in bank.rows
                 if r.energy and R._finite(r.energy.get("ann_cnn_lstm_macs"))), None)
    cnn_f1 = [float(r.macro_f1) for r in cnn_rows if R._finite(r.macro_f1)]
    if macs is not None and cnn_f1:
        ann_uj = float(energy_uj(1.0, macs)["ann_uJ"])
        series.append(("CNN-LSTM", [ann_uj] * len(cnn_f1), cnn_f1,
                       SERIES_ALLPAIRS[2], "s"))
    else:
        missing.append("CNN-LSTM")
    # Mamba: dense model costed by the same arithmetic-only accounting (its
    # analytic MACs stand in for SynOps). Drawn in the neutral ink with a
    # distinct marker so the three validated hues keep their all-pairs floors.
    ssm_rows = [r for r in bank.select(block="t1a", arm="fedavg", model="ssm")]
    ssm_macs = next((float(r.energy["macs_per_inference"]) for r in ssm_rows
                     if r.energy and R._finite(r.energy.get("macs_per_inference"))), None)
    ssm_f1 = [float(r.macro_f1) for r in ssm_rows if R._finite(r.macro_f1)]
    if ssm_macs is not None and ssm_f1:
        ssm_uj = float(energy_uj(1.0, ssm_macs)["ann_uJ"])
        series.append(("Mamba", [ssm_uj] * len(ssm_f1), ssm_f1, INK_2, "^"))
    else:
        missing.append("Mamba")
    if not series:
        return _no_data(path, "no runs carry an energy block")

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * GOLDEN))
    for label, uj, f1, colour, marker in series:
        ax.scatter(uj, f1, s=14, color=colour, marker=marker, label=label,
                   edgecolors=SURFACE, linewidths=0.8, zorder=3)
        mx, my = float(np.mean(uj)), float(np.mean(f1))
        ci = R.bca(f1, ctx.resamples)
        if ci:
            ax.errorbar([mx], [my], yerr=[[my - ci[0]], [ci[1] - my]], fmt="none",
                        ecolor=colour, elinewidth=0.9, capsize=1.8, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"estimated energy per inference ($\mu$J, log scale)")
    ax.set_ylabel("macro-F$_1$")
    ax.set_ylim(bottom=0.0)
    _legend(ax, ncol=1, loc="center left")
    return _finish(fig, path, missing, y=-0.14)


# --------------------------------------------------------------------------- #
# 5. Latency against the GOOSE envelope.
# --------------------------------------------------------------------------- #


def fig_latency_goose(ctx: Ctx, path: Path) -> bool:
    from fedgrid.metrics.latency import GOOSE_BUDGET_MS

    have = [r for r in ctx.bank.rows if isinstance(r.latency, dict) and r.latency]
    if not have:
        return _no_data(
            path,
            "no RunResult carries a latency block: the p50/p95/p99 measurement "
            f"(n=1000, batch 1) against the {GOOSE_BUDGET_MS:g} ms GOOSE envelope "
            "is a separate measurement pass and has not run")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * GOLDEN))
    labels: list[str] = []
    data: list[list[float]] = []
    for row in have:
        for device, blk in sorted(row.latency.items()):
            if isinstance(blk, dict) and R._finite(blk.get("p99_ms")):
                labels.append(f"{R.MODEL_LABEL.get(row.model, row.model)}\n"
                              f"{row.precision}/{device}")
                data.append([float(blk[k]) for k in ("p50_ms", "p95_ms", "p99_ms")
                             if R._finite(blk.get(k))])
    bp = _boxplot(ax, data, widths=0.55, patch_artist=True,
                  medianprops={"color": INK, "linewidth": 1.0},
                  whiskerprops={"color": AXIS, "linewidth": 0.7},
                  capprops={"color": AXIS, "linewidth": 0.7},
                  flierprops={"marker": "o", "markersize": 2, "markerfacecolor":
                              MUTED, "markeredgecolor": "none"})
    for patch, colour in zip(bp["boxes"], SERIES * 4, strict=False):
        patch.set(facecolor=colour, alpha=0.35, edgecolor=colour, linewidth=0.8)
    ax.set_yticklabels(labels, fontsize=6)
    ax.axvline(GOOSE_BUDGET_MS, color=CRITICAL, linewidth=1.0, zorder=1)
    ax.annotate(f"{GOOSE_BUDGET_MS:g} ms GOOSE budget", (GOOSE_BUDGET_MS, 0.5),
                xytext=(3, 0), textcoords="offset points", fontsize=6.5,
                color=CRITICAL, rotation=90, va="bottom")
    ax.set_xlabel("inference latency (ms)")
    return _finish(fig, path)


# --------------------------------------------------------------------------- #
# 6. Nemenyi critical-difference diagram (Demsar 2006).
# --------------------------------------------------------------------------- #


def fig_cd_diagram(ctx: Ctx, path: Path) -> bool:
    import matplotlib.pyplot as plt

    fr = ctx.stats["h2_friedman"]
    if not fr.get("available"):
        return _no_data(path, "the five defense arms do not yet share enough "
                              "(attack x model x seed) blocks for a Friedman test")
    # Compact labels: a one-column critical-difference diagram is text-width
    # bound, and scaling the type would scale the box with it.
    short = {"Pipeline + geo-median": "Pipe.+geomed", "Pipeline + Krum": "Pipe.+Krum",
             "Geo-median alone": "Geomed alone"}
    arms = [short.get(label, label) for label, _arm in R._DEFENSE_ROWS]
    ranks = [float(v) for v in fr["avg_ranks"]]
    cd = float(fr["cd"])
    k = len(arms)
    order = list(np.argsort(ranks))  # best (lowest) rank first
    half = (k + 1) // 2
    side = {i: ("left" if pos < half else "right") for pos, i in enumerate(order)}

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.72))
    ax.set_axis_off()
    lo, hi = 1.0, float(k)
    y_axis = 0.0
    ax.plot([lo, hi], [y_axis, y_axis], color=INK, linewidth=0.9, zorder=3)
    for tick in range(1, k + 1):
        ax.plot([tick, tick], [y_axis, y_axis + 0.035], color=INK, linewidth=0.8)
        ax.text(tick, y_axis + 0.055, str(tick), ha="center", va="bottom",
                fontsize=6.5, color=INK_2)
    ax.text((lo + hi) / 2, y_axis + 0.185, "average rank (1 = best)", ha="center",
            fontsize=7, color=INK_2)

    # Cliques (arms within one critical difference) sit just under the axis;
    # the arm connectors start below them so the two layers never collide.
    cliques: list[tuple[float, float]] = []
    for i in order:
        group = [b for b in order if abs(ranks[i] - ranks[b]) < cd]
        if len(group) < 2:
            continue
        span = (min(ranks[b] for b in group), max(ranks[b] for b in group))
        if not any(a <= span[0] and span[1] <= b for a, b in cliques):
            cliques = [c for c in cliques if not (span[0] <= c[0] and c[1] <= span[1])]
            cliques.append(span)
    y_clique = y_axis - 0.055
    for a, b in cliques:
        ax.plot([a, b], [y_clique, y_clique], color=INK, linewidth=2.4,
                solid_capstyle="butt", zorder=4)
        y_clique -= 0.045
    y_top = y_clique - 0.03

    step = 0.10
    counts = {"left": 0, "right": 0}
    y_min = y_top
    for i in order:
        s = side[i]
        y = y_top - step * counts[s]
        counts[s] += 1
        y_min = min(y_min, y)
        x_end = lo - 0.25 if s == "left" else hi + 0.25
        # Monochrome connectors (Demsar's own convention): each arm is named by
        # its own direct label, so hue would carry no information here.
        ax.plot([ranks[i], ranks[i]], [y_axis, y], color=INK_2, linewidth=0.9)
        ax.plot([ranks[i], x_end], [y, y], color=INK_2, linewidth=0.9)
        ax.text(x_end + (-0.06 if s == "left" else 0.06), y,
                f"{arms[i]} ({ranks[i]:.2f})",
                ha="right" if s == "left" else "left", va="center", fontsize=5.8,
                color=INK)

    ax.plot([lo, lo + cd], [y_axis + 0.115, y_axis + 0.115], color=CRITICAL,
            linewidth=1.8, solid_capstyle="butt")
    ax.text(lo + cd + 0.08, y_axis + 0.115, f"CD = {cd:.2f}", va="center",
            fontsize=6.5, color=CRITICAL)
    p_txt = "$p$ < 0.001" if fr["p"] < 5e-4 else f"$p$ = {fr['p']:.3f}"
    ax.text((lo + hi) / 2, y_axis + 0.27, f"Friedman $Q$ = {fr['Q']:.1f}, {p_txt}, "
                                          f"$n$ = {fr['n']} blocks",
            ha="center", va="bottom", fontsize=6.5, color=INK_2)
    ax.set_xlim(lo - 3.6, hi + 3.6)
    ax.set_ylim(y_min - 0.06, y_axis + 0.36)
    return _finish(fig, path)


# --------------------------------------------------------------------------- #
# 7. Attack fraction sweep.
# --------------------------------------------------------------------------- #


def fig_attack_fraction(ctx: Ctx, path: Path) -> bool:
    import matplotlib.pyplot as plt

    bank = ctx.bank
    # Attack is the hue, defense is the dash: two hues clear every colour-vision
    # gate pairwise, where four would put the flagged orange/yellow pair on one
    # axis. The composite encoding also reads as the comparison actually made.
    combos = (("sign_flip", "no_defense", "sign-flip, no defense", 0, "-"),
              ("sign_flip", "pipeline_krum", "sign-flip, pipeline", 0, "--"),
              ("adaptive_cosine", "no_defense", "adaptive, no defense", 1, "-"),
              ("adaptive_cosine", "pipeline_krum", "adaptive, pipeline", 1, "--"))
    models = ("cnn_lstm", "snn", "ssm")
    fig, axes = plt.subplots(len(models), 1, figsize=(COL_W, COL_W * 1.72),
                             sharex=True)
    missing: list[str] = []
    drawn = 0
    for ax, model in zip(axes, models, strict=True):
        for attack, arm, label, slot, dash in combos:
            colour = SERIES_ALLPAIRS[slot]
            xs, ys, los, his = [], [], [], []
            for n_att in (0, 1, 2, 3):
                vals = _fraction_cell(bank, n_att, attack, arm, model)
                if not vals:
                    continue
                xs.append(n_att)
                ys.append(float(np.mean(vals)))
                ci = R.bca(vals, ctx.resamples)
                los.append(ci[0] if ci else ys[-1])
                his.append(ci[1] if ci else ys[-1])
            if len(xs) < 2:
                missing.append(f"{label} ({R.MODEL_LABEL[model]})")
                continue
            ax.plot(xs, ys, color=colour, linestyle=dash, marker="o", label=label,
                    markeredgecolor=SURFACE, markeredgewidth=0.7)
            _band(ax, xs, los, his, colour)
            drawn += 1
        ax.set_ylabel(f"macro-F$_1$\n({R.MODEL_LABEL[model]})", fontsize=7)
        ax.set_ylim(bottom=0.0)
    if not drawn:
        plt.close(fig)
        return _no_data(path, "no attack-fraction sweep runs (block T2b) yet")
    axes[-1].set_xlabel("Byzantine substations (of 10)")
    axes[-1].set_xticks([0, 1, 2, 3])
    fig.subplots_adjust(hspace=0.12)
    _shared_legend(fig, axes[0], ncol=2)
    return _finish(fig, path, missing, y=-0.13)


def _fraction_cell(bank: R.Bank, n_att: int, attack: str, arm: str,
                   model: str) -> list[float]:
    """0 = the clean reference frame, 3 = T1b's f_max, 1 and 2 = the T2b sweep."""
    if n_att == 0:
        clean_arm = "fedavg" if arm == "no_defense" else arm
        return bank.f1(block="t1a", arm=clean_arm, model=model)
    if n_att == 3:
        return bank.f1(block="t1b", arm=arm, model=model, attack=attack)
    return [float(r.macro_f1) for r in bank.select(block="t2b", arm=arm, model=model,
                                                   attack=attack)
            if R._finite(r.macro_f1) and abs(r.attack_fraction - n_att / 10) < 1e-9]


# --------------------------------------------------------------------------- #
# 8. Fidelity-score distributions against the admission threshold.
# --------------------------------------------------------------------------- #


def fig_fidelity_dist(ctx: Ctx, path: Path) -> bool:
    import matplotlib.pyplot as plt

    groups = (("honest", "honest"), ("sign_flip", "sign-flip"),
              ("adaptive_cosine", "defense-\naware"))
    pooled: dict[str, list[float]] = {k: [] for k, _ in groups}
    taus: set[float] = set()
    for row in ctx.bank.rows:
        if not row.has_gate or row.block != "t1b":
            continue
        if row.attack not in ("sign_flip", "adaptive_cosine"):
            continue
        if row.tau is not None:
            taus.add(row.tau)
        for rnd in row.rounds:
            for c in rnd.get("clients", []):
                score = float(c.get("fidelity", 0.0))
                key = "honest" if c.get("attack") is None else row.attack
                if key in pooled:
                    pooled[key].append(score)
    present = [(k, lbl) for k, lbl in groups if pooled[k]]
    if not present:
        return _no_data(path, "no pipeline runs under attack carry per-round "
                              "fidelity scores yet")
    rng = np.random.default_rng(0)  # jitter only; never moves a value
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.66))
    ticks: list[str] = []
    for i, ((key, label), colour) in enumerate(zip(present, SERIES_ALLPAIRS,
                                                   strict=False)):
        vals = np.asarray(pooled[key], dtype=float)
        y = i + 1
        sample = vals if vals.size <= 2000 else rng.choice(vals, 2000, replace=False)
        ax.scatter(sample, y + rng.uniform(-0.16, 0.16, sample.size), s=1.2,
                   color=colour, alpha=0.25, linewidths=0, zorder=2)
        bp = _boxplot(ax, [vals], positions=[y], widths=0.42, patch_artist=True,
                      showfliers=False, zorder=3,
                      medianprops={"color": INK, "linewidth": 1.0},
                      whiskerprops={"color": colour, "linewidth": 0.8},
                      capprops={"color": colour, "linewidth": 0.8})
        bp["boxes"][0].set(facecolor=SURFACE, edgecolor=colour, linewidth=1.0,
                           alpha=0.9)
        ticks.append(f"{label}\n($n$ = {vals.size:,})".replace(",", " "))
    if len(taus) == 1:
        tau = taus.pop()
        ax.axvline(tau, color=CRITICAL, linewidth=1.0, zorder=1)
        ax.annotate(rf"admission threshold $\tau$ = {tau:g}",
                    (tau, len(present) + 0.52), xytext=(3, 0),
                    textcoords="offset points", fontsize=6.5, color=CRITICAL,
                    va="center")
    ax.set_yticks(range(1, len(present) + 1))
    ax.set_yticklabels(ticks, fontsize=6.5)
    ax.set_ylim(0.45, len(present) + 0.7)
    ax.set_xlabel("realized fidelity score")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    missing = [lbl for k, lbl in groups if not pooled[k]]
    return _finish(fig, path, missing)


# --------------------------------------------------------------------------- #
# 9. The LDP operating curve.
# --------------------------------------------------------------------------- #


def fig_ldp_curve(ctx: Ctx, path: Path) -> bool:
    import matplotlib.pyplot as plt

    from fedgrid.fl.ldp import epsilon as ldp_epsilon

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * GOLDEN))
    missing: list[str] = []
    drawn = 0
    # Operating points can coincide exactly (they do when the gate saturates), so
    # annotations are pooled per distinct point instead of overprinting.
    tags_at: dict[tuple[float, float], list[str]] = {}
    for tau, colour, marker in zip(R.LDP_TAUS, SERIES_ALLPAIRS, ("o", "s", "^"),
                                   strict=False):
        xs, ys = [], []
        for sigma in R.LDP_SIGMAS:
            rows = [r for r in ctx.bank.select(block="t2c", tau=tau)
                    if r.sigma is not None and abs(r.sigma - sigma) < 1e-9]
            gates = [g for g in (r.gate_stats() for r in rows) if g]
            if not gates:
                continue
            x = float(np.mean([g["attacker_admission"] for g in gates]))
            y = float(np.mean([g["honest_frr"] for g in gates]))
            xs.append(x)
            ys.append(y)
            n_rounds = next((r.rounds_planned for r in rows if r.rounds_planned), 0)
            eps = ldp_epsilon(sigma, n_rounds, R.EPSILON_DELTA) if n_rounds else None
            tag = rf"$\sigma$={sigma:g}" + (rf" ($\varepsilon$={eps:.0f})" if eps else "")
            tags_at.setdefault((round(x, 3), round(y, 3)), [])
            if tag not in tags_at[(round(x, 3), round(y, 3))]:
                tags_at[(round(x, 3), round(y, 3))].append(tag)
        if not xs:
            missing.append(rf"$\tau$={tau:g}")
            continue
        ax.plot(xs, ys, color=colour, marker=marker, markeredgecolor=SURFACE,
                markeredgewidth=0.7, label=rf"$\tau$ = {tau:g}", alpha=0.85)
        drawn += 1
    for (x, y), tags in tags_at.items():
        ax.annotate("\n".join(tags), (x, y), textcoords="offset points",
                    xytext=(5, -1), fontsize=5.8, color=INK_2, va="top")
    if not drawn:
        plt.close(fig)
        return _no_data(path, "no LDP grid runs (block T2c) carry gate statistics yet")
    ax.set_xlabel("defense-aware attacker admission rate")
    ax.set_ylabel("honest-client false-rejection rate")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    _legend(ax, ncol=1, loc="lower right")
    return _finish(fig, path, missing)


# --------------------------------------------------------------------------- #
# 10. Heterogeneity sweep.
# --------------------------------------------------------------------------- #


#: Legend-width-safe spellings of the two swept arms (the tables keep the long form).
_ALPHA_ARM_LABELS: tuple[tuple[str, str], ...] = (("FedAvg", "fedavg"),
                                                  ("Pipeline+Krum", "pipeline_krum"))


def fig_alpha(ctx: Ctx, path: Path) -> bool:
    import matplotlib.pyplot as plt

    order = [(0.1, "0.1"), (0.5, "0.5"), (1.0, "1.0"), (100.0, "near-\nuniform")]
    models = ("cnn_lstm", "snn", "ssm")
    fig, axes = plt.subplots(len(models), 1, figsize=(COL_W, COL_W * 1.72),
                             sharex=True)
    missing: list[str] = []
    drawn = 0
    for ax, model in zip(axes, models, strict=True):
        for (arm_label, arm), colour in zip(_ALPHA_ARM_LABELS, SERIES, strict=False):
            for attack, dash, suffix in ((None, "-", "clean"),
                                         ("adaptive_cosine", "--", "adaptive")):
                xs, ys, los, his = [], [], [], []
                for i, (alpha, _lbl) in enumerate(order):
                    vals, _planned = R._alpha_cell(ctx.bank, alpha, arm, model, attack)
                    if not vals:
                        continue
                    xs.append(i)
                    ys.append(float(np.mean(vals)))
                    ci = R.bca(vals, ctx.resamples)
                    los.append(ci[0] if ci else ys[-1])
                    his.append(ci[1] if ci else ys[-1])
                if len(xs) < 2:
                    missing.append(f"{arm_label} {suffix} ({R.MODEL_LABEL[model]})")
                    continue
                ax.plot(xs, ys, color=colour, linestyle=dash, marker="o",
                        markeredgecolor=SURFACE, markeredgewidth=0.7,
                        label=f"{arm_label}, {suffix}")
                _band(ax, xs, los, his, colour)
                drawn += 1
        ax.set_ylabel(f"macro-F$_1$\n({R.MODEL_LABEL[model]})", fontsize=7)
        ax.set_ylim(bottom=0.0)
    if not drawn:
        plt.close(fig)
        return _no_data(path, "no heterogeneity sweep runs (block T2a) yet")
    axes[-1].set_xticks(range(len(order)))
    axes[-1].set_xticklabels([lbl for _a, lbl in order], fontsize=6.5)
    axes[-1].set_xlabel(r"Dirichlet concentration $\alpha$ (less skewed $\rightarrow$)")
    fig.subplots_adjust(hspace=0.12)
    _shared_legend(fig, axes[0], ncol=2, y=-0.04)
    return _finish(fig, path, missing, y=-0.15)


# --------------------------------------------------------------------------- #
# 11-17. Body figures: the manuscript's tables re-drawn as data graphics.
#
# Each renderer reads the same Bank cells as its ``experiments.report`` table
# twin -- same blocks, same arms, same pooling -- so a figure can never drift
# from the number the corresponding Supplementary table prints.
#
# Model hue mapping is fixed across the whole figure set (the same assignment
# ``fig_energy_accuracy`` uses): CNN-LSTM = slot 2, Spiking = slot 0 (rate
# coding = slot 1), Mamba = the neutral ink, since it has no validated slot.
# --------------------------------------------------------------------------- #

MODEL_HUE: dict[str, str] = {"cnn_lstm": SERIES_ALLPAIRS[2],
                             "snn": SERIES_ALLPAIRS[0], "ssm": INK_2}
#: Short arm spellings for axes -- the tables keep the long form in prose.
_FRAME_ARM_SHORT: tuple[tuple[str, str], ...] = (
    ("centralized", "Central-\nized"), ("local", "Local-\nonly"),
    ("fedavg", "FedAvg"), ("fedprox", "FedProx"),
    ("pipeline_krum", "Pipe.\n+Krum"), ("pipeline_geomed", "Pipe.\n+geomed"))


def _sd(vals: Sequence[float]) -> float:
    return float(np.std(np.asarray(vals, dtype=float), ddof=1)) if len(vals) > 1 else 0.0


def fig_split_frame(ctx: Ctx, path: Path) -> bool:
    """Table tab_temporal_frame -> grouped bars, random vs chronological split."""
    import matplotlib.pyplot as plt

    models = R.MODEL_ORDER
    fig, axes = plt.subplots(len(models), 1,
                             figsize=(COL_W, COL_W * 1.55), sharex=True)
    missing: list[str] = []
    drawn = 0
    w = 0.36
    for ax, model in zip(axes, models, strict=True):
        for i, (arm, _short) in enumerate(_FRAME_ARM_SHORT):
            rand = ctx.bank.f1(block="t1a", arm=arm, model=model)
            temp = ctx.bank.f1(block="t1a_temporal", arm=arm, model=model)
            if rand:
                ax.bar(i - w / 2, float(np.mean(rand)), w, yerr=_sd(rand),
                       color=MUTED, edgecolor="none", capsize=1.4,
                       error_kw={"elinewidth": 0.7, "ecolor": INK_2},
                       label="random split" if i == 0 and model == models[0]
                       else None, zorder=2)
            if temp:
                ax.bar(i + w / 2, float(np.mean(temp)), w, yerr=_sd(temp),
                       color=SERIES[0], edgecolor="none", capsize=1.4,
                       error_kw={"elinewidth": 0.7, "ecolor": INK_2},
                       label="chronological" if i == 0 and model == models[0]
                       else None, zorder=3)
            if rand and temp:
                d = float(np.mean(temp)) - float(np.mean(rand))
                top = max(float(np.mean(rand)) + _sd(rand),
                          float(np.mean(temp)) + _sd(temp))
                ax.annotate(f"{d:+.2f}", (i, top + 0.02), ha="center",
                            va="bottom", fontsize=5.3, color=INK_2)
                drawn += 1
            if not rand and not temp:
                missing.append(f"{arm} ({R.MODEL_LABEL[model]})")
        ax.set_ylabel(f"macro-F$_1$\n({R.MODEL_LABEL[model]})", fontsize=7)
        ax.set_ylim(bottom=0.0)
        ax.grid(axis="x", visible=False)
    if not drawn:
        plt.close(fig)
        return _no_data(path, "no T1a / T1a_temporal reference-frame runs")
    axes[-1].set_xticks(range(len(_FRAME_ARM_SHORT)))
    axes[-1].set_xticklabels([s for _a, s in _FRAME_ARM_SHORT], fontsize=6.3)
    fig.subplots_adjust(hspace=0.14)
    _shared_legend(fig, axes[0], ncol=2, y=-0.02)
    return _finish(fig, path, missing, y=-0.115)


_ATTACK_SHORT: dict[str | None, str] = {
    None: "Clean", "sign_flip": "Sign-flip", "gaussian_noise": "Gaussian",
    "label_flip": "Label-flip", "adaptive_cosine": "Adaptive"}
_DEFENSE_SHORT: dict[str, str] = {
    "no_defense": "No defense", "krum_alone": "Krum\nalone",
    "geomed_alone": "Geo-median\nalone", "pipeline_krum": "Pipeline\n+Krum",
    "pipeline_geomed": "Pipeline\n+geomed"}


def fig_attack_heatmap(ctx: Ctx, path: Path) -> bool:
    """Table tab_temporal_attack -> defence x attack heatmap per model.

    Cell colour is the chronological-split macro-F1 mean; the small second
    line is the change against the matched random-split cell, exactly the
    parenthesised value the table prints.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)
    attacks = list(R.ATTACK_ORDER)
    arms = [arm for _label, arm in R._DEFENSE_ROWS]
    models = R.MODEL_ORDER
    fig, axes = plt.subplots(len(models), 1,
                             figsize=(COL_W * 0.9, COL_W * 1.9))
    grids: list[tuple[Any, np.ndarray, np.ndarray]] = []
    missing: set[str] = set()
    for ax, model in zip(axes, models, strict=True):
        val = np.full((len(arms), len(attacks)), np.nan)
        delta = np.full((len(arms), len(attacks)), np.nan)
        for i, arm in enumerate(arms):
            for j, attack in enumerate(attacks):
                if attack is None:
                    twin = R._CLEAN_TWIN[arm]
                    if twin is None:
                        continue
                    rand = ctx.bank.f1(block="t1a", arm=twin, model=model)
                    temp = ctx.bank.f1(block="t1a_temporal", arm=twin,
                                       model=model)
                else:
                    rand = ctx.bank.f1(block="t1b", arm=arm, model=model,
                                       attack=attack)
                    temp = ctx.bank.f1(block="t1b_temporal", arm=arm,
                                       model=model, attack=attack)
                if not temp:
                    missing.add(f"{arm}/{R._atk_label(attack)} "
                                f"({R.MODEL_LABEL[model]})")
                    continue
                val[i, j] = float(np.mean(temp))
                if rand:
                    delta[i, j] = val[i, j] - float(np.mean(rand))
        grids.append((ax, val, delta))
    if all(np.isnan(g[1]).all() for g in grids):
        plt.close(fig)
        return _no_data(path, "no T1b_temporal runs for the attack grid")
    vmax = float(np.nanmax([np.nanmax(g[1]) for g in grids
                            if not np.isnan(g[1]).all()]))
    for (ax, val, delta), model in zip(grids, models, strict=True):
        ax.imshow(np.where(np.isnan(val), 0.0, val), cmap=cmap, vmin=0.0,
                  vmax=vmax, aspect="auto", interpolation="nearest")
        n_i, n_j = val.shape
        for i in range(n_i):
            for j in range(n_j):
                if np.isnan(val[i, j]):
                    ax.text(j, i, "--", ha="center", va="center",
                            fontsize=6, color=MUTED)
                    continue
                dark = val[i, j] > 0.62 * vmax
                fg = "#ffffff" if dark else INK
                ax.text(j, i - 0.13, f"{val[i, j]:.2f}", ha="center",
                        va="center", fontsize=5.9, color=fg)
                if np.isfinite(delta[i, j]):
                    ax.text(j, i + 0.24, f"{delta[i, j]:+.2f}", ha="center",
                            va="center", fontsize=4.9,
                            color=("#dfe9f6" if dark else INK_2))
        ax.set_xticks(range(n_j))
        ax.set_xticklabels([_ATTACK_SHORT[a] for a in attacks], fontsize=5.9)
        ax.set_yticks(range(n_i))
        ax.set_yticklabels([_DEFENSE_SHORT[a] for a in arms], fontsize=5.9)
        ax.set_title(R.MODEL_LABEL[model], fontsize=7, color=INK, pad=3)
        ax.grid(False)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-0.5, n_j), minor=True)
        ax.set_yticks(np.arange(-0.5, n_i), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=1.1)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cb = fig.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(0.0, vmax), cmap=cmap),
        ax=[g[0] for g in grids], fraction=0.02, pad=0.015, aspect=30)
    cb.set_label("macro-F$_1$ (chronological)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5, length=0)
    cb.outline.set_visible(False)
    fig.subplots_adjust(hspace=0.5)
    return _finish(fig, path, sorted(missing), y=-0.055)


def fig_magnitude(ctx: Ctx, path: Path) -> bool:
    """Table tab_magnitude -> the in-cone magnitude lever, per model + gate."""
    import matplotlib.pyplot as plt

    scales = list(R.MAG_SCALES)
    xs = [math.log2(s) for s in scales]
    arm_style: dict[str, tuple[str, str]] = {
        "no_defense": (INK_2, "-"), "geomed_alone": (SERIES[0], "-"),
        "pipeline_krum": (SERIES[1], "--"), "pipeline_geomed": (SERIES[6], "--")}
    models = R.MODEL_ORDER
    fig, axes = plt.subplots(len(models) + 1, 1,
                             figsize=(COL_W, COL_W * 1.9), sharex=True)
    missing: list[str] = []
    drawn = 0
    for ax, model in zip(axes[:len(models)], models, strict=True):
        for label, arm in R.MAG_ARMS:
            ys, los, his = [], [], []
            for s in scales:
                vals = [float(r.macro_f1)
                        for r in R._mag_rows(ctx.bank, s, arm, model)
                        if R._finite(r.macro_f1)]
                if not vals:
                    continue
                ys.append(float(np.mean(vals)))
                ci = R.bca(vals, ctx.resamples)
                los.append(ci[0] if ci else ys[-1])
                his.append(ci[1] if ci else ys[-1])
            if len(ys) < 2:
                missing.append(f"{label} ({R.MODEL_LABEL[model]})")
                continue
            xx = [math.log2(scales[k]) for k in range(len(ys))]
            colour, dash = arm_style[arm]
            ax.plot(xx, ys, color=colour, linestyle=dash, marker="o",
                    markeredgecolor=SURFACE, markeredgewidth=0.7, label=label)
            _band(ax, xx, los, his, colour)
            drawn += 1
        ax.set_ylabel(f"macro-F$_1$\n({R.MODEL_LABEL[model]})", fontsize=7)
        ax.set_ylim(bottom=0.0)
    # Gate panel: the two pipeline arms run the same admission stage; the
    # table reports them per arm, the figure pools them (identical statistic).
    ax = axes[-1]
    for key, label, colour in (
            ("honest_frr", "honest false-rejection", CRITICAL),
            ("attacker_admission", "attacker admission", SERIES[0])):
        ys, los, his = [], [], []
        for s in scales:
            rows = [r for arm in ("pipeline_krum", "pipeline_geomed")
                    for r in R._mag_rows(ctx.bank, s, arm)]
            gates = [g for g in (r.gate_stats(True) for r in rows) if g]
            vals = [g[key] for g in gates if R._finite(g[key])]
            if not vals:
                continue
            ys.append(float(np.mean(vals)))
            ci = R.bca(vals, ctx.resamples)
            los.append(ci[0] if ci else ys[-1])
            his.append(ci[1] if ci else ys[-1])
        if len(ys) < 2:
            missing.append(f"gate {label}")
            continue
        xx = [math.log2(scales[k]) for k in range(len(ys))]
        ax.plot(xx, ys, color=colour, marker="s", markeredgecolor=SURFACE,
                markeredgewidth=0.7, label=label)
        _band(ax, xx, los, his, colour)
        drawn += 1
    ax.set_ylabel("gate rate\n(pooled)", fontsize=7)
    ax.set_ylim(-0.03, 1.03)
    if not drawn:
        plt.close(fig)
        return _no_data(path, "no T4mag attacker-magnitude runs yet")
    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels([f"{s:g}×" for s in scales], fontsize=6.5)
    axes[-1].set_xlabel("attacker magnitude scale (in-cone, log$_2$ axis)")
    fig.subplots_adjust(hspace=0.14)
    _shared_legend(fig, axes[0], ncol=2, y=-0.02)
    return _finish(fig, path, missing, y=-0.10)


def fig_ldp_tau(ctx: Ctx, path: Path) -> bool:
    """Table tab_ldp_tau_extended -> the LDP operating curve against tau."""
    import matplotlib.pyplot as plt

    from fedgrid.fl.ldp import epsilon as ldp_epsilon

    taus = [t for t, _b in R.LDP_TAUS_EXTENDED]
    sigmas = list(R.LDP_SIGMAS)
    # cell[(tau, sigma)] -> rows, deduplicated exactly as the table does.
    cells: dict[tuple[float, float], list[Any]] = {}
    rounds_at: dict[float, int] = {}
    for tau, block in R.LDP_TAUS_EXTENDED:
        for sigma in sigmas:
            rows = R._dedupe([r for r in ctx.bank.select(block=block, tau=tau)
                              if r.sigma is not None
                              and abs(r.sigma - sigma) < 1e-9])
            cells[(tau, sigma)] = rows
            n = next((r.rounds_planned for r in rows if r.rounds_planned), 0)
            if n:
                rounds_at[tau] = n
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, COL_W * 1.18), sharex=True)
    missing: list[str] = []
    drawn = 0
    # Panel A: gate rates vs tau, mean over sigma with the across-sigma range.
    for key, label, colour in (
            ("honest_frr", "honest false-rejection", CRITICAL),
            ("attacker_admission", "attacker admission", SERIES[0])):
        ys, lo, hi = [], [], []
        xs = []
        for tau in taus:
            per_sigma = []
            for sigma in sigmas:
                gates = [g for g in (r.gate_stats() for r in cells[(tau, sigma)])
                         if g]
                vals = [g[key] for g in gates if R._finite(g[key])]
                if vals:
                    per_sigma.append(float(np.mean(vals)))
            if per_sigma:
                xs.append(tau)
                ys.append(float(np.mean(per_sigma)))
                lo.append(min(per_sigma))
                hi.append(max(per_sigma))
        if len(xs) < 2:
            missing.append(label)
            continue
        axes[0].plot(xs, ys, color=colour, marker="o",
                     markeredgecolor=SURFACE, markeredgewidth=0.7, label=label)
        _band(axes[0], xs, lo, hi, colour)
        drawn += 1
    axes[0].set_ylabel("gate rate", fontsize=7)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].axvspan(0.72, 0.78, color=GRID, alpha=0.6, linewidth=0, zorder=0)
    axes[0].annotate("gate first\nopens", (0.75, 0.6), fontsize=5.6,
                     color=INK_2, ha="center", va="top")
    # Panel B: macro-F1 vs tau per model (converged runs only, as the table).
    for model in R.MODEL_ORDER:
        ys, lo, hi = [], [], []
        xs = []
        for tau in taus:
            per_sigma = []
            for sigma in sigmas:
                ok, _bad = R._split_diverged(
                    [r for r in cells[(tau, sigma)] if r.model == model])
                vals = [float(r.macro_f1) for r in ok
                        if R._finite(r.macro_f1)]
                if vals:
                    per_sigma.append(float(np.mean(vals)))
            if per_sigma:
                xs.append(tau)
                ys.append(float(np.mean(per_sigma)))
                lo.append(min(per_sigma))
                hi.append(max(per_sigma))
        if len(xs) < 2:
            missing.append(R.MODEL_LABEL[model])
            continue
        axes[1].plot(xs, ys, color=MODEL_HUE[model], marker="o",
                     markeredgecolor=SURFACE, markeredgewidth=0.7,
                     label=R.MODEL_LABEL[model])
        _band(axes[1], xs, lo, hi, MODEL_HUE[model])
        drawn += 1
    axes[1].set_ylabel("macro-F$_1$", fontsize=7)
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_xlabel(r"admission threshold $\tau$ (local DP, all $\sigma$)")
    if not drawn:
        plt.close(fig)
        return _no_data(path, "no extended LDP grid runs (T2c/T3tau) yet")
    eps0 = ldp_epsilon(sigmas[-1], next(iter(rounds_at.values()), 30),
                       R.EPSILON_DELTA)
    eps1 = ldp_epsilon(sigmas[0], next(iter(rounds_at.values()), 30),
                       R.EPSILON_DELTA)
    axes[0].annotate(rf"$\varepsilon$ from {eps0:.0f} to {eps1:.0f} "
                     r"($\sigma$-indexed)", (0.06, 0.05),
                     xycoords="axes fraction", fontsize=5.8, color=INK_2)
    _shared_legend(fig, axes[0], ncol=2, y=-0.02)
    fig.subplots_adjust(hspace=0.16)
    return _finish(fig, path, missing, y=-0.13)


def fig_energy_bars(ctx: Ctx, path: Path) -> bool:
    """Table tab_energy -> estimated per-inference energy, log scale."""
    import matplotlib.pyplot as plt

    from fedgrid.metrics.energy import E_MAC_INT8_PJ, E_MAC_PJ, energy_uj

    bank = ctx.bank
    snn = {"delta": [r for r in bank.select(block="t1a", arm="fedavg",
                                            model="snn", attack=None)
                     if r.coding == "delta" and r.t_steps == R.ENERGY_T_STEPS],
           "rate": [r for r in bank.select(block="t2e", arm="fedavg",
                                           model="snn", attack=None)
                    if r.coding == "rate" and r.t_steps == R.ENERGY_T_STEPS]}
    macs = next((float(r.energy["ann_cnn_lstm_macs"])
                 for r in (snn["delta"] + snn["rate"])
                 if r.energy and R._finite(r.energy.get("ann_cnn_lstm_macs"))),
                None)
    ssm_ref = bank.select(block="t1a", arm="fedavg", model="ssm", attack=None)
    ssm_macs = next((float(r.energy["macs_per_inference"]) for r in ssm_ref
                     if r.energy
                     and R._finite(r.energy.get("macs_per_inference"))), None)

    # rows: (label, uJ, ops text, colour, precision group) -- one per table row.
    rows: list[tuple[str, float | None, str, str, str]] = []
    missing: list[str] = []
    for precision in ("fp32", "INT8"):
        for coding in ("delta", "rate"):
            e = R._energy_cells(snn[coding], precision)
            label = "SNN-δ" if coding == "delta" else "SNN-rate"
            if e["uj"] is None:
                missing.append(f"{label} {precision}")
                rows.append((label, None, "--", MODEL_HUE["snn"], precision))
            else:
                rows.append((label, e["uj"],
                             f"{e['synops']:,.0f} SynOps",
                             MODEL_HUE["snn"] if coding == "delta"
                             else SERIES_ALLPAIRS[1], precision))
        cnn_uj = (macs * (E_MAC_INT8_PJ if precision == "INT8" else E_MAC_PJ)
                  * 1e-6 if macs is not None else None)
        if cnn_uj is None:
            missing.append(f"CNN-LSTM {precision}")
            rows.append(("CNN-LSTM", None, "--", MODEL_HUE["cnn_lstm"],
                         precision))
        else:
            rows.append(("CNN-LSTM", cnn_uj, f"{macs:,.0f} MACs",
                         MODEL_HUE["cnn_lstm"], precision))
        ssm_uj = (ssm_macs * (E_MAC_INT8_PJ if precision == "INT8" else E_MAC_PJ)
                  * 1e-6 if ssm_macs is not None else None)
        rows.append(("Mamba", ssm_uj,
                     f"{ssm_macs:,.0f} MACs" if ssm_macs is not None
                     else "not instrumented",
                     MODEL_HUE["ssm"], precision))
        if ssm_uj is None:
            missing.append(f"Mamba {precision} (no MAC counter)")
    if all(v is None for _l, v, _o, _c, _p in rows):
        return _no_data(path, "no runs carry an energy block")

    fig, ax = plt.subplots(figsize=(COL_W * 0.9, COL_W * 0.95))
    y = 0.0
    yticks, ylabels = [], []
    for gi, precision in enumerate(("fp32", "INT8")):
        group = [r for r in rows if r[4] == precision]
        for label, uj, ops, colour, _p in group:
            yticks.append(y)
            ylabels.append(f"{label} · {precision}")
            if uj is not None:
                ax.barh(y, uj, height=0.62, color=colour, edgecolor="none",
                        zorder=3)
                # Long bars take the value inside (white); short bars take it
                # just past the tip, so neither collides with the ops column.
                if uj > 0.15:
                    ax.annotate(f"{uj:.3g} µJ", (uj * 0.88, y),
                                va="center", ha="right", fontsize=5.8,
                                color="white", fontweight="bold", zorder=4)
                else:
                    ax.annotate(f"{uj:.3g} µJ", (uj * 1.2, y), va="center",
                                fontsize=5.8, color=INK_2, zorder=4)
                ax.annotate(ops, (0.985, y), xycoords=("axes fraction",
                            "data"), va="center", ha="right",
                            fontsize=5.4, color=MUTED)
            else:
                ax.annotate(f"--   ({ops})", (0.985, y),
                            xycoords=("axes fraction", "data"),
                            va="center", ha="right", fontsize=5.4,
                            color=MUTED)
            y -= 1.0
        y -= 0.5  # gap between precision groups
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=5.9)
    ax.set_xscale("log")
    ax.set_xlim(1.5e-3, 60.0)
    ax.set_xlabel(r"est.\ energy per inference ($\mu$J, log)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    return _finish(fig, path, missing, prefix="not measured", y=-0.13)


def fig_latency_env(ctx: Ctx, path: Path) -> bool:
    """Table tab_latency -> p50-p99 spans per model x device vs the 4 ms line."""
    import matplotlib.pyplot as plt

    from fedgrid.metrics.latency import GOOSE_BUDGET_MS

    measured: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in ctx.bank.rows:
        lat = r.latency
        if isinstance(lat, dict) and lat:
            for device in ("cpu", "cuda"):
                blk = lat.get(device, lat if "p99_ms" in lat else None)
                if isinstance(blk, dict) and R._finite(blk.get("p99_ms")):
                    measured.setdefault((r.model, device), []).append(blk)
    models = R.MODEL_ORDER
    devices = (("cpu", "CPU"), ("cuda", "GPU"))
    fig, ax = plt.subplots(figsize=(COL_W * 0.9, COL_W * 0.78))
    y = 0.0
    yticks, ylabels = [], []
    missing: list[str] = []
    drawn = 0
    for model in models:
        for device, dev_label in devices:
            blks = measured.get((model, device), [])
            stats = {k: [float(b[k]) for b in blks if R._finite(b.get(k))]
                     for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms")}
            if not stats["p99_ms"] or not stats["p50_ms"]:
                missing.append(f"{R.MODEL_LABEL[model]} {dev_label}")
                continue
            p50, p95, p99 = (float(np.mean(stats[k]))
                             for k in ("p50_ms", "p95_ms", "p99_ms"))
            mean = float(np.mean(stats["mean_ms"])) if stats["mean_ms"] else None
            colour = MODEL_HUE[model]
            ax.barh(y, p99 - p50, left=p50, height=0.6, color=colour,
                    alpha=0.32, edgecolor=colour, linewidth=0.7, zorder=3)
            ax.plot([p95, p95], [y - 0.3, y + 0.3], color=colour,
                    linewidth=1.0, zorder=4)
            if mean is not None:
                ax.plot(mean, y, marker="D", markersize=2.6, color=INK,
                        markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=5)
            share = 100.0 * p99 / GOOSE_BUDGET_MS
            ax.annotate(f"p99 {p99:.1f} ms ({share:.0f}%)", (0.985, y),
                        xycoords=("axes fraction", "data"), va="center",
                        ha="right", fontsize=5.4, color=INK_2)
            yticks.append(y)
            ylabels.append(f"{R.MODEL_LABEL[model]} · {dev_label}")
            y -= 1.0
            drawn += 1
        y -= 0.45
    if not drawn:
        plt.close(fig)
        return _no_data(path, "no RunResult carries a latency block "
                              "(the n=1000 p50/p95/p99 measurement pass "
                              "has not run)")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=6.2)
    ax.axvline(GOOSE_BUDGET_MS, color=CRITICAL, linewidth=1.0, zorder=1)
    ax.annotate(f"{GOOSE_BUDGET_MS:g} ms fast-message envelope",
                (GOOSE_BUDGET_MS, 0.4), xytext=(3, 0),
                textcoords="offset points", fontsize=6.0, color=CRITICAL,
                va="bottom", rotation=90)
    ax.set_xlabel("inference latency (ms)")
    ax.set_xlim(0.0, 33.0)
    ax.set_ylim(y + 0.2, 0.6)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    return _finish(fig, path, missing, y=-0.16)


def fig_perclass(ctx: Ctx, path: Path) -> bool:
    """Table tab_per_class -> per-class F1 and recall at the FedAvg reference."""
    import matplotlib.pyplot as plt

    models = ("cnn_lstm", "snn")
    n_cls = len(CLASS_SHORT)
    stats: dict[str, dict[str, list[list[float]]]] = {}
    support: list[list[float]] = []
    for model in models:
        rows = ctx.bank.select(block="t1a", arm="fedavg", model=model)
        f1s, recs = [], []
        for r in rows:
            if r.per_class_f1:
                f1s.append([float(v) for v in r.per_class_f1])
            if r.confusion:
                conf = np.asarray(r.confusion, dtype=float)
                tot = conf.sum(axis=1)
                recs.append([(conf[i, i] / tot[i]) if tot[i] else float("nan")
                             for i in range(conf.shape[0])])
                if model == "cnn_lstm":
                    support.append([float(v) for v in tot])
        stats[model] = {"f1": f1s, "recall": recs}
    if not any(stats[m][k] for m in models for k in ("f1", "recall")):
        return _no_data(path, "no T1a FedAvg runs carry per-class metrics")

    fig, axes = plt.subplots(2, 1, figsize=(COL_W, COL_W * 1.05), sharex=True)
    w = 0.38
    missing: list[str] = []
    for ax, key, ylabel in ((axes[0], "f1", "per-class F$_1$"),
                            (axes[1], "recall", "per-class recall")):
        for mi, model in enumerate(models):
            rows_v = stats[model][key]
            if not rows_v:
                missing.append(f"{key} {R.MODEL_LABEL[model]}")
                continue
            means, sds = [], []
            for c in range(n_cls):
                vals = [v[c] for v in rows_v
                        if c < len(v) and R._finite(v[c])]
                means.append(float(np.nanmean(vals)) if vals else np.nan)
                sds.append(float(np.nanstd(vals, ddof=1))
                           if len(vals) > 1 else 0.0)
            xs = np.arange(n_cls) + (mi - 0.5) * w
            ax.bar(xs, means, w * 0.92, yerr=sds, color=MODEL_HUE[model],
                   edgecolor="none", capsize=1.0,
                   error_kw={"elinewidth": 0.6, "ecolor": INK_2},
                   label=R.MODEL_LABEL[model], zorder=3)
            if key == "f1":
                macro = ctx.bank.f1(block="t1a", arm="fedavg", model=model)
            else:
                macro = [float(np.nanmean(v)) for v in rows_v if v]
            if macro:
                ax.axhline(float(np.mean(macro)), color=MODEL_HUE[model],
                           linestyle="--", linewidth=0.7, alpha=0.7, zorder=2)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_ylim(bottom=0.0)
        ax.grid(axis="x", visible=False)
    sup_lbls = []
    for c in range(n_cls):
        sup = [s[c] for s in support if c < len(s)]
        sup_lbls.append(f"{CLASS_SHORT[c]}\n{int(round(np.mean(sup))):,}"
                        .replace(",", "\u2009") if sup else CLASS_SHORT[c])
    axes[1].set_xticks(range(n_cls))
    axes[1].set_xticklabels(sup_lbls, fontsize=5.8)
    axes[1].set_xlabel("event class (test support)")
    _shared_legend(fig, axes[0], ncol=2, y=-0.02)
    fig.subplots_adjust(hspace=0.14)
    return _finish(fig, path, missing, y=-0.12)


# --------------------------------------------------------------------------- #
# Context + driver.
# --------------------------------------------------------------------------- #


class Ctx:
    """Everything the figure functions read: loaded runs, federations, statistics."""

    def __init__(self, bank: R.Bank, feds: list[dict[str, Any]],
                 stats: dict[str, Any], data_dir: Path | None,
                 resamples: int) -> None:
        self.bank = bank
        self.feds = feds
        self.stats = stats
        self.data_dir = data_dir
        self.resamples = resamples

    def reference_federation_dir(self) -> Path | None:
        """Directory of the k=10, alpha=0.5, seed=0 federation, if it is staged."""
        if self.data_dir is None:
            return None
        for man in sorted(Path(self.data_dir).glob("*/manifest.json")):
            try:
                cfg = A.json.loads(man.read_text(encoding="utf-8")).get("config", {})
            except (OSError, ValueError):
                continue
            if cfg.get("alpha") == 0.5 and cfg.get("seed") == 0:
                return man.parent
        return None

    def client_for_feeder(self, fed_dir: Path,
                          feeder: str) -> tuple[int | None, str]:
        try:
            man = A.json.loads((fed_dir / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None, ""
        for c in man.get("clients", []):
            if R._get(c, "feeder", "name", default="") == feeder:
                return int(c["cid"]), R.FEEDER_LABEL.get(feeder, feeder)
        return None, R.FEEDER_LABEL.get(feeder, feeder)


#: Hook name -> renderer, in the order the manuscript introduces them.
FIGURES: dict[str, Any] = {
    "fig_label_histogram": fig_label_histogram,
    "fig_example_windows": fig_example_windows,
    "fig_rounds": fig_rounds,
    "fig_energy_accuracy": fig_energy_accuracy,
    "fig_latency_goose": fig_latency_goose,
    "fig_cd_diagram": fig_cd_diagram,
    "fig_attack_fraction": fig_attack_fraction,
    "fig_fidelity_dist": fig_fidelity_dist,
    "fig_ldp_curve": fig_ldp_curve,
    "fig_alpha": fig_alpha,
    # Body figures (the manuscript's tables re-drawn as data graphics).
    "fig_split_frame": fig_split_frame,
    "fig_attack_heatmap": fig_attack_heatmap,
    "fig_magnitude": fig_magnitude,
    "fig_ldp_tau": fig_ldp_tau,
    "fig_energy_bars": fig_energy_bars,
    "fig_latency_env": fig_latency_env,
    "fig_perclass": fig_perclass,
}


def render_all(runs_dir: Path | str | Sequence[Path | str], out_dir: Path | str,
               data_dir: Path | str | None = None,
               resamples: int = 2000) -> dict[str, bool]:
    """Render every figure to ``out_dir``; ``True`` = drawn from data, ``False`` = placeholder."""
    import matplotlib

    matplotlib.use("Agg")
    style()
    if isinstance(runs_dir, (list, tuple)):
        rows = R.load_corpus(list(runs_dir))
    else:
        rows = R.load_rows(runs_dir)
    bank = R.Bank(rows)
    feds = R.load_federations(data_dir) if data_dir else []
    stats = R.statistical_battery(bank, resamples)
    ctx = Ctx(bank, feds, stats, Path(data_dir) if data_dir else None, resamples)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    status: dict[str, bool] = {}
    for name, fn in FIGURES.items():
        target = out / f"{name}.pdf"
        try:
            status[name] = bool(fn(ctx, target))
        except Exception as exc:  # noqa: BLE001 - one bad figure must not kill the rest
            status[name] = _no_data(target, f"renderer failed: {type(exc).__name__}: {exc}")
    return status


def _pdf_width_pt(path: Path) -> float | None:
    """Natural width of a saved PDF, in PostScript points, from its MediaBox."""
    import re

    match = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+[\d.]+\s+([\d.]+)",
                      path.read_bytes())
    return float(match.group(2)) - float(match.group(1)) if match else None


def column_scale(path: Path) -> float | None:
    """Factor ``width=\\linewidth`` will apply to this PDF; 1.0 means no rescaling.

    Anything away from 1.0 means that figure's type renders at a different size
    from the rest, so the driver prints it and the tests assert on it. Font size
    cannot fix a text-driven width (scaling the type scales the box with it) --
    the fix is always fewer or shorter labels.
    """
    width = _pdf_width_pt(path)
    return None if width is None else COL_W_PT / width


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m experiments.figures",
        description="Render the manuscript's figures as vector PDFs.")
    p.add_argument("--runs-dir", type=Path, nargs="+", default=[Path("runs")],
                   help="run directories in increasing precedence (later wins on "
                        "collision): the paper corpus is runs_snn_fixed runs "
                        "runs_x3 runs_ssm_x3 runs_gap_x3")
    p.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
    p.add_argument("--data-dir", type=Path, default=Path("data_battery"))
    p.add_argument("--resamples", type=int, default=2000,
                   help="BCa resamples for the figure bands (10000 for final art)")
    args = p.parse_args(argv)

    status = render_all(args.runs_dir, args.out_dir, args.data_dir, args.resamples)
    for name, ok in status.items():
        scale = column_scale(Path(args.out_dir) / f"{name}.pdf")
        flag = "" if scale is None or scale >= 0.98 else f"  [WIDE: x{scale:.2f}]"
        print(f"  {name:<24} {'drawn' if ok else 'PLACEHOLDER (no data)'}{flag}")
    drawn = sum(status.values())
    print(f"wrote {len(status)} PDFs -> {args.out_dir} ({drawn} from data, "
          f"{len(status) - drawn} placeholders)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
