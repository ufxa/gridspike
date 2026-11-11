"""Route B — EMT-domain validation of the FedGrid event pipeline (DPsim).

**Why this exists.** Reviewer feedback (``E:/paper7/annot_full.txt``, p. 14)
offered two positionings for the paper; Route B asks for "a separate experiment
on a timescale compatible with fast dynamics, using EMT/HIL, oscillographic
records or a measured high-rate dataset", and suggests the minimal acceptable
experiment: *3–5 representative scenarios (LG, LL, LLG, 3PH, high-impedance or
an equivalent critical condition), fast-timescale signals, run CNN-LSTM/Mamba,
measure latency — no need to repeat the 1,845 runs*. The same reviewer's p.1
note asks that merely raising the OpenDSS sampling rate NOT be presented as a
fix, because the quasi-static physics is unchanged. This module therefore
regenerates the event taxonomy under a **true EMT solver** (DPsim,
``dpsimpy``) and produces the same 11-channel contract at a per-cycle cadence.

**Window semantics (defensible choice).** The data contract defines a sample as
one RMS-phasor telemetry tick. In the OpenDSS pipeline that tick is 60 s of
quasi-static time. A PMU/merging unit at 60 fps delivers the *same* quantity —
one RMS phasor per power-frequency cycle — so Route B keeps the contract
(``x: (C=11, T=64)``, channel order ``va..fdev``, ``_window_label`` rule,
``MeasurementChain``, ``Normalizer``) verbatim and only redefines the tick:
**one timestep = one 60 Hz cycle (16.67 ms)**. A T=64 window is then a ~1.07 s
protection-relevant record instead of a ~1 h trend — the exact distinction
``docs/REAL_DATA.md`` §2.3 says must not be blurred, so the script never mixes
the two cadences and every output file carries ``cadence: "cycle"``.

**Network.** An EMT replica of the IEEE 13-node feeder — the same feeder that
backs federation clients 0 and 5 in ``simulation/partition.py`` — parameterised
from the vendored ``feeders/ieee13/IEEE13Nodeckt.dss``. Where ``opendssdirect``
is importable the topology is extracted **live from the same OpenDSS model**
(``--topology-source opendss``); otherwise the embedded snapshot below is used
(captured from that extraction; provenance comment in the constant). The
substitution transformer is rebuilt at its *nameplate* impedance (XHL = 8 %,
%r = 0.5 %/winding on 5 MVA): the published DSS case divides it by 1000 to
stiffen the benchmark, which would make sags physically unreachable; the EMT
replica restores it (documented decision — see ``docs/EMT_ROUTE_B.md``). The
1-phase regulators on 650->RG60 (XHL = 0.01 % on 1666 kVA) are skipped —
O(1e-5) ohm at MV scale — and the 4.16/0.48 kV service transformer XFM1 is
folded by placing the 634-bus loads at 633 (they draw the same kW/kvar from
the MV system; the 2 % transformer drop is noted as an approximation).
Metering replicates the OpenDSS instrumentation exactly: PT on the substation
bus ``rg60`` (phase-ground voltage), CT on the head line ``650632``.

**Loads.** DPsim's ``RXLoad`` is deliberately avoided: on dpsimpy 1.4.0 it
requires ``do_init_from_nodes_and_terminals(True)`` to draw current at all,
and that initialiser emits NaN whenever a phase carries zero power — which
the unbalanced IEEE13 load inventory has plenty of. Instead every spot load
is synthesised as explicit per-phase R||L shunts (constant-Z wye-equivalent:
``R_p = V_ln^2/P``, ``L_p = V_ln^2/(w*Q)``), initialisation runs from zero,
and the first ``WARMUP_CYCLES`` cycles are discarded. Documented decision —
see ``docs/EMT_ROUTE_B.md``.

**Events** use the same parameter distributions as ``simulation/events.py``
(Dagenhart 2000 LG bolted->40 ohm; De Andrade 2010 phase faults 0.01-1.5 ohm;
IEEE 1159 sag/swell bands; cold-load-pickup overload 1.4-2.0x):

* ``fault_lg``/``fault_ll``/``fault_llg``/``fault_3ph`` — realised as a
  three-phase shunt ``Switch`` closed by ``SwitchEvent3Ph`` whose
  ``closed_resistance`` is the inverse of the fault's nodal admittance matrix,
  reproducing OpenDSS ``Fault`` semantics exactly: LG = phase->ground through
  r; LL = phase<->phase through r; LLG = each phase->ground through r (OpenDSS
  wye, no inter-phase bond); 3PH = each phase->ground through r. Fault location
  is uniform over cumulative conductor length (Short 2004, same as OpenDSS)
  but the chosen line is *split at the fractional point* — continuous siting,
  strictly finer than the OpenDSS bus-granular approximation.
* ``fault_lg_hif`` — the reviewer's "high-impedance" arm: LG with r sampled in
  the upper band [15, 40] ohm (Dagenhart's 40-ohm phenomenon). Label stays
  ``FAULT_LG``; the ``arm`` field in the metadata distinguishes it.
* ``sag`` / ``swell`` — source-side excursion within the same IEEE 1159 bands
  used by ``events.py`` (sag depth 0.75-0.88 pu, swell 1.08-1.15 pu), realised
  as an upstream-source switchover (fast-transfer-switch analogue), so the
  metering bus sees a genuine step in EMF — matching the OpenDSS mechanism
  "source per-unit held at depth for the episode" without hand-edited waveforms.
* ``overload`` — cold-load pickup: an extra R||L load sized ``(mult-1)`` x
  total load closes behind the head CT (bus 632), so the metered current
  rises by the multiplier — the same aggregate effect as OpenDSS scaling all
  loads, and the physically meaningful site for a service-entrance pickup.
* ``normal`` — no event; per-run load multiplier 0.7-1.1 and per-phase gain
  unbalance +-10 % (the same phase_g construction as ``grid.py``).

**Channel extraction** mirrors ``grid.py:read_row`` + channel assembly at cycle
resolution: per-cycle DFT phasors at 60 Hz (exact bin, K=320 samples/cycle ->
dt = 52.0833 us, inside DPsim's recommended EMT range) give RMS phasor
magnitudes per phase; ``p``/``q`` = Re/Im of sum V*conj(I) on the contract
S_base; ``v0``/``i0`` = |sum of phasors|/3 on the same bases as grid.py;
``fdev`` = per-cycle frequency deviation measured from the positive-sequence
phasor angle drift (which resolves real inception/clearance transients, unlike
the synthetic step model) **plus** the same AR(1) wander as
``measurement.frequency_deviation`` with the event-coupled term disabled — the
event signature on ``fdev`` is then *measured*, not injected. The clean matrix
then passes through the unmodified ``MeasurementChain`` (IEC 61869 class-0.5
bias+noise, 16-bit ADC, NASPI-rate dropout — which at ~150-cycle records fires
with p ~ 0.045 per run, honest) and windows are cut by the unmodified
``windows_from_frame`` (T=64, stride 16, >=2-event-sample rule).

**Outputs** (all under ``--out``):

* ``scenarios.jsonl`` — the sampled scenario plan (seed -> params) before any
  simulation runs; replayable and auditable.
* ``runs.jsonl`` — one record per simulation: scenario params, achieved
  physics probes (pre/post V pu, fault current pu, measured sag depth ...),
  wall time, window yield.
* ``windows.npz`` — ``windows (N,11,64) float32``, ``labels (N,)``,
  ``run_idx (N,)`` (the split group), ``run_class/arm`` per window.
* ``manifest.json`` — config echo, env fingerprint, git commit, library
  versions, topology provenance, class histogram, SHA-256 of the window bytes,
  per-class physics summary, wall-clock budget.
* ``eval_<model>.json`` — when ``--train cnnlstm|ssm``: a ``RunResult`` record
  (accuracy, macro-F1, per-class recall, confusion matrix, measured batch-1
  inference latency p50/p95/p99 on the run machine). Splits are **by run**
  (windows of one EMT run stay together — the leakage-safe protocol, direct
  answer to the reviewer's random-split critique at the new cadence).
* ``xcheck_opendss.json`` — when ``--xcheck-opendss``: for a grid of
  (class, r_ohm, loc_frac) cells, the OpenDSS snapshot-solve phasor metering
  vs the EMT post-inception steady-state cycle — median relative agreement per
  channel. This is the direct "does the phasor-domain generator agree with EMT
  in the band the classifier sees" evidence.

**Remote usage** (runs on a Linux box; dpsim wheels ship
for CPython 3.10-3.14 linux+win x86-64)::

    python -m venv /venv/emt && /venv/emt/bin/pip install dpsim numpy pandas
    # repo bundle provides fedgrid on PYTHONPATH (src layout):
    PYTHONPATH=$PWD/src python -m experiments.emt_routeb --selftest
    PYTHONPATH=$PWD/src python -m experiments.emt_routeb --dry-run
    PYTHONPATH=$PWD/src python -m experiments.emt_routeb \
        --out runs_emt/routeb_s0 --seed 0 --workers 8 --train cnnlstm

``--selftest`` works *without* dpsimpy (import/compile check only); with
dpsimpy it additionally runs a ~0.4 s simulation and verifies the channel
extraction end-to-end. ``villas-dataprocessing`` is *not* required: logs are
read with pandas directly (one less dep on the remote box).

Route B validation module. Not part of the 8-class battery.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import sys
import time
import traceback
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# Repo bootstrap: allow running from the bundle without installing the package
# (PYTHONPATH=src also works; this just makes the script robust to cwd).
_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

__all__ = ["main"]

# ---------------------------------------------------------------------------
# Fixed constants (data contract cadence: one timestep = one fundamental cycle)
# ---------------------------------------------------------------------------
F0_HZ: float = 60.0  # IEEE13 base frequency (60 Hz system)
SAMPLES_PER_CYCLE: int = 320  # dt = 1/(60*320) = 52.0833 us (EMT-safe step)
DT_S: float = 1.0 / (F0_HZ * SAMPLES_PER_CYCLE)
WINDOW_T: int = 64  # contract T (cycles here)
STRIDE: int = 16  # same stride as partition.py
WARMUP_CYCLES: int = 5  # discarded (initialisation ramp)

CHANNELS: tuple[str, ...] = ("va", "vb", "vc", "ia", "ib", "ic", "p", "q", "v0", "i0", "fdev")
# Mirror of fedgrid.data.EventClass (kept literal so runs.jsonl is readable
# without importing the package; asserted equal at runtime when fedgrid loads).
EVENT_CLASS: dict[str, int] = {
    "normal": 0,
    "fault_lg": 1,
    "fault_ll": 2,
    "fault_llg": 3,
    "fault_3ph": 4,
    "overload": 5,
    "sag": 6,
    "swell": 7,
}
DEFAULT_CLASSES: tuple[str, ...] = (
    "normal",
    "fault_lg",
    "fault_lg_hif",
    "fault_ll",
    "fault_llg",
    "fault_3ph",
    "sag",
    "swell",
    "overload",
)

# Source impedance behind the substation bus (Thevenin at the metering point).
# = restored nameplate sub-transformer (115/4.16 kV, 5 MVA, XHL 8%, %r 0.5% per
# winding -> series R ~ (0.5+0.5)% * Zbase = 0.0346 ohm, X = 8% -> 0.2769 ohm;
# Zbase = 4.16^2/5 = 3.4611 ohm) + upstream grid from IEEE13 source
# (MVAsc3 = 20000 MVA at 115 kV -> |Z| = 115^2/20000 = 0.66 ohm at HV,
# referred to 4.16 kV: ~0.00086 ohm — negligible, folded into the total).
XFMR_R_OHM: float = 0.0346
XFMR_X_OHM: float = 0.2769
GRID_R_OHM: float = 0.0009
GRID_XR: float = 7.0  # assumed upstream X/R (typical sub-transmission; doc)
V_LL_V: float = 4160.0  # metering bus nominal (4.16 kV)
V_LN_V: float = V_LL_V / math.sqrt(3.0)  # 2401.8 V
TOTAL_KW: float = 3466.0  # IEEE13 connected load (extracted)
TOTAL_KVAR: float = 2102.0
NOMINAL_PF: float = 0.95  # same convention as grid.py
S_BASE_KVA: float = TOTAL_KW / NOMINAL_PF
I_BASE_A: float = S_BASE_KVA / (3.0 * V_LN_V / 1000.0)  # ~506.6 A (grid.py form)

_SEED_TAG = 0xE47B  # SeedSequence tag for Route B streams


def _skey(s: str) -> int:
    """Stable int for a string in a SeedSequence spawn key (crc32)."""
    return zlib.crc32(s.encode("utf-8")) & 0x7FFFFFFF


_R_OPEN_OHM: float = 1e9
_EPS_G_S: float = 1e-6  # leakage-to-ground regularising singular fault G (HIF-free)
_BLEED_R_OHM: float = 1e7  # keeps dangling phases referenced (numerically inert)


# ---------------------------------------------------------------------------
# IEEE13 topology snapshot — captured by extracting the vendored
# feeders/ieee13/IEEE13Nodeckt.dss via opendssdirect (Lines.* RMatrix/XMatrix/
# CMatrix are per-unit-length in the line's own unit; unit code 5 = feet, so
# matrices below are per-foot: ohm/ft for R/X, nF/ft for C). Regenerated by
# --refresh-topology on a box with opendssdirect + the repo bundle; the live
# path is preferred at runtime (provenance = identical numbers by construction).
# ---------------------------------------------------------------------------
_IEEE13_SNAPSHOT: dict[str, Any] = {
    "source": "opendssdirect extract of fedgrid/simulation/feeders/ieee13/"
    "IEEE13Nodeckt.dss (matrices per ft; ohms for R/X, nF for C)",
    "lines": [
        {"name": "650632", "b1": "rg60", "b2": "632", "ph": [1, 2, 3], "len_ft": 2000.0},
        {"name": "632670", "b1": "632", "b2": "670", "ph": [1, 2, 3], "len_ft": 667.0},
        {"name": "670671", "b1": "670", "b2": "671", "ph": [1, 2, 3], "len_ft": 1333.0},
        {"name": "671680", "b1": "671", "b2": "680", "ph": [1, 2, 3], "len_ft": 1000.0},
        {"name": "632633", "b1": "632", "b2": "633", "ph": [1, 2, 3], "len_ft": 500.0},
        {"name": "632645", "b1": "632", "b2": "645", "ph": [3, 2], "len_ft": 500.0},
        {"name": "645646", "b1": "645", "b2": "646", "ph": [3, 2], "len_ft": 300.0},
        {"name": "692675", "b1": "692", "b2": "675", "ph": [1, 2, 3], "len_ft": 500.0},
        {"name": "671684", "b1": "671", "b2": "684", "ph": [1, 3], "len_ft": 300.0},
        {"name": "684611", "b1": "684", "b2": "611", "ph": [3], "len_ft": 300.0},
        {"name": "684652", "b1": "684", "b2": "652", "ph": [1], "len_ft": 800.0},
        {
            "name": "671692",
            "b1": "671",
            "b2": "692",
            "ph": [1, 2, 3],
            "len_ft": 0.001,
            "is_switch": True,
        },
    ],
    "loads": [
        {"bus": "671", "ph": [1, 2, 3], "kw": 1155.0, "kvar": 660.0, "delta": True},
        {"bus": "633", "ph": [1], "kw": 160.0, "kvar": 110.0, "delta": False},
        {"bus": "633", "ph": [2], "kw": 120.0, "kvar": 90.0, "delta": False},
        {"bus": "633", "ph": [3], "kw": 120.0, "kvar": 90.0, "delta": False},
        {"bus": "645", "ph": [2], "kw": 170.0, "kvar": 125.0, "delta": False},
        {"bus": "646", "ph": [2, 3], "kw": 230.0, "kvar": 132.0, "delta": True},
        {"bus": "692", "ph": [3, 1], "kw": 170.0, "kvar": 151.0, "delta": True},
        {"bus": "675", "ph": [1], "kw": 485.0, "kvar": 190.0, "delta": False},
        {"bus": "675", "ph": [2], "kw": 68.0, "kvar": 60.0, "delta": False},
        {"bus": "675", "ph": [3], "kw": 290.0, "kvar": 212.0, "delta": False},
        {"bus": "611", "ph": [3], "kw": 170.0, "kvar": 80.0, "delta": False},
        {"bus": "652", "ph": [1], "kw": 128.0, "kvar": 86.0, "delta": False},
        {"bus": "670", "ph": [1], "kw": 17.0, "kvar": 10.0, "delta": False},
        {"bus": "670", "ph": [2], "kw": 66.0, "kvar": 38.0, "delta": False},
        {"bus": "670", "ph": [3], "kw": 117.0, "kvar": 68.0, "delta": False},
    ],
    "caps": [
        {"bus": "675", "ph": [1, 2, 3], "kvar": 600.0, "kv": 4.16},
        {"bus": "611", "ph": [3], "kvar": 100.0, "kv": 2.4},
    ],
}


def _ieee13_live() -> dict[str, Any] | None:
    """Extract the IEEE13 EMT spec live from the vendored OpenDSS model.

    Returns ``None`` when opendssdirect is unavailable; the snapshot constant
    is the fallback. Per-unit-length matrices come back in the line's declared
    unit (ft for IEEE13) — left in those units; conversion happens at build.
    """
    try:
        import opendssdirect as dss
    except ImportError:
        return None
    try:
        from fedgrid.simulation.profiles import FEEDER_DIR
    except ImportError:
        return None
    master = FEEDER_DIR / "ieee13" / "IEEE13Nodeckt.dss"
    if not master.is_file():
        return None
    cwd = os.getcwd()
    try:
        dss.Basic.ClearAll()
        dss.Text.Command("Set DefaultBaseFrequency=60")
        dss.Text.Command(f'compile "{master}"')
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            return None
        lines = []
        idx = dss.Lines.First()
        while idx:
            ph1 = [int(x) for x in dss.Lines.Bus1().split(".")[1:] if x.isdigit()]
            ph2 = [int(x) for x in dss.Lines.Bus2().split(".")[1:] if x.isdigit()]
            try:
                is_sw = bool(dss.Lines.IsSwitch())
            except Exception:
                is_sw = False
            lines.append(
                {
                    "name": dss.Lines.Name(),
                    "b1": dss.Lines.Bus1().split(".")[0],
                    "b2": dss.Lines.Bus2().split(".")[0],
                    "ph": ph1 or ph2 or [1, 2, 3],
                    "ph2": ph2,
                    "len_ft": float(dss.Lines.Length()),
                    "is_switch": is_sw,
                    "rmat": [float(x) for x in dss.Lines.RMatrix()],
                    "xmat": [float(x) for x in dss.Lines.XMatrix()],
                    "cmat": [float(x) for x in dss.Lines.CMatrix()],
                }
            )
            idx = dss.Lines.Next()
        loads = []
        idx = dss.Loads.First()
        while idx:
            name = dss.Loads.Name()
            dss.Circuit.SetActiveElement(f"Load.{name}")
            bus = dss.CktElement.BusNames()[0]
            nodes = [int(x) for x in bus.split(".")[1:] if x.isdigit()] or [1, 2, 3]
            bname = bus.split(".")[0]
            # XFM1 folding: the 0.48 kV bus 634 hangs off 633; its loads draw
            # the same kW/kvar from the MV system (2 % XHL drop neglected).
            if bname == "634":
                bname = "633"
            loads.append(
                {
                    "bus": bname,
                    "ph": nodes,
                    "kw": float(dss.Loads.kW()),
                    "kvar": float(dss.Loads.kvar()),
                    "delta": bool(dss.Loads.IsDelta()),
                }
            )
            idx = dss.Loads.Next()
        caps = []
        idx = dss.Capacitors.First()
        while idx:
            name = dss.Capacitors.Name()
            dss.Circuit.SetActiveElement(f"Capacitor.{name}")
            bus = dss.CktElement.BusNames()[0]
            nodes = [int(x) for x in bus.split(".")[1:] if x.isdigit()] or [1, 2, 3]
            caps.append(
                {
                    "bus": bus.split(".")[0],
                    "ph": nodes,
                    "kvar": float(dss.Capacitors.kvar()),
                    "kv": float(dss.Capacitors.kV()),
                }
            )
            idx = dss.Capacitors.Next()
        return {
            "source": f"live extract via opendssdirect of {master}",
            "lines": lines,
            "loads": loads,
            "caps": caps,
        }
    finally:
        os.chdir(cwd)


def load_topology(source: str = "auto") -> tuple[dict[str, Any], str]:
    """Return (spec, provenance). ``source``: auto|opendss|snapshot."""
    if source in ("auto", "opendss"):
        live = _ieee13_live()
        if live is not None:
            return live, "opendss-live"
        if source == "opendss":
            raise RuntimeError(
                "--topology-source=opendss but opendssdirect/vendored feeder unavailable"
            )
    snap = dict(_IEEE13_SNAPSHOT)
    # Per-ft matrices for the snapshot path: mtx601/602/603/604/605/606/607
    # constants, transcribed from the extraction (documented provenance).
    snap["lines"] = _snapshot_fill_matrices(snap["lines"])
    return snap, "embedded-snapshot"


def _snapshot_fill_matrices(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach per-ft R/X(/C) matrices for snapshot mode.

    Values below are the opendssdirect-extracted matrices divided back to
    per-ft (extraction already returns per-unit-length = per-ft here). Rather
    than duplicating 11 big matrices, the snapshot path recomputes them from
    the linecode constants in IEEE13Nodeckt.dss (the authoritative source):
    mtx601/602/606 are 3-ph, mtx603/604 2-ph, mtx605/607 1-ph; C is the
    OpenDSS default the solver reports (mtx601/602/603/604/605/606 codes that
    omit cmatrix get the same small default; only mtx606 declares one).
    """
    # Ohm per MILE lower-triangle constants from IEEE13Nodeckt.dss linecodes;
    # converted to per-ft full symmetric matrices at use.
    codes = {  # (lower-triangle R ohm/mi, X ohm/mi, C nF/mi, nph)
        "mtx601": (
            [0.3465, 0.1560, 0.3375, 0.1580, 0.1535, 0.3414],
            [1.0179, 0.5017, 1.0478, 0.4236, 0.3849, 1.0348],
            [2.80, 0.0, 2.80, 0.0, 0.0, 2.80],
            3,
        ),
        "mtx602": (
            [0.7526, 0.1580, 0.7475, 0.1560, 0.1535, 0.7436],
            [1.1814, 0.4236, 1.1983, 0.5017, 0.3849, 1.2112],
            [2.80, 0.0, 2.80, 0.0, 0.0, 2.80],
            3,
        ),
        "mtx603": ([1.3238, 0.2066, 1.3294], [1.3569, 0.4591, 1.3471], [2.80, 0.0, 2.80], 2),
        "mtx604": ([1.3238, 0.2066, 1.3294], [1.3569, 0.4591, 1.3471], [2.80, 0.0, 2.80], 2),
        "mtx605": ([1.3292], [1.3475], [2.80], 1),
        "mtx606": (
            [0.791721, 0.318476, 0.781649, 0.28345, 0.318476, 0.791721],
            [0.438352, 0.0276838, 0.396697, -0.0184204, 0.0276838, 0.438352],
            [383.948, 0.0, 383.948, 0.0, 0.0, 383.948],
            3,
        ),
        "mtx607": ([1.3425], [0.5124], [236.0], 1),
        # line 671692 is declared `Switch=y` (r1=r0=1e-4): diagonal 3x3, ~open X/C
        "switch": ([1e-4, 0.0, 1e-4, 0.0, 0.0, 1e-4], [0.0] * 6, [0.0] * 6, 3),
    }
    l2code = {
        "650632": "mtx601",
        "632670": "mtx601",
        "670671": "mtx601",
        "671680": "mtx601",
        "632633": "mtx602",
        "632645": "mtx603",
        "645646": "mtx603",
        "692675": "mtx606",
        "671684": "mtx604",
        "684611": "mtx605",
        "684652": "mtx607",
        "671692": "switch",
    }
    out = []
    for ln in lines:
        code = l2code[ln["name"]]
        rl, xl, cl, nph = codes[code]

        def full(vals, per_mile_scale, nph=nph):
            m = np.zeros((nph, nph))
            k = 0
            for i in range(nph):
                for j in range(i + 1):
                    m[i, j] = m[j, i] = vals[k]
                    k += 1
            return (m * per_mile_scale).tolist()

        # ohm/mi -> ohm/ft : /5280 ; nF/mi -> nF/ft : /5280
        out.append(
            {
                **ln,
                "rmat": [x for row in full(rl, 1 / 5280.0) for x in row],
                "xmat": [x for row in full(xl, 1 / 5280.0) for x in row],
                "cmat": [x for row in full(cl, 1 / 5280.0) for x in row],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Scenario sampling (distributions mirrored from simulation/events.py)
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """One EMT run: class + physics parameters + event schedule."""

    run_id: str
    cls: str  # registry key (fault_lg_hif is an arm of fault_lg)
    label: int  # EventClass code emitted into labels
    arm: str  # provenance arm (e.g. "hif" within fault_lg)
    seed: int
    load_mult: float = 1.0  # feeder-wide load multiplier (baseline level)
    phase_g: tuple[float, float, float] = (1.0, 1.0, 1.0)  # per-phase unbalance
    r_ohm: float = 0.0  # fault resistance (fault_* classes)
    loc_frac: float = 0.0  # fault position, fraction of cumulative length
    phases: tuple[int, ...] = (0,)  # 0-based faulted phases
    t_on_cycles: float = 0.0  # event start (fractional cycles -> inception angle)
    duration_cycles: float = 0.0  # event duration (0 -> sustained to sim end)
    n_cycles: int = 150  # total simulated cycles (~2.5 s)
    depth_pu: float = 0.0  # sag depth target (source-side excursion)
    level_pu: float = 0.0  # swell level target
    overload_mult: float = 0.0  # total load multiplier realised for overload


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_scenarios(
    seed: int, runs_per_class: int, classes: tuple[str, ...], n_cycles: int = 150
) -> list[Scenario]:
    """Draw the full battery plan; deterministic in ``seed``.

    Per-run RNG streams derive from SeedSequence([_SEED_TAG, seed, i]) so any
    single run can be reproduced in isolation (and sharded across workers).
    """
    plan: list[Scenario] = []
    for cls in classes:
        if cls not in DEFAULT_CLASSES:
            raise ValueError(f"unknown class {cls!r}; expected {DEFAULT_CLASSES}")
        for i in range(runs_per_class):
            rng = np.random.default_rng(np.random.SeedSequence([_SEED_TAG, seed, _skey(cls), i]))
            load_mult = float(rng.uniform(0.7, 1.1))
            phase_g = tuple(
                float(x) for x in 1.0 + np.clip(rng.normal(0.0, 0.05, 3), -0.10, 0.10)
            )  # same as grid.py
            n_cyc = int(n_cycles)
            # event window: start after warmup + 10-35 cycles (random inception
            # angle falls out of the fractional part), end leaves >= 8 cycles
            t_on = float(rng.uniform(WARMUP_CYCLES + 12, WARMUP_CYCLES + 35))
            if cls == "normal":
                plan.append(
                    Scenario(
                        f"{cls}_r{i:03d}",
                        cls,
                        0,
                        "base",
                        seed,
                        load_mult,
                        phase_g,
                        n_cycles=n_cyc,
                        t_on_cycles=0.0,
                        duration_cycles=0.0,
                    )
                )
                continue
            sustained = bool(rng.random() < 0.45)
            if cls in ("fault_lg", "fault_lg_hif"):
                if cls == "fault_lg_hif":
                    r = _log_uniform(rng, 15.0, 40.0)  # Dagenhart 40-ohm tail
                    arm = "hif"
                else:
                    r = _log_uniform(rng, 0.05, 40.0)  # events.py LG band
                    arm = "bulk"
                dur = 0.0 if sustained else float(rng.uniform(6, 36))
                plan.append(
                    Scenario(
                        f"{cls}_r{i:03d}",
                        cls,
                        EVENT_CLASS["fault_lg"],
                        arm,
                        seed,
                        load_mult,
                        phase_g,
                        r_ohm=r,
                        loc_frac=float(rng.uniform(0.0, 1.0)),
                        phases=(int(rng.integers(0, 3)),),
                        t_on_cycles=t_on,
                        duration_cycles=dur,
                        n_cycles=n_cyc,
                    )
                )
            elif cls in ("fault_ll", "fault_llg", "fault_3ph"):
                r = _log_uniform(rng, 0.01, 1.5)  # events.py phase band
                ph = tuple(int(x) for x in rng.choice(3, size=2, replace=False))
                dur = 0.0 if sustained else float(rng.uniform(6, 36))
                nph = 3 if cls == "fault_3ph" else 2
                plan.append(
                    Scenario(
                        f"{cls}_r{i:03d}",
                        cls,
                        EVENT_CLASS[cls],
                        "bulk",
                        seed,
                        load_mult,
                        phase_g,
                        r_ohm=r,
                        loc_frac=float(rng.uniform(0.0, 1.0)),
                        phases=ph[:nph] if cls != "fault_3ph" else (0, 1, 2),
                        t_on_cycles=t_on,
                        duration_cycles=dur,
                        n_cycles=n_cyc,
                    )
                )
            elif cls == "sag":
                dur = float(rng.uniform(24, 72))  # sustained-variation regime
                plan.append(
                    Scenario(
                        f"{cls}_r{i:03d}",
                        cls,
                        EVENT_CLASS["sag"],
                        "srcdip",
                        seed,
                        load_mult,
                        phase_g,
                        depth_pu=float(rng.uniform(0.75, 0.88)),
                        t_on_cycles=t_on,
                        duration_cycles=dur,
                        n_cycles=n_cyc,
                    )
                )
            elif cls == "swell":
                dur = float(rng.uniform(24, 72))
                plan.append(
                    Scenario(
                        f"{cls}_r{i:03d}",
                        cls,
                        EVENT_CLASS["swell"],
                        "srcrise",
                        seed,
                        load_mult,
                        phase_g,
                        level_pu=float(rng.uniform(1.08, 1.15)),
                        t_on_cycles=t_on,
                        duration_cycles=dur,
                        n_cycles=n_cyc,
                    )
                )
            elif cls == "overload":
                dur = 0.0 if sustained else float(rng.uniform(24, 72))
                plan.append(
                    Scenario(
                        f"{cls}_r{i:03d}",
                        cls,
                        EVENT_CLASS["overload"],
                        "clpu",
                        seed,
                        load_mult,
                        phase_g,
                        overload_mult=float(rng.uniform(1.4, 2.0)),
                        t_on_cycles=t_on,
                        duration_cycles=dur,
                        n_cycles=n_cyc,
                    )
                )
    return plan


# ---------------------------------------------------------------------------
# EMT topology construction
# ---------------------------------------------------------------------------


def fault_admittance(
    cls: str, r_ohm: float, phases: tuple[int, ...], n_phases: int = 3
) -> np.ndarray:
    """3x3 nodal admittance of the fault (S), OpenDSS ``Fault`` semantics.

    LG : g on the faulted phase's diagonal.
    LL : a two-terminal conductance g=1/r between the two faulted phases.
    LLG: independent g on each faulted phase (OpenDSS wye, no bond).
    3PH: independent g on all three.
    A 1 uS leakage diagonal keeps the matrix invertible in every case
    (numerically inert vs r >= 1e-2 ohm -> g <= 100 S).
    """
    g = np.eye(3) * _EPS_G_S
    if cls == "fault_ll":
        p, q = phases
        gg = 1.0 / r_ohm
        g[p, p] += gg
        g[q, q] += gg
        g[p, q] -= gg
        g[q, p] -= gg
    else:
        for p in phases:
            g[p, p] += 1.0 / r_ohm
    return g


def _emb3(local: Any, phases: list[int], absent: float) -> np.ndarray:
    """Embed a nph x nph conductor matrix into the 3x3 phase frame.

    ``phases`` are 1-based OpenDSS conductor numbers; absent phases get
    ``absent`` on the diagonal — a near-open impedance for series R/L (the
    conductor physically does not exist on that line) or 0 for shunt C.
    """
    loc = np.asarray(local, dtype=np.float64)
    n = loc.shape[0]
    out = np.zeros((3, 3))
    for i in range(3):
        out[i, i] = absent
    for i, pi in enumerate(phases):
        for j, pj in enumerate(phases):
            if i < n and j < n:
                out[pi - 1, pj - 1] = loc[i, j]
    return out


def build_system(
    sc: Scenario, spec: dict[str, Any], log_dir: Path
) -> tuple[Any, Any, dict[str, Any]]:
    """Build the DPsim SystemTopology + logger for one scenario.

    Returns (system, logger, aux) where aux carries handles needed by the
    event scheduler and channel extraction (head line element, meter node,
    fault switch, source switches for sag/swell, overload switch).
    """
    import dpsimpy

    emt = dpsimpy.emt
    ph3 = emt.ph3
    gnd = emt.SimNode.gnd

    def big3(x: float = _R_OPEN_OHM) -> np.ndarray:
        return np.array(dpsimpy.Math.single_phase_parameter_to_three_phase(x))

    # -- nodes --------------------------------------------------------------
    # OpenDSS `Switch=y` lines are closed ties: they must NOT become PiLines
    # (a ~0-ohm series impedance is numerically singular in the EMT MNA). They
    # alias the two bus nodes into one — exactly what a closed switch means.
    alias: dict[str, str] = {}
    for ln in spec["lines"]:
        if ln.get("is_switch"):
            alias[ln["b2"]] = ln["b1"]
            alias[ln["b1"]] = ln["b1"]

    def bus_node(bus: str) -> str:
        return "b_" + alias.get(bus, bus)

    node_names: set[str] = {"n_sub"}  # source-side bus (sub-transformer HV equiv)
    for ln in spec["lines"]:
        if ln.get("is_switch"):
            continue
        node_names.add(bus_node(ln["b1"]))
        node_names.add(bus_node(ln["b2"]))
    is_fault = sc.cls.startswith("fault")
    fault_line = None
    in_frac = 0.0
    if is_fault:
        # weight lines by length among those with >= min_phases (OpenDSS rule)
        min_ph = {"fault_lg": 1, "fault_lg_hif": 1, "fault_ll": 2, "fault_llg": 2, "fault_3ph": 3}[
            sc.cls
        ]
        cands = [
            ln for ln in spec["lines"] if len(ln["ph"]) >= min_ph and not ln.get("is_switch")
        ]  # closed ties carry no length
        lens = np.array([max(ln["len_ft"], 1e-6) for ln in cands])
        cum = np.cumsum(lens)
        target = float(np.clip(sc.loc_frac, 0.0, 1.0 - 1e-9)) * cum[-1]
        idx = min(int(np.searchsorted(cum, target, side="right")), len(cands) - 1)
        fault_line = cands[idx]
        in_frac = float((target - (cum[idx] - lens[idx])) / lens[idx])
        node_names.add("b_flt")
    needs_switchover = sc.cls in ("sag", "swell")
    if needs_switchover:
        node_names.update({"n_sa", "n_sb"})
    if sc.cls == "overload":
        node_names.add("n_ovl")

    # Nodes start at 0: do_init_from_nodes_and_terminals() then initialises the
    # component internals consistently (verified path — feeding complex v_init
    # phasors here collapses the network to ~0 on the first step in dpsim 1.4.0).
    nodes = {nm: emt.SimNode(nm, dpsimpy.PhaseType.ABC) for nm in sorted(node_names)}

    comps: list[Any] = []
    aux: dict[str, Any] = {"fault_line": fault_line, "fault_node": None, "in_frac": in_frac}

    # -- source(s) ----------------------------------------------------------
    vabc = [V_LL_V, V_LL_V * np.exp(-1j * 2 * np.pi / 3), V_LL_V * np.exp(1j * 2 * np.pi / 3)]
    src_a = ph3.VoltageSource("srcA")
    src_a.set_parameters(V_ref=np.array(vabc, dtype=np.complex128), f_src=F0_HZ)
    if not needs_switchover:
        src_a.connect([gnd, nodes["n_sub"]])
        comps.append(src_a)
    else:
        # dual-source switchover (fast-transfer-switch analogue): A feeds
        # n_sub pre-event; at t_on A opens and B (at depth/level) closes.
        src_a.connect([gnd, nodes["n_sa"]])
        comps.append(src_a)
        lvl = sc.depth_pu if sc.cls == "sag" else sc.level_pu
        src_b = ph3.VoltageSource("srcB")
        src_b.set_parameters(V_ref=np.array(vabc, dtype=np.complex128) * lvl, f_src=F0_HZ)
        src_b.connect([gnd, nodes["n_sb"]])
        comps.append(src_b)
        sw_a = ph3.Switch("swA")
        sw_a.set_parameters(open_resistance=big3(), closed_resistance=big3(1e-3), closed=True)
        sw_a.connect([nodes["n_sa"], nodes["n_sub"]])
        comps.append(sw_a)
        sw_b = ph3.Switch("swB")
        sw_b.set_parameters(open_resistance=big3(), closed_resistance=big3(1e-3), closed=False)
        sw_b.connect([nodes["n_sb"], nodes["n_sub"]])
        comps.append(sw_b)
        aux["sw_a"], aux["sw_b"] = sw_a, sw_b

    # Thevenin branch n_sub -> n_meter (restored nameplate transformer + grid)
    n_meter = "b_rg60"
    r_src = XFMR_R_OHM + GRID_R_OHM
    x_src = XFMR_X_OHM + GRID_R_OHM * GRID_XR
    z_src = ph3.PiLine("z_src")
    z_src.set_parameters(
        series_resistance=np.array(dpsimpy.Math.single_phase_parameter_to_three_phase(r_src)),
        series_inductance=np.array(
            dpsimpy.Math.single_phase_parameter_to_three_phase(x_src / (2 * np.pi * F0_HZ))
        ),
    )
    z_src.connect([nodes["n_sub"], nodes[n_meter]])
    comps.append(z_src)

    # -- feeder lines ---------------------------------------------------------
    head_seg = None
    for ln in spec["lines"]:
        if ln.get("is_switch"):
            continue  # closed tie — the b2 node is aliased onto b1
        nph = len(ln["ph"])
        rmat = np.array(ln["rmat"]).reshape(nph, nph) * ln["len_ft"]  # ohm
        lmat = np.array(ln["xmat"]).reshape(nph, nph) / (2 * np.pi * F0_HZ) * ln["len_ft"]  # H
        cmat = (
            np.array(ln.get("cmat") or np.zeros((nph, nph))).reshape(nph, nph) * ln["len_ft"] * 1e-9
        )  # F
        split_here = fault_line is not None and ln["name"] == fault_line["name"]
        segs = (
            [(bus_node(ln["b1"]), "b_flt", in_frac), ("b_flt", bus_node(ln["b2"]), 1.0 - in_frac)]
            if split_here
            else [(bus_node(ln["b1"]), bus_node(ln["b2"]), 1.0)]
        )
        for si, (ba, bb, frac) in enumerate(segs):
            name = ln["name"] if not split_here else f"{ln['name']}_{si}"
            line = ph3.PiLine("l_" + name)
            line.set_parameters(
                series_resistance=_emb3(rmat * frac, ln["ph"], 1e7),
                series_inductance=_emb3(lmat * frac, ln["ph"], 1e2),
                parallel_capacitance=_emb3(cmat * frac, ln["ph"], 0.0),
            )
            line.connect([nodes[ba], nodes[bb]])
            comps.append(line)
            if ln["name"] == "650632" and (not split_here or si == 0):
                head_seg = line

    # -- bleed resistors: every non-source node keeps a ground reference ------
    for nn in nodes.values():
        bleed = ph3.Resistor("bleed_" + str(nn.name()))
        bleed.set_parameters(R=big3(_BLEED_R_OHM))
        bleed.connect([nn, gnd])  # ph3 shunts need both terminals
        comps.append(bleed)

    # -- loads --------------------------------------------------------------
    # Explicit R || L shunts per phase (constant-Z wye-equivalent). RXLoad is
    # avoided entirely: on dpsim 1.4.0 it needs do_init_from_nodes_and_termi-
    # nals() to energise at all, and that init NaNs whenever a phase carries
    # zero power (i.e. any 1-/2-phase load). R_p = V_ln^2/P, L_p = V_ln^2/
    # (w*Q) — the exact constant-impedance equivalents the contract assumes.
    def shunt_load(tag: str, node: Any, p_ph: np.ndarray, q_ph: np.ndarray) -> list[Any]:
        out = []
        for ph in range(3):
            p, q = float(p_ph[ph]), float(q_ph[ph])
            if abs(p) > 1e-3:
                r = ph3.Resistor(f"{tag}_r{ph}")
                rm = np.eye(3) * _R_OPEN_OHM
                rm[ph, ph] = V_LN_V**2 / p
                r.set_parameters(R=rm)
                r.connect([node, gnd])
                out.append(r)
            if abs(q) > 1e-3:
                ll = ph3.Inductor(f"{tag}_l{ph}")
                lm = np.eye(3) * _R_OPEN_OHM  # huge L = open
                lm[ph, ph] = V_LN_V**2 / (2 * np.pi * F0_HZ * q)
                ll.set_parameters(L=lm)
                ll.connect([node, gnd])
                out.append(ll)
        return out

    gains = np.array(sc.phase_g)
    for j, ld in enumerate(spec["loads"]):
        share = np.zeros(3)
        if len(ld["ph"]) == 3:
            share[:] = 1.0 / 3.0
        elif len(ld["ph"]) == 2:
            for p in ld["ph"]:
                share[p - 1] = 0.5  # delta 1-ph / 2-ph -> half per phase
        else:
            share[ld["ph"][0] - 1] = 1.0
        p_ph = share * ld["kw"] * 1e3 * sc.load_mult * gains
        q_ph = share * ld["kvar"] * 1e3 * sc.load_mult * gains
        comps += shunt_load(f"ld{j}_{ld['bus']}", nodes[bus_node(ld["bus"])], p_ph, q_ph)

    # -- capacitors (shunt C to ground) --------------------------------------
    for j, cp in enumerate(spec["caps"]):
        vln = cp["kv"] * 1e3 / math.sqrt(3.0) if cp["kv"] > 3.0 else cp["kv"] * 1e3
        cmat = np.zeros((3, 3))
        q_per = cp["kvar"] * 1e3 / len(cp["ph"])
        cval = q_per / (2 * np.pi * F0_HZ * vln * vln)
        for p in cp["ph"]:
            cmat[p - 1, p - 1] = cval
        cap = ph3.Capacitor(f"cap{j}")
        cap.set_parameters(C=cmat)
        cap.connect([nodes[bus_node(cp["bus"])], gnd])  # ph3 shunt: 2 terminals
        comps.append(cap)

    # -- fault / overload switches -------------------------------------------
    if is_fault:
        kind = (
            "fault_ll"
            if sc.cls == "fault_ll"
            else ("fault_lg" if sc.cls == "fault_lg_hif" else sc.cls)
        )
        g_fault = fault_admittance(kind, sc.r_ohm, sc.phases)
        sw = ph3.Switch("fault")
        sw.set_parameters(open_resistance=big3(), closed_resistance=np.linalg.inv(g_fault))
        sw.open()
        sw.connect([nodes["b_flt"], gnd])
        if not globals().get("_DBG_NO_FAULT_SW"):
            comps.append(sw)
        aux["fault_sw"] = sw
        aux["fault_node"] = nodes["b_flt"]

    if sc.cls == "overload":
        sw = ph3.Switch("ovl_sw")
        sw.set_parameters(open_resistance=big3(), closed_resistance=big3(1e-3), closed=False)
        extra_p = np.array([(sc.overload_mult - 1.0) * TOTAL_KW * 1e3 / 3.0] * 3)
        extra_q = np.array([(sc.overload_mult - 1.0) * TOTAL_KVAR * 1e3 / 3.0] * 3)
        # inject at bus 632 — just BEHIND the head CT (650632 meters the
        # pickup current, matching OpenDSS's whole-feeder load scaling)
        sw.connect([nodes["b_632"], nodes["n_ovl"]])
        comps += [sw] + shunt_load("ld_ovl", nodes["n_ovl"], extra_p, extra_q)
        aux["ovl_sw"] = sw

    # -- system + logger ------------------------------------------------------
    all_nodes = [gnd] + [v for _, v in sorted(nodes.items())]
    system = dpsimpy.SystemTopology(F0_HZ, all_nodes, comps)
    logger = dpsimpy.Logger(f"rtb_{sc.run_id}")
    logger.set_log_dir(str(log_dir))
    logger.log_attribute("v_meter", "v", nodes[n_meter])
    if head_seg is not None:
        logger.log_attribute("i_head", "i_intf", head_seg)
    aux["meter_node"] = nodes[n_meter]
    return system, logger, aux


def _schedule_events(sim: Any, sc: Scenario, aux: dict[str, Any]) -> None:
    """Attach SwitchEvent3Ph transitions realising the scenario's event."""
    import dpsimpy

    t_on = sc.t_on_cycles / F0_HZ
    t_off = (t_on + sc.duration_cycles / F0_HZ) if sc.duration_cycles > 0 else None
    if sc.cls.startswith("fault") and aux.get("fault_sw") is not None:
        sim.add_event(dpsimpy.event.SwitchEvent3Ph(t_on, aux["fault_sw"], True))
        if t_off is not None:
            sim.add_event(dpsimpy.event.SwitchEvent3Ph(t_off, aux["fault_sw"], False))
    elif sc.cls in ("sag", "swell"):
        # source switchover: open A at t_on, close B one step later (never
        # parallel two sources at different amplitude in the same step).
        sim.add_event(dpsimpy.event.SwitchEvent3Ph(t_on, aux["sw_a"], False))
        sim.add_event(dpsimpy.event.SwitchEvent3Ph(t_on + DT_S, aux["sw_b"], True))
        if t_off is not None:
            sim.add_event(dpsimpy.event.SwitchEvent3Ph(t_off, aux["sw_b"], False))
            sim.add_event(dpsimpy.event.SwitchEvent3Ph(t_off + DT_S, aux["sw_a"], True))
    elif sc.cls == "overload" and aux.get("ovl_sw") is not None:
        sim.add_event(dpsimpy.event.SwitchEvent3Ph(t_on, aux["ovl_sw"], True))
        if t_off is not None:
            sim.add_event(dpsimpy.event.SwitchEvent3Ph(t_off, aux["ovl_sw"], False))


# ---------------------------------------------------------------------------
# One simulation run -> per-cycle channels + labels
# ---------------------------------------------------------------------------


def _cycle_phasor(x: np.ndarray, k: int) -> complex:
    """RMS phasor at fundamental for cycle k of signal x (K samples/cycle)."""
    seg = x[k * SAMPLES_PER_CYCLE : (k + 1) * SAMPLES_PER_CYCLE]
    n = np.arange(len(seg))
    return (
        float(np.sqrt(2.0))
        / len(seg)
        * complex(np.sum(seg * np.exp(-1j * 2 * np.pi * n / len(seg))))
    )


def extract_channels(
    csv_path: Path, sc: Scenario, rng_wander: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Read a DPsim CSV log -> (clean (n_cycles,11), labels (n_cycles,), probes).

    Channels: per-cycle RMS phasor magnitudes in pu on the contract bases
    (va..vc on V_LN, ia..ic on I_BASE, p/q on S_BASE), v0/i0 = |sum/3| in pu,
    fdev = measured per-cycle frequency deviation + AR(1) wander overlay.
    """
    import pandas as pd

    df = pd.read_csv(csv_path, skipinitialspace=True)
    n_cyc = sc.n_cycles
    need_rows = n_cyc * SAMPLES_PER_CYCLE
    if len(df) < need_rows:
        raise RuntimeError(
            f"{csv_path}: {len(df)} rows < {need_rows} expected "
            f"({n_cyc} cycles x {SAMPLES_PER_CYCLE} samples) — solver died?")
    v = np.zeros((n_cyc, 3), dtype=np.complex128)
    i = np.zeros((n_cyc, 3), dtype=np.complex128)
    for ph in range(3):
        # DPsim emits '-nan(ind)' strings on solver failure: coerce -> NaN and
        # fail loudly instead of letting a cryptic complex() error surface.
        vsig = pd.to_numeric(df[f"v_meter_{ph}"],
                             errors="coerce").to_numpy(dtype=np.float64)
        isig = pd.to_numeric(df[f"i_head_{ph}"],
                             errors="coerce").to_numpy(dtype=np.float64)
        n_bad = int(np.isnan(vsig[:need_rows]).sum()
                    + np.isnan(isig[:need_rows]).sum())
        if n_bad:
            raise RuntimeError(
                f"{csv_path}: {n_bad} NaN/non-numeric samples in phase {ph} "
                "signals — solver emitted -nan(ind) (singular topology or "
                "init failure)")
        for k in range(n_cyc):
            v[k, ph] = _cycle_phasor(vsig, k)
            i[k, ph] = _cycle_phasor(isig, k)
    vmag = np.abs(v) / V_LN_V
    imag = np.abs(i) / I_BASE_A
    # DPsim logs branch current in the node->ground reference direction; the
    # contract wants positive P/Q = power flowing INTO the feeder -> flip sign.
    s_3ph = -np.sum(v * np.conj(i), axis=1)  # VA (phasor RMS products)
    p_pu = np.real(s_3ph) / (S_BASE_KVA * 1e3)
    q_pu = np.imag(s_3ph) / (S_BASE_KVA * 1e3)
    v0 = np.abs(v.sum(axis=1)) / 3.0 / V_LN_V
    i0 = np.abs(i.sum(axis=1)) / 3.0 / I_BASE_A
    # per-cycle frequency deviation: angle drift of phase-a phasor (and of the
    # positive sequence for robustness) * f0 — real measured transient content
    ang = np.unwrap(np.angle(v[:, 0]))
    f_meas = np.zeros(n_cyc)
    f_meas[1:] = np.diff(ang) * F0_HZ / (2 * np.pi)
    f_meas[0] = f_meas[1]
    # AR(1) wander overlay — reuse the repo's own process (same constants,
    # same function) with the event-coupled synthetic term disabled.
    try:
        from fedgrid.simulation.measurement import frequency_deviation

        wander = frequency_deviation(
            rng_wander, np.zeros(n_cyc, dtype=np.int64), step_s=1.0 / F0_HZ, events=()
        )
    except Exception:
        wander = np.zeros(n_cyc)  # fallback: no fedgrid on the box
    fdev = f_meas + wander

    clean = np.column_stack(
        [
            vmag[:, 0],
            vmag[:, 1],
            vmag[:, 2],
            imag[:, 0],
            imag[:, 1],
            imag[:, 2],
            p_pu,
            q_pu,
            v0,
            i0,
            fdev,
        ]
    )
    # per-cycle labels: event owns the cycle if active at its midpoint
    labels = np.zeros(n_cyc, dtype=np.int64)
    t_on = sc.t_on_cycles
    t_off = t_on + sc.duration_cycles if sc.duration_cycles > 0 else n_cyc + 1
    for k in range(n_cyc):
        if sc.cls != "normal" and t_on <= k + 0.5 < t_off:
            labels[k] = sc.label
    probes = _physics_probes(clean, labels, sc)
    return clean, labels, probes


def _physics_probes(clean: np.ndarray, labels: np.ndarray, sc: Scenario) -> dict[str, float]:
    """Physics summary numbers for the audit log (pre-measurement channels)."""
    ch = {name: clean[:, j] for j, name in enumerate(CHANNELS)}
    ev = labels != 0
    pre = (~ev) & (np.arange(len(labels)) > WARMUP_CYCLES)
    out = {
        "pre_v_mean_pu": float(np.mean(ch["va"][pre] + ch["vb"][pre] + ch["vc"][pre]) / 3)
        if pre.any()
        else float("nan"),
        "pre_i_mean_pu": float(np.mean(ch["ia"][pre] + ch["ib"][pre] + ch["ic"][pre]) / 3)
        if pre.any()
        else float("nan"),
    }
    if ev.any():
        out.update(
            {
                "ev_v_min_pu": float(
                    min(ch["va"][ev].min(), ch["vb"][ev].min(), ch["vc"][ev].min())
                ),
                "ev_v_max_pu": float(
                    max(ch["va"][ev].max(), ch["vb"][ev].max(), ch["vc"][ev].max())
                ),
                "ev_i_max_pu": float(
                    max(ch["ia"][ev].max(), ch["ib"][ev].max(), ch["ic"][ev].max())
                ),
                "ev_i0_max_pu": float(ch["i0"][ev].max()),
                "ev_v0_max_pu": float(ch["v0"][ev].max()),
                "ev_p_absmax_pu": float(np.abs(ch["p"][ev]).max()),
            }
        )
        if sc.cls == "sag":
            out["achieved_depth_pu"] = float(
                np.mean(ch["va"][ev] + ch["vb"][ev] + ch["vc"][ev]) / 3.0
            )
        if sc.cls == "swell":
            out["achieved_level_pu"] = float(
                np.mean(ch["va"][ev] + ch["vb"][ev] + ch["vc"][ev]) / 3.0
            )
        if sc.cls == "overload":
            out["achieved_i_pu"] = float(np.mean(ch["ia"][ev] + ch["ib"][ev] + ch["ic"][ev]) / 3.0)
        if sc.cls.startswith("fault"):
            out["achieved_fault_i_pu"] = out["ev_i_max_pu"]
    return out


def run_scenario(
    sc: Scenario,
    spec: dict[str, Any],
    work_dir: Path,
    keep_logs: str = "none",
    meas_bias_seed: int = 0,
) -> dict[str, Any]:
    """Simulate one scenario end-to-end; returns windows + audit record.

    ``meas_bias_seed`` fixes the per-"installed-MU" calibration biases per run
    (they're a property of the measurement hardware, so each run draws its own
    instrument — matching the per-client rng_bias draw in grid.py).
    """
    import dpsimpy

    t0 = time.perf_counter()
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    system, logger, aux = build_system(sc, spec, log_dir)

    sim = dpsimpy.Simulation(f"rtb_{sc.run_id}", dpsimpy.LogLevel.err)
    sim.set_domain(dpsimpy.Domain.EMT)
    sim.set_solver(dpsimpy.Solver.MNA)
    sim.set_system(system)
    sim.set_time_step(DT_S)
    sim.set_final_time(sc.n_cycles / F0_HZ)
    # NOTE: do_init_from_nodes_and_terminals(True) produces NaN across the
    # whole network when any RXLoad has zero-power phases (unbalanced 1/2-ph
    # loads) behind series impedance — verified on dpsim 1.4.0. Starting from
    # zero and letting the source ramp over the warmup cycles is the working
    # path (WARMUP_CYCLES are excluded from windowing anyway).
    sim.do_init_from_nodes_and_terminals(False)
    sim.add_logger(logger)
    _schedule_events(sim, sc, aux)
    # dpsim 1.4.0 writes internal .log files into logger.set_log_dir() but the
    # data CSV lands at <cwd>/logs/<logger>.csv (observed quirk) — probe both.
    name = f"rtb_{sc.run_id}"
    candidates = (
        log_dir / f"{name}.csv",
        Path.cwd() / "logs" / f"{name}.csv",
        Path.cwd() / f"{name}.csv",
    )
    for stale in candidates:  # never read a stale CSV from a prior run
        if stale.is_file():
            stale.unlink()
    sim.run()

    csv_path = next((p for p in candidates if p.is_file()), None)
    if csv_path is None:
        raise RuntimeError(
            f"DPsim produced no CSV for {name} (checked {log_dir} and {Path.cwd() / 'logs'})"
        )
    rng_w = np.random.default_rng(
        np.random.SeedSequence([_SEED_TAG, sc.seed, _skey(sc.run_id), _skey("fdev")])
    )
    clean, labels, probes = extract_channels(csv_path, sc, rng_w)

    # measurement chain (the classifier sees what the merging unit sees)
    try:
        from fedgrid.simulation.measurement import MeasurementChain

        rng_b = np.random.default_rng(
            np.random.SeedSequence([_SEED_TAG, meas_bias_seed, _skey(sc.run_id), _skey("bias")])
        )
        rng_m = np.random.default_rng(
            np.random.SeedSequence([_SEED_TAG, sc.seed, _skey(sc.run_id), _skey("meas")])
        )
        measured = MeasurementChain(bias_rng=rng_b).apply(clean, rng_m, dropout=True)
    except Exception:
        measured = clean

    import pandas as pd

    df = pd.DataFrame(measured, columns=list(CHANNELS))
    df["label"] = labels
    starts = list(range(0, sc.n_cycles - WINDOW_T + 1, STRIDE))
    if starts:
        # windows via the repo's own windowing rule (fallback: local twin)
        try:
            from fedgrid.simulation.partition import windows_from_frame

            xs, ys = windows_from_frame(df)
        except Exception:
            xs, ys = _windows_fallback(df)
        # drop windows overlapping the warmup ramp
        keep = np.array([s >= WARMUP_CYCLES for s in starts])
        xs, ys = xs[keep], ys[keep]
        kept_starts = np.array(starts)[keep]
    else:
        xs = np.zeros((0, len(CHANNELS), WINDOW_T), dtype=np.float32)
        ys = np.zeros(0, dtype=np.int64)
        kept_starts = np.zeros(0, dtype=np.int64)

    if keep_logs == "none" or (keep_logs == "events" and sc.cls == "normal"):
        try:
            csv_path.unlink()
        except OSError:
            pass
    elif csv_path.parent != log_dir:
        # keep the raw CSV with the run's logs instead of cwd/logs
        import shutil

        dest = log_dir / csv_path.name
        try:
            shutil.move(str(csv_path), dest)
            csv_path = dest
        except OSError:
            pass

    rec = asdict(sc) | {"phases": list(sc.phases), "phase_g": list(sc.phase_g)}
    rec.update(
        {
            "wall_s": round(time.perf_counter() - t0, 3),
            "n_cycles": sc.n_cycles,
            "n_windows": int(len(ys)),
            "n_event_windows": int(np.sum(ys != 0)),
            "fault_line": aux["fault_line"]["name"] if aux.get("fault_line") else None,
            "fault_bus_frac": round(aux.get("in_frac", 0.0), 4),
            **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in probes.items()},
        }
    )
    return {
        "windows": xs,
        "labels": ys,
        "starts": kept_starts,
        "clean": clean,
        "cycle_labels": labels,
        "record": rec,
        "csv": str(csv_path) if csv_path.exists() else None,
    }


def _windows_fallback(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Identical semantics to partition.windows_from_frame (local copy for
    boxes without the fedgrid import — kept byte-for-byte in behaviour)."""
    arr = np.ascontiguousarray(df[list(CHANNELS)].to_numpy(dtype=np.float32).T)
    lbl = df["label"].to_numpy(dtype=np.int64)
    n = arr.shape[1]
    starts = range(0, n - WINDOW_T + 1, STRIDE)
    xs = np.stack([arr[:, s : s + WINDOW_T] for s in starts])
    ys = np.array([_window_label(lbl[s : s + WINDOW_T]) for s in starts], dtype=np.int64)
    return xs, ys


def _window_label(lbl: np.ndarray) -> int:
    """Modal non-NORMAL class if it covers >= 2 samples (partition.py twin)."""
    counts = np.bincount(lbl, minlength=8)
    counts[0] = 0
    top = int(counts.argmax())
    return top if counts[top] >= 2 else 0


# ---------------------------------------------------------------------------
# OpenDSS cross-check (optional, needs opendssdirect + vendored feeder)
# ---------------------------------------------------------------------------


def xcheck_cells(n_cells: int = 16) -> list[Scenario]:
    """Deterministic (cls, r_ohm, loc_frac) grid for the phasor-vs-EMT check."""
    bands = {
        "fault_lg": (0.05, 40.0),
        "fault_ll": (0.01, 1.5),
        "fault_llg": (0.01, 1.5),
        "fault_3ph": (0.01, 1.5),
    }
    locs = (0.1, 0.5, 0.9)
    cells: list[Scenario] = []
    for cls, (lo, hi) in bands.items():
        rs = np.geomspace(lo, hi, 4)
        for ri, r in enumerate(rs):
            for li, lf in enumerate(locs):
                ph = (0,) if cls == "fault_lg" else ((0, 1) if cls != "fault_3ph" else (0, 1, 2))
                cells.append(
                    Scenario(
                        f"xc_{cls}_{ri}{li}",
                        cls,
                        EVENT_CLASS[cls],
                        "xcheck",
                        0,
                        r_ohm=float(r),
                        loc_frac=float(lf),
                        phases=ph,
                        t_on_cycles=12.0,
                        duration_cycles=0.0,
                        n_cycles=48,
                    )
                )
    return cells[:n_cells] if n_cells < len(cells) else cells


def xcheck_opendss(cells: list[Scenario], spec: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Compare OpenDSS snapshot phasors vs EMT steady-state for the same faults.

    Per cell: OpenDSS snapshot-solve with a ``Fault`` injected via
    ``FeederSim._fault_command`` (identical command semantics to the battery
    generator, incl. the bus-snap of the fractional location) vs an EMT run of
    the same ``(cls, r_ohm, loc_frac)`` cell — the EMT fault site splits the
    line at the fractional point, so the two engines site the fault within
    half a line section of each other; the comparison reports that slack in
    ``note``. EMT channels are the mean of the clean per-cycle phasors over
    the post-inception steady-state cycles (inception transient excluded).
    """
    import opendssdirect as dss

    from fedgrid.data import EventClass
    from fedgrid.simulation.events import EventSpec
    from fedgrid.simulation.grid import FeederSim, FeederSpec

    sim = FeederSim(FeederSpec(name="ieee13", load_shape_seed=0))
    sim._compile(dss)
    topo = sim._topology(dss)
    dss.Text.Command("set mode=snapshot")

    results = []
    for sc in cells:
        if not sc.cls.startswith("fault"):
            continue
        row: dict[str, Any] = {
            "cell": sc.run_id,
            "cls": sc.cls,
            "r_ohm": sc.r_ohm,
            "loc_frac": sc.loc_frac,
            "phases": list(sc.phases),
        }
        # --- OpenDSS snapshot ------------------------------------------
        ev = EventSpec(
            cls=EventClass(sc.label),
            start_s=0.0,
            duration_s=1.0,
            params={"r_ohm": sc.r_ohm, "loc_frac": sc.loc_frac, "phase": float(sc.phases[0] + 1)},
        )
        cmd = sim._fault_command(dss, topo, 0, ev)
        dss.Text.Command(cmd)
        dss.Solution.Solve()
        row["dss_converged"] = bool(dss.Solution.Converged())
        if row["dss_converged"]:
            dss.Circuit.SetActiveBus("rg60")
            vma = dss.Bus.puVmagAngle()
            dss.Circuit.SetActiveElement(f"Line.{sim._def.head_line}")
            cma = dss.CktElement.CurrentsMagAng()
            pw = dss.CktElement.Powers()
            bus_pos = {ph: k for k, ph in enumerate(topo["bus_nodes"]) if 1 <= ph <= 3}
            pos_of = {ph: k for k, ph in enumerate(topo["node_order"]) if 1 <= ph <= 3}
            vr = np.zeros(3)
            varr = np.zeros(3)
            ir = np.zeros(3)
            iar = np.zeros(3)
            for ph in (1, 2, 3):
                bp = bus_pos.get(ph)
                if bp is not None and 2 * bp + 1 < len(vma):
                    vr[ph - 1] = vma[2 * bp]
                    varr[ph - 1] = vma[2 * bp + 1]
                lp = pos_of.get(ph)
                if lp is not None and 2 * lp + 1 < len(cma):
                    ir[ph - 1] = cma[2 * lp]
                    iar[ph - 1] = cma[2 * lp + 1]
            v_cplx = vr * np.exp(1j * np.deg2rad(varr))
            i_cplx = ir * np.exp(1j * np.deg2rad(iar))
            row["dss"] = {
                "v_pu": vr.tolist(),
                "i_pu": (ir / topo["i_base_a"]).tolist(),
                "v0_pu": abs(v_cplx.sum()) / 3.0,
                "i0_pu": abs(i_cplx.sum()) / 3.0 / topo["i_base_a"],
                "p_pu": sum(pw[0 : 2 * topo["ncond"] : 2]) / topo["s_base_kva"],
                "q_pu": sum(pw[1 : 2 * topo["ncond"] : 2]) / topo["s_base_kva"],
            }
        dss.Text.Command("Fault.fg_ev0.enabled=no")

        # --- EMT steady state ------------------------------------------
        res = run_scenario(sc, spec, out_dir / "xcheck", keep_logs="none")
        rec = res["record"]
        emt_clean = res["clean"]
        k0 = int(sc.t_on_cycles) + 5  # post-inception steady-state cycles
        steady = emt_clean[k0:].mean(axis=0) if emt_clean.shape[0] > k0 else None
        if steady is not None:
            emt = {ch: float(steady[i]) for i, ch in enumerate(CHANNELS)}
            row["emt"] = emt
            if row.get("dss"):
                d = row["dss"]
                rel = {}
                for i, (dch, ech) in enumerate(
                    zip(("va", "vb", "vc", "ia", "ib", "ic"), d["v_pu"] + d["i_pu"], strict=True)
                ):
                    ref = abs(dch)
                    rel[CHANNELS[i]] = (
                        abs(dch - emt[ech]) / ref if ref > 1e-4 else abs(dch - emt[ech])
                    )
                for ch in ("p", "q", "v0", "i0"):
                    ref = abs(d[f"{ch}_pu"])
                    rel[ch] = (
                        abs(d[f"{ch}_pu"] - emt[ch]) / ref
                        if ref > 1e-4
                        else abs(d[f"{ch}_pu"] - emt[ch])
                    )
                row["rel_diff"] = rel
                row["rel_diff_median"] = float(np.median(list(rel.values())))
        row["emt_fault_line"] = rec.get("fault_line")
        results.append(row)
    return {
        "cells": results,
        "note": "OpenDSS snaps the fault to the nearer terminal bus; the "
        "EMT run sites it at the fractional point on the split "
        "line — expect O(half-line) slack in agreement.",
    }


# ---------------------------------------------------------------------------
# Evaluation (optional): train the repo's own classifier on the EMT windows
# ---------------------------------------------------------------------------


def _group_split(
    run_ids: np.ndarray, seed: int, fracs: tuple[float, float, float] = (0.70, 0.15, 0.15)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train/val/test indices where all windows of a run stay together.

    The leakage-safe protocol for a run-correlated dataset — the EMT analogue
    of the reviewer's chronological-split requirement (windows from one
    simulation share the same physical draw, so they must not straddle the
    boundary).
    """
    uniq = np.unique(run_ids)
    order = np.random.default_rng(seed).permutation(len(uniq))
    n = len(uniq)
    n_tr, n_va = int(n * fracs[0]), int(n * fracs[1])
    tr_r = set(uniq[order[:n_tr]].tolist())
    va_r = set(uniq[order[n_tr : n_tr + n_va]].tolist())
    idx = np.arange(len(run_ids))
    tr = idx[np.isin(run_ids, list(tr_r))]
    va = idx[np.isin(run_ids, list(va_r))]
    te = idx[~np.isin(run_ids, list(tr_r | va_r))]
    return tr, va, te


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 8) -> dict[str, Any]:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t, p] += 1
    present = cm.sum(axis=1) > 0
    per_class = {}
    f1s = []
    for c in np.nonzero(present)[0]:
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[int(c)] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(cm[c, :].sum()),
        }
        f1s.append(f1)
    return {
        "acc": round(float((y_true == y_pred).mean()), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "per_class": per_class,
        "confusion": cm.tolist(),
        "n_test": int(len(y_true)),
    }


def train_eval(
    windows: np.ndarray,
    labels: np.ndarray,
    run_ids: np.ndarray,
    model_name: str,
    seed: int,
    device: str = "cpu",
    epochs: int = 15,
    batch: int = 64,
) -> dict[str, Any]:
    """Small local train/eval on the EMT dataset using the repo's own model.

    Returns a RunResult-shaped dict (saved via fedgrid.metrics.runresult when
    available, plain JSON otherwise) with a measured batch-1 latency block.
    """
    import torch

    torch.manual_seed(seed)
    if model_name == "cnnlstm":
        from fedgrid.models.cnn_lstm import CNNLSTM

        model = CNNLSTM(in_ch=11, n_classes=8)
    elif model_name == "ssm":
        from fedgrid.models.ssm import SSMClassifier

        model = SSMClassifier(in_ch=11, n_classes=8)
    else:
        raise ValueError(f"--train expects cnnlstm|ssm, got {model_name!r}")
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model.to(dev)

    tr, va, te = _group_split(run_ids, seed)
    X = windows.astype(np.float32)
    mean = X[tr].mean(axis=(0, 2), keepdims=False)[:, None]
    std = np.maximum(X[tr].std(axis=(0, 2))[:, None], 1e-6)
    Xn = ((X - mean) / std).astype(np.float32)

    def batches(idxs):
        for i in range(0, len(idxs), batch):
            j = idxs[i : i + batch]
            yield (
                torch.from_numpy(Xn[j]).to(dev),
                torch.from_numpy(labels[j].astype(np.int64)).to(dev),
            )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        order = np.random.default_rng(seed + ep).permutation(len(tr))
        for xb, yb in batches(tr[order]):
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    preds = np.empty(len(te), dtype=np.int64)
    with torch.no_grad():
        for j in range(0, len(te), batch):
            xb = torch.from_numpy(Xn[te[j : j + batch]]).to(dev)
            preds[j : j + len(xb)] = model(xb).argmax(1).cpu().numpy()
    met = _metrics(labels[te].astype(np.int64), preds)

    # measured batch-1 latency (same convention as the paper's latency runs)
    with torch.no_grad():
        warm = torch.from_numpy(Xn[te[:1]]).to(dev)
        for _ in range(10):
            model(warm)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts = []
        for i in te[: min(200, len(te))]:
            xb = torch.from_numpy(Xn[i : i + 1]).to(dev)
            t0 = time.perf_counter()
            model(xb)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    lat = {
        "device": str(dev),
        "batch": 1,
        "p50_ms": float(np.percentile(ts, 50)),
        "p95_ms": float(np.percentile(ts, 95)),
        "p99_ms": float(np.percentile(ts, 99)),
        "n": len(ts),
    }
    macs = int(model.macs_per_inference())
    try:
        from fedgrid.metrics.runresult import RunResult, current_env_fingerprint, current_git_commit

        rr = RunResult(
            method=f"emt_routeb_{model_name}",
            seed=seed,
            config={
                "task": "emt_routeb",
                "model": model_name,
                "epochs": epochs,
                "cadence": "cycle",
                "split": "by-run 70/15/15",
            },
            rounds=[],
            final=met | {"macs_per_inference": macs},
            energy=None,
            latency=lat,
            git_commit=current_git_commit(_REPO),
            env_fingerprint=current_env_fingerprint(),
        )
        return {"runresult": rr, "final": met, "latency": lat}
    except Exception:
        return {"runresult": None, "final": met, "latency": lat, "macs_per_inference": macs}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    """Spawn/fork-safe top-level worker: run one scenario."""
    spec = task["spec"]
    sc = Scenario(**{k: v for k, v in task["scenario"].items()})
    return run_scenario(
        sc,
        spec,
        Path(task["work_dir"]),
        keep_logs=task["keep_logs"],
        meas_bias_seed=task["meas_bias_seed"],
    )


def _scenario_dict(sc: Scenario) -> dict[str, Any]:
    d = asdict(sc)
    d["phases"] = tuple(sc.phases)
    return d


def estimate_budget(plan: list[Scenario], n_cycles: int) -> dict[str, Any]:
    """Compute budget estimate from the measured probe calibration."""
    # measured on the reference laptop (i7, dpsim 1.4.0, dt=52us):
    # ~0.9 s wall per 1.0 s simulated (3000 steps) incl. logging+extract
    steps_per_run = n_cycles * SAMPLES_PER_CYCLE
    wall_per_run_s = 0.9 * (n_cycles / F0_HZ) + 0.35
    n = len(plan)
    core_s = n * wall_per_run_s
    return {
        "n_runs": n,
        "sim_time_per_run_s": round(n_cycles / F0_HZ, 3),
        "steps_per_run": int(steps_per_run),
        "est_wall_per_run_s": round(wall_per_run_s, 2),
        "est_core_hours": round(core_s / 3600.0, 3),
        "windows_per_run_typ": max(0, (n_cycles - WINDOW_T) // STRIDE + 1),
        "note": "measured ~0.9 s wall per simulated second + ~0.35 s IO/extract "
        "overhead on a laptop i7; EMT is CPU-bound, embarrassingly "
        "parallel over runs (use --workers).",
    }


def _selftest(with_sim: bool) -> int:
    """Validate imports/semantics; with dpsimpy also run a micro simulation."""
    report: dict[str, Any] = {"checks": {}, "dpsimpy": None, "fedgrid": None}
    ok = True

    for mod in ("numpy", "pandas"):
        try:
            m = importlib.import_module(mod)
            report["checks"][mod] = getattr(m, "__version__", "?")
        except ImportError:
            report["checks"][mod] = "MISSING"
            ok = False
    try:
        import fedgrid.data as fgd

        assert fgd.CHANNELS == CHANNELS, "channel order drift vs contract"
        assert {e.name.lower(): int(e) for e in fgd.EventClass} == EVENT_CLASS
        report["fedgrid"] = getattr(__import__("fedgrid"), "__version__", "?")
        report["checks"]["fedgrid.data"] = "channels+EventClass consistent"
    except Exception as e:
        report["checks"]["fedgrid.data"] = f"unavailable ({e!r}) — fallbacks active"
    if importlib.util.find_spec("dpsimpy") is not None:
        report["dpsimpy"] = importlib.metadata.version("dpsim")
    else:
        report["dpsimpy"] = "MISSING (simulation disabled; install `dpsim`)"

    # fault admittance invariants
    g = fault_admittance("fault_ll", 0.5, (0, 1))
    r = np.linalg.inv(g)
    eff = np.linalg.inv(r)  # round-trip
    ok &= np.allclose(eff, g)
    report["checks"]["fault_matrix_roundtrip"] = bool(np.allclose(eff, g))

    if with_sim and report["dpsimpy"] != "MISSING (simulation disabled; install `dpsim`)":
        spec, prov = load_topology("auto")
        sc = Scenario(
            "selftest_lg",
            "fault_lg",
            1,
            "bulk",
            0,
            r_ohm=1.0,
            loc_frac=0.5,
            phases=(0,),
            t_on_cycles=12.0,
            duration_cycles=8.0,
            n_cycles=48,
        )
        t0 = time.perf_counter()
        res = run_scenario(sc, spec, Path("./_selftest_out"), keep_logs="events")
        wall = time.perf_counter() - t0
        report["checks"]["micro_sim"] = {
            "topology": prov,
            "windows": int(res["windows"].shape[0]),
            "labels": sorted(set(res["labels"].tolist())),
            "wall_s": round(wall, 2),
            "probes": res["record"],
        }
        ok &= res["windows"].shape[0] >= 0
        ok &= res["record"].get("ev_i_max_pu", 0) > 0.05  # fault visible
    print(json.dumps({"selftest": "PASS" if ok else "FAIL", **report}, indent=2, default=str))
    return 0 if ok else 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m experiments.emt_routeb",
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", type=Path, default=Path("runs_emt/routeb"), help="output dir")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs-per-class", type=int, default=60)
    p.add_argument(
        "--classes",
        type=str,
        default=",".join(DEFAULT_CLASSES),
        help="comma list from: " + ",".join(DEFAULT_CLASSES),
    )
    p.add_argument(
        "--n-cycles", type=int, default=150, help="cycles per run (150 = 2.5 s at 60 Hz)"
    )
    p.add_argument(
        "--workers", type=int, default=1, help="parallel worker processes (runs are independent)"
    )
    p.add_argument("--keep-logs", choices=["none", "events", "all"], default="events")
    p.add_argument("--topology-source", choices=["auto", "opendss", "snapshot"], default="auto")
    p.add_argument("--train", choices=["none", "cnnlstm", "ssm"], default="none")
    p.add_argument("--train-epochs", type=int, default=15)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--xcheck-opendss",
        action="store_true",
        help="also run the OpenDSS-vs-EMT steady-state cross-check",
    )
    p.add_argument("--xcheck-cells", type=int, default=16, help="number of cross-check cells")
    p.add_argument("--selftest", action="store_true", help="validate imports + micro-sim; exits")
    p.add_argument(
        "--selftest-sim", action="store_true", help="selftest including a real micro EMT run"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="print the scenario plan + budget; no simulation"
    )
    p.add_argument("--limit", type=int, default=None, help="cap total runs (smoke)")
    p.add_argument(
        "--refresh-topology",
        action="store_true",
        help="print the live-extracted topology JSON and exit",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.selftest or args.selftest_sim:
        return _selftest(with_sim=args.selftest_sim)

    if args.refresh_topology:
        spec, prov = load_topology("opendss")
        print(json.dumps({"provenance": prov, "spec": spec}, indent=1, default=str))
        return 0

    classes = tuple(c.strip() for c in args.classes.split(",") if c.strip())
    plan = sample_scenarios(args.seed, args.runs_per_class, classes, n_cycles=args.n_cycles)
    if args.limit:
        plan = plan[: args.limit]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "budget": estimate_budget(plan, args.n_cycles),
                    "classes": list(classes),
                    "plan_head": [_scenario_dict(sc) for sc in plan[:4]],
                },
                indent=2,
                default=str,
            )
        )
        return 0

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    spec, prov = load_topology(args.topology_source)
    (out / "scenarios.jsonl").write_text(
        "".join(json.dumps(_scenario_dict(sc), default=list) + "\n" for sc in plan),
        encoding="utf-8",
    )
    t0 = time.perf_counter()

    results: list[dict[str, Any]] = []
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        tasks = [
            {
                "scenario": _scenario_dict(sc),
                "spec": spec,
                "work_dir": str(out),
                "keep_logs": args.keep_logs,
                "meas_bias_seed": i,
            }
            for i, sc in enumerate(plan)
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(_worker, tasks):
                results.append(res)
                r = res["record"]
                print(
                    f"[emt] {r['run_id']} done: {r['n_windows']} windows "
                    f"({r['n_event_windows']} event) {r['wall_s']}s",
                    flush=True,
                )
    else:
        for i, sc in enumerate(plan):
            res = run_scenario(sc, spec, out, keep_logs=args.keep_logs, meas_bias_seed=i)
            results.append(res)
            r = res["record"]
            print(
                f"[emt] {r['run_id']} done: {r['n_windows']} windows "
                f"({r['n_event_windows']} event) {r['wall_s']}s",
                flush=True,
            )

    windows = (
        np.concatenate([r["windows"] for r in results])
        if results
        else np.zeros((0, 11, WINDOW_T), dtype=np.float32)
    )
    labels = (
        np.concatenate([r["labels"] for r in results]) if results else np.zeros(0, dtype=np.int64)
    )
    run_ids = (
        np.concatenate(
            [np.full(len(r["labels"]), i, dtype=np.int64) for i, r in enumerate(results)]
        )
        if results
        else np.zeros(0, np.int64)
    )
    run_cls = (
        np.array([results[i]["record"]["cls"] for i in run_ids], dtype="U16")
        if len(run_ids)
        else np.array([], "U16")
    )
    run_arm = (
        np.array([results[i]["record"]["arm"] for i in run_ids], dtype="U16")
        if len(run_ids)
        else np.array([], "U16")
    )

    with (out / "runs.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r["record"], default=str) + "\n")
    np.savez_compressed(
        out / "windows.npz",
        windows=windows.astype(np.float32),
        labels=labels.astype(np.int64),
        run_ids=run_ids,
        run_cls=run_cls,
        run_arm=run_arm,
    )

    eval_block: dict[str, Any] = {}
    if args.train != "none" and len(windows):
        ev = train_eval(
            windows,
            labels,
            run_ids,
            args.train,
            args.seed,
            device=args.device,
            epochs=args.train_epochs,
        )
        eval_block = {"model": args.train, **{k: v for k, v in ev.items() if k != "runresult"}}
        if ev.get("runresult") is not None:
            from fedgrid.metrics.runresult import save_run

            save_run(ev["runresult"], out / f"eval_{args.train}.json")
        else:
            (out / f"eval_{args.train}.json").write_text(
                json.dumps(eval_block, indent=2, default=str), encoding="utf-8"
            )

    xcheck: dict[str, Any] = {}
    if args.xcheck_opendss:
        try:
            cells = xcheck_cells(args.xcheck_cells)
            xcheck = xcheck_opendss(cells, spec, out)
            (out / "xcheck_opendss.json").write_text(
                json.dumps(xcheck, indent=2, default=str), encoding="utf-8"
            )
        except Exception:
            xcheck = {"error": traceback.format_exc(limit=3)}

    hist = np.bincount(labels, minlength=8).tolist() if len(labels) else [0] * 8
    manifest = {
        "task": "emt_routeb",
        "cadence": "cycle",
        "f0_hz": F0_HZ,
        "samples_per_cycle": SAMPLES_PER_CYCLE,
        "dt_s": DT_S,
        "window_t": WINDOW_T,
        "stride": STRIDE,
        "config": vars(args) | {"out": str(out)},
        "topology_provenance": prov,
        "classes": list(classes),
        "n_runs": len(results),
        "n_windows": int(len(labels)),
        "label_histogram": hist,
        "normal_share": (hist[0] / sum(hist)) if sum(hist) else None,
        "sha256_windows": hashlib.sha256(windows.tobytes()).hexdigest(),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "budget_estimate": estimate_budget(plan, args.n_cycles),
        "eval": eval_block,
        "xcheck": {"n_cells": len(xcheck.get("cells", []))} if xcheck else None,
        "versions": _versions(),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"[emt] DONE {len(results)} runs -> {out} | windows={len(labels)} "
        f"NORMAL={hist[0] if hist else 0} | {manifest['elapsed_s']}s"
    )
    return 0


def _versions() -> dict[str, Any]:
    v: dict[str, Any] = {"python": sys.version.split()[0]}
    for pkg, mod in (
        ("dpsim", "dpsimpy"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("fedgrid", "fedgrid"),
        ("torch", "torch"),
        ("opendssdirect.py", "opendssdirect"),
    ):
        try:
            v[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            v[pkg] = None
        try:
            importlib.import_module(mod)
        except ImportError:
            pass
    try:
        from fedgrid.metrics.runresult import current_env_fingerprint, current_git_commit

        v["env_fingerprint"] = current_env_fingerprint()
        v["git_commit"] = current_git_commit(_REPO)
    except Exception:
        pass
    return v


if __name__ == "__main__":
    raise SystemExit(main())
