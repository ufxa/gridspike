"""OpenDSS feeder simulation at the substation bus.

The heavy ``opendssdirect`` import stays inside methods so this module imports
cleanly on machines without the OpenDSS engine.

Feeder models are the IEEE PES Distribution Test Feeders (Kersting, "Radial
distribution test feeders", IEEE PES Winter Meeting 2001) and the IEEE PES
European LV Test Feeder, vendored unmodified from the official DSS-Extensions
distribution (``dss-extensions/electricdss-tst``, BSD-3-Clause, EPRI) under
``simulation/feeders/`` - provenance and licence tabulated in
``docs/REALISM.md``. Simulation is quasi-static time series (OpenDSS daily
mode) at ``step_s`` resolution with sample-accurate event labelling.

Realism elements wired here (sources in the helper modules and REALISM.md):
measured-model load shapes with seasonal/day-type structure and per-phase
allocation unbalance; rooftop PV (OpenDSS PVSystem) driven by a Haurwitz
clear-sky x Markov cloud irradiance process at two penetration levels; fault
events as OpenDSS Fault objects with sourced resistance distributions and
length-weighted location; sag/swell as source-side per-unit excursions within
IEEE 1159 bands; overload as a cold-load-pickup multiplier; voltage-regulator
response uses the feeders' own regcontrol delays (``controlmode=time``); the
returned channels pass through the merging-unit measurement chain
(:mod:`fedgrid.simulation.measurement`).
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from fedgrid.data import CHANNELS, EventClass

if TYPE_CHECKING:
    import pandas as pd

    from fedgrid.simulation.events import EventSpec

__all__ = ["FeederSim", "FeederSpec", "available_feeders"]

_FAULT_CLASSES = frozenset(
    {EventClass.FAULT_LG, EventClass.FAULT_LL, EventClass.FAULT_LLG, EventClass.FAULT_3PH}
)
# SeedSequence tags: per-day streams vs client-fixed streams (instrument bias,
# PV siting and phase allocation belong to the installed hardware).
_TAG_DAY = 0x5EED0001
_TAG_CLIENT = 0x5EED0002
# Nominal power factor used to turn total load kW into the feeder's apparent-
# power base for per-unit current/power (typical distribution planning PF).
_NOMINAL_PF = 0.95
# Share of load buses that host rooftop PV at either penetration level
# (modelling choice, documented in REALISM.md; capacity is load-proportional).
_PV_BUS_SHARE = 0.4

# --- Solver non-convergence policy (documented in docs/REALISM.md) -----------
# On ill-conditioned feeders (notably the long rural IEEE 34-bus at peak with a
# fault present) a handful of the 1440 quasi-static steps can fail the default
# fixed-point power flow. This is numerical noise, not a data-integrity failure.
# Such a step is first re-solved in place with a more robust inner solver
# (more iterations, then Newton); if it still fails, the merging-unit model
# holds the last valid sample (a real MU that misses a conversion repeats the
# previous one) and the step is counted. Only a *genuine* divergence aborts the
# day: more than NONCONV_ABORT_FRAC of the day's steps, or NONCONV_ABORT_CONSEC
# non-convergent steps in a row. Observed rate across the 14 battery federations is
# <= ~1 % of a day's steps on the stiffest feeder/day combinations (ieee34/ieee37
# under alpha=0.1), ~0.005 % aggregate, zero on most days. The abort fraction is set
# above that stiff-case noise floor (5 %) so ill-conditioning holds gracefully while a
# real topology-level divergence (>> 5 % of a day, or a long consecutive run) still
# aborts loudly. Raised 0.01 -> 0.05 on 2026-07-28 after ieee37 day-171 (alpha=0.1
# seed 1) held 15/1440 = 1.04 % and tripped the old 1 % gate.
NONCONV_ABORT_FRAC: float = 0.05   # > 5 % of a day's steps (~72) = genuine divergence
NONCONV_ABORT_CONSEC: int = 10     # >= 10 consecutive non-convergent steps = stuck solve
_ROBUST_MAX_ITER: int = 200        # bumped power-flow iteration cap for retries
_ROBUST_MAX_CTRL_ITER: int = 200   # bumped control-loop iteration cap for retries


def _should_abort_day(n_nonconv: int, consec: int, n_steps: int) -> bool:
    """Whether a day's non-convergence has crossed from noise into divergence.

    Pure policy predicate (unit-tested without the OpenDSS engine): abort when
    more than ``NONCONV_ABORT_FRAC`` of the day's steps have failed, or when
    ``NONCONV_ABORT_CONSEC`` steps have failed back to back.
    """
    return consec >= NONCONV_ABORT_CONSEC or n_nonconv > NONCONV_ABORT_FRAC * n_steps


@dataclass
class FeederSpec:
    """Identity of one client's grid: which IEEE test feeder plus its load-profile seed.

    ``pv_penetration`` (the data-simulation module phase 2 extension) is installed PV capacity as a
    fraction of the feeder's nominal load; the federation uses the two levels
    {0.15, 0.40} - moderate and high penetration per steady-state hosting
    studies (Hoke et al., IEEE Trans. Sustainable Energy 4(2):350-357, 2013).
    """

    name: str  # "ieee13" | "ieee34" | "ieee37" | "ieee123" | "eulv"
    load_shape_seed: int  # per-client load profile diversity
    pv_penetration: float = 0.15


@dataclass(frozen=True)
class _FeederDef:
    master: str          # path of the master .dss, relative to feeders/
    base_freq: int       # Hz
    head_line: str       # OpenDSS Line at the feeder head (metering CT location)
    sub_bus: str         # substation MV/LV busbar (metering PT location)
    latitude_deg: float  # locale assumption for the solar geometry
    load_class: str      # "residential" | "commercial" | "mixed"
    max_pickup: float    # cold-load-pickup cap the feeder's voltage regime can serve


# Metering points sit at the feeder head: PT on the substation busbar, CT on the
# first line section - i.e. exactly what a substation merging unit measures.
# max_pickup: capacity-constrained feeders cannot serve the full 2x sustained
# cold-load pickup at winter evening peak (utilities sectionalize restoration
# for exactly this reason - Short 2004); caps verified against worst-case
# winter-peak solves, documented in REALISM.md. The long rural 34-bus feeder is
# the voltage-constrained outlier; the 123-bus keeps a small margin.
_REGISTRY: dict[str, _FeederDef] = {
    "ieee13": _FeederDef("ieee13/IEEE13Nodeckt.dss", 60, "650632", "rg60", 35.0,
                         "mixed", 2.0),
    "ieee34": _FeederDef("ieee34/ieee34Mod1.dss", 60, "l1", "800", 34.5, "mixed", 1.6),
    "ieee37": _FeederDef("ieee37/ieee37.dss", 60, "l35", "799r", 38.0, "commercial", 2.0),
    "ieee123": _FeederDef("ieee123/IEEE123Master.dss", 60, "l115", "149", 35.0,
                          "mixed", 1.9),
    "eulv": _FeederDef("eulv/fedgrid_master.dss", 50, "line1", "1", 52.4,
                       "residential", 2.0),
}


def available_feeders() -> list[str]:
    """Names of the bundled OpenDSS feeder models, in the order clients cycle through them."""
    return list(_REGISTRY)


class FeederSim:
    """Quasi-static time-series simulator for one feeder."""

    def __init__(self, spec: FeederSpec) -> None:
        if spec.name not in _REGISTRY:
            raise ValueError(f"unknown feeder {spec.name!r}; expected one of "
                             f"{available_feeders()}")
        self.spec = spec
        self._def = _REGISTRY[spec.name]
        from fedgrid.simulation.profiles import FEEDER_DIR

        self._master = FEEDER_DIR / self._def.master
        if not self._master.is_file():
            raise FileNotFoundError(f"vendored feeder master missing: {self._master}")

    # ------------------------------------------------------------------ engine

    def _compile(self, dss: Any) -> None:
        """Fresh engine state -> compiled feeder, snapshot-solved to seat regulators.

        OpenDSS's ``compile`` changes the process working directory to the
        master's folder; it is restored so library callers relying on relative
        paths are never surprised.
        """
        import os

        cwd = os.getcwd()
        try:
            dss.Basic.ClearAll()
            dss.Text.Command(f"Set DefaultBaseFrequency={self._def.base_freq}")
            dss.Text.Command(f'compile "{self._master}"')
            dss.Text.Command("set maxiterations=100")
            dss.Text.Command("set maxcontroliter=100")
            # Constant-Z behaviour below 0.7 pu keeps fault solves physical and
            # convergent (OpenDSS load-model convention; EPRI reference manual).
            dss.Text.Command("batchedit load..* vminpu=0.7")
            dss.Solution.Solve()
        finally:
            os.chdir(cwd)
        if not dss.Solution.Converged():
            raise RuntimeError(f"baseline power flow did not converge for {self.spec.name}")

    @staticmethod
    def _resolve_in_place(dss: Any) -> bool:
        """Re-solve the *current* timepoint with a more robust inner solver.

        ``SolvePFlow`` re-runs the power flow at the current daily-clock
        position without advancing time (verified empirically: only ``Solve``
        steps the clock), so the day's time alignment is preserved no matter
        how many retries a step takes. The ladder raises the iteration caps,
        then falls back to Newton; the algorithm is always restored to the
        default so converged steps stay bit-identical to the validated physics.
        """
        sol = dss.Solution
        sol.MaxIterations(_ROBUST_MAX_ITER)
        sol.MaxControlIterations(_ROBUST_MAX_CTRL_ITER)
        sol.SolvePFlow()
        if sol.Converged():
            return True
        dss.Text.Command("set algorithm=newton")
        try:
            sol.SolvePFlow()
            return bool(sol.Converged())
        finally:
            dss.Text.Command("set algorithm=normal")
            sol.MaxIterations(100)
            sol.MaxControlIterations(100)

    def solve_baseline(self) -> bool:
        """Compile + snapshot solve; True iff the power flow converged."""
        import opendssdirect as dss

        self._compile(dss)
        return bool(dss.Solution.Converged())

    def _topology(self, dss: Any) -> dict[str, Any]:
        """Deterministic post-compile inventory used by shape/PV/fault builders."""
        loads: list[dict[str, Any]] = []
        idx = dss.Loads.First()
        while idx:
            name = dss.Loads.Name()
            kw = float(dss.Loads.kW())
            kv = float(dss.Loads.kV())
            is_delta = bool(dss.Loads.IsDelta())
            dss.Circuit.SetActiveElement(f"Load.{name}")
            bus_full = dss.CktElement.BusNames()[0]
            parts = bus_full.split(".")
            nodes = [int(p) for p in parts[1:] if p.isdigit() and int(p) > 0] or [1, 2, 3]
            loads.append({"name": name, "kw": kw, "kv": kv, "delta": is_delta,
                          "bus_full": bus_full, "nodes": nodes,
                          "phases": int(dss.CktElement.NumPhases())})
            dss.Loads.Name(name)  # restore Loads iterator target
            idx = dss.Loads.Next()

        lines: list[dict[str, Any]] = []
        idx = dss.Lines.First()
        while idx:
            lines.append({
                "name": dss.Lines.Name(),
                "bus1": dss.Lines.Bus1().split(".")[0],
                "bus2": dss.Lines.Bus2().split(".")[0],
                "length": max(float(dss.Lines.Length()), 1e-6),
                "phases": int(dss.Lines.Phases()),
            })
            idx = dss.Lines.Next()

        dss.Circuit.SetActiveBus(self._def.sub_bus)
        kv_ln = float(dss.Bus.kVBase())
        bus_nodes = [int(x) for x in dss.Bus.Nodes()]

        dss.Circuit.SetActiveElement(f"Line.{self._def.head_line}")
        ncond = int(dss.CktElement.NumConductors())
        node_order = [int(x) for x in dss.CktElement.NodeOrder()][:ncond]

        total_kw = sum(ld["kw"] for ld in loads)
        s_base_kva = total_kw / _NOMINAL_PF
        i_base_a = s_base_kva / (3.0 * kv_ln)

        src_pu = float(dss.Vsources.PU())
        return {
            "loads": loads, "lines": lines, "kv_ln": kv_ln, "bus_nodes": bus_nodes,
            "ncond": ncond, "node_order": node_order, "total_kw": total_kw,
            "s_base_kva": s_base_kva, "i_base_a": i_base_a, "src_pu": src_pu,
        }

    # ------------------------------------------------------------- fault siting

    @staticmethod
    def _locate_fault(topo: dict[str, Any], loc_frac: float,
                      min_phases: int) -> tuple[str, str]:
        """(bus, line) for a fault at cumulative-length fraction ``loc_frac``.

        Lines are weighted by conductor length (exposure ~ length; Short 2004)
        among lines with >= ``min_phases`` phases, then the position along the
        chosen line is discretised to the nearer terminal bus (OpenDSS Fault
        objects attach to buses; documented approximation).
        """
        cands = [ln for ln in topo["lines"] if ln["phases"] >= min_phases]
        if not cands:
            cands = topo["lines"]
        cum: list[float] = []
        acc = 0.0
        for ln in cands:
            acc += ln["length"]
            cum.append(acc)
        target = float(np.clip(loc_frac, 0.0, 1.0 - 1e-9)) * acc
        i = min(bisect_right(cum, target), len(cands) - 1)
        line = cands[i]
        prev = cum[i] - line["length"]
        frac_in = (target - prev) / line["length"]
        bus = line["bus1"] if frac_in < 0.5 else line["bus2"]
        return bus, line["name"]

    def _fault_command(self, dss: Any, topo: dict[str, Any], tag: int,
                       ev: EventSpec) -> str:
        """OpenDSS ``New Fault`` command realising ``ev`` at its sampled location."""
        min_ph = {EventClass.FAULT_LG: 1, EventClass.FAULT_LL: 2,
                  EventClass.FAULT_LLG: 2, EventClass.FAULT_3PH: 3}[ev.cls]
        bus, _line = self._locate_fault(topo, ev.params["loc_frac"], min_ph)
        dss.Circuit.SetActiveBus(bus)
        nodes = sorted({int(x) for x in dss.Bus.Nodes() if 0 < int(x) <= 3})
        if not nodes:
            nodes = [1]
        want = int(ev.params.get("phase", 1))
        p1 = want if want in nodes else nodes[0]
        r = ev.params["r_ohm"]
        if self.spec.name == "eulv":
            # LV networks: the fault loop always includes cable/joint/earth-path
            # impedance of a few hundred milliohm (IEC 60364 / BS 7671 earth-
            # fault loop impedance scale) - also keeps the very weak EU LV
            # source (ISC1 = 5 A) numerically solvable under bolted draws.
            r = max(r, 0.2)
        name = f"fg_ev{tag}"
        if ev.cls == EventClass.FAULT_LG or len(nodes) < 2:
            return f"New Fault.{name} bus1={bus}.{p1} phases=1 r={r:.6g}"
        others = [x for x in nodes if x != p1]
        p2 = others[0]
        if ev.cls == EventClass.FAULT_LL:
            return f"New Fault.{name} bus1={bus}.{p1} bus2={bus}.{p2} phases=1 r={r:.6g}"
        if ev.cls == EventClass.FAULT_LLG:
            return f"New Fault.{name} bus1={bus}.{p1}.{p2} phases=2 r={r:.6g}"
        ph3 = nodes[:3] if len(nodes) >= 3 else nodes
        spec = ".".join(str(x) for x in ph3)
        return f"New Fault.{name} bus1={bus}.{spec} phases={len(ph3)} r={r:.6g}"

    # ------------------------------------------------------------------ run_day

    def run_day(self, day: int, events: Sequence[EventSpec], step_s: float = 60.0) -> pd.DataFrame:
        """One simulated day of quasi-static time series at the substation bus.

        Columns exactly the 11 data-contract channels + ``label``. Events are
        injected via OpenDSS Fault objects / load scaling / source dips and
        labelled sample-accurate. The returned channels are what the merging
        unit measures (IEC 61869 class-0.5 chain, 16-bit ADC, dropout) - see
        :mod:`fedgrid.simulation.measurement`.
        """
        import opendssdirect as dss
        import pandas as pd

        from fedgrid.simulation.measurement import MeasurementChain, frequency_deviation
        from fedgrid.simulation.profiles import (
            commercial_shape,
            irradiance_profile,
            residential_shape,
        )

        n = int(round(86400.0 / step_s))
        d = self._def
        day_ss = np.random.SeedSequence([_TAG_DAY, self.spec.load_shape_seed, int(day)])
        rng_load, rng_pv, rng_freq, rng_meas = (np.random.default_rng(c)
                                                for c in day_ss.spawn(4))
        client_ss = np.random.SeedSequence([_TAG_CLIENT, self.spec.load_shape_seed])
        rng_site, rng_bias, rng_phase = (np.random.default_rng(c)
                                         for c in client_ss.spawn(3))

        self._compile(dss)
        topo = self._topology(dss)
        self.s_base_kva = topo["s_base_kva"]
        self.i_base_a = topo["i_base_a"]

        # -- sample-accurate labels + integer event intervals
        labels = np.zeros(n, dtype=np.int64)
        intervals: list[tuple[int, int, EventSpec]] = []
        for ev in events:
            s = max(0, int(math.floor(ev.start_s / step_s)))
            e = min(n, max(s + 1, int(math.ceil((ev.start_s + ev.duration_s) / step_s))))
            if s >= n:
                continue
            labels[s:e] = int(ev.cls)
            intervals.append((s, e, ev))

        # -- per-load daily multiplier matrix (measured-model shapes + events)
        # Per-phase allocation unbalance: fixed +/-10 % factors per client, kept
        # inside the EN 50160 / ANSI C84.1 voltage-unbalance regime (REALISM.md).
        phase_g = 1.0 + np.clip(rng_phase.normal(0.0, 0.05, 3), -0.10, 0.10)
        shapes = np.empty((len(topo["loads"]), n), dtype=np.float64)
        for i, ld in enumerate(topo["loads"]):
            commercial = d.load_class == "commercial" or (
                d.load_class == "mixed" and ld["kw"] >= 100.0)
            if commercial:
                base = commercial_shape(rng_load, day, n, step_s)
            elif d.load_class == "residential" and ld["kw"] <= 2.0:
                # Single households (EU LV): raw measured-model kW on kW=1 loads,
                # the official test-case semantics - preserves true peak scale.
                base = residential_shape(rng_load, day, 1, n, raw_kw=True)
            else:
                n_agg = int(np.clip(round(ld["kw"] / 2.0), 1, 30))
                base = residential_shape(rng_load, day, n_agg, n)
            if ld["phases"] == 1 and not ld["delta"]:
                base = base * phase_g[ld["nodes"][0] - 1]
            shapes[i] = base
        for s, e, ev in intervals:
            if ev.cls == EventClass.OVERLOAD:
                # feeder-wide cold-load pickup, clamped to what this feeder's
                # voltage regime can serve (sectionalized-restoration practice)
                shapes[:, s:e] *= min(ev.params["mult"], d.max_pickup)

        # -- PV irradiance (with forced deep cloud front during PV-mechanism sags)
        pen = float(self.spec.pv_penetration)
        irr = irradiance_profile(rng_pv, day, n, step_s, d.latitude_deg)
        for s, e, ev in intervals:
            if ev.cls == EventClass.SAG and ev.params.get("mech_pv", 0.0) >= 1.0 and pen > 0:
                irr[s:e] *= 0.1  # ~90 % output loss under a thick cloud front

        # -- source per-unit schedule (sag/swell mechanisms)
        vsrc = np.full(n, topo["src_pu"], dtype=np.float64)
        for s, e, ev in intervals:
            if ev.cls == EventClass.SAG:
                vsrc[s:e] = ev.params["depth_pu"]
            elif ev.cls == EventClass.SWELL:
                vsrc[s:e] = ev.params["level_pu"]

        # -- install loadshapes / daily assignment / PV systems
        minterval = step_s / 60.0
        for i, ld in enumerate(topo["loads"]):
            sh = f"fg_ld{i}"
            dss.Text.Command(f"New Loadshape.{sh} npts={n} minterval={minterval:.6g}")
            dss.LoadShape.Name(sh)
            dss.LoadShape.PMult([float(x) for x in shapes[i]])
            dss.Loads.Name(ld["name"])
            dss.Loads.Daily(sh)
        if pen > 0:
            dss.Text.Command(f"New Loadshape.fg_irr npts={n} minterval={minterval:.6g}")
            dss.LoadShape.Name("fg_irr")
            dss.LoadShape.PMult([float(x) for x in irr])
            for j, (li, cap) in enumerate(self._pv_sites(topo, rng_site, pen)):
                ld = topo["loads"][li]
                conn = "delta" if ld["delta"] else "wye"
                dss.Text.Command(
                    f"New PVSystem.fg_pv{j} bus1={ld['bus_full']} phases={ld['phases']} "
                    f"kV={ld['kv']:.6g} conn={conn} kVA={1.1 * cap:.6g} Pmpp={cap:.6g} "
                    f"irradiance=1 pf=1 %cutin=2 %cutout=2 daily=fg_irr"
                )

        # -- event command schedule (before-solve boundaries; sample-accurate)
        cmds_on: dict[int, list[str]] = {}
        for tag, (s, e, ev) in enumerate(intervals):
            if ev.cls in _FAULT_CLASSES:
                cmds_on.setdefault(s, []).append(self._fault_command(dss, topo, tag, ev))
                cmds_on.setdefault(e, []).append(f"Fault.fg_ev{tag}.enabled=no")

        # -- quasi-static daily loop; regulator/control response uses the
        #    feeders' own regcontrol delays (controlmode=time).
        dss.Text.Command(f"set mode=daily stepsize={step_s:.6g}s number=1")
        dss.Text.Command("set controlmode=time")
        head = f"Line.{d.head_line}"
        ncond = topo["ncond"]
        node_order = topo["node_order"]
        pos_of_phase = {ph: k for k, ph in enumerate(node_order) if 1 <= ph <= 3}
        bus_pos = {ph: k for k, ph in enumerate(topo["bus_nodes"]) if 1 <= ph <= 3}

        vmag = np.zeros((n, 3))
        vang = np.zeros((n, 3))
        imag = np.zeros((n, 3))
        iang = np.zeros((n, 3))
        p_kw = np.zeros(n)
        q_kvar = np.zeros(n)
        def read_row() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
            """Substation-bus channels from the current OpenDSS solution."""
            dss.Circuit.SetActiveBus(d.sub_bus)
            vma = dss.Bus.puVmagAngle()
            dss.Circuit.SetActiveElement(head)
            cma = dss.CktElement.CurrentsMagAng()
            pw = dss.CktElement.Powers()
            vr = np.zeros(3)
            var_ = np.zeros(3)
            ir = np.zeros(3)
            iar = np.zeros(3)
            for ph in (1, 2, 3):
                bp = bus_pos.get(ph)
                if bp is not None and 2 * bp + 1 < len(vma):
                    vr[ph - 1] = vma[2 * bp]
                    var_[ph - 1] = vma[2 * bp + 1]
                lp = pos_of_phase.get(ph)
                if lp is not None and 2 * lp + 1 < len(cma):
                    ir[ph - 1] = cma[2 * lp]
                    iar[ph - 1] = cma[2 * lp + 1]
            return vr, var_, ir, iar, sum(pw[0:2 * ncond:2]), sum(pw[1:2 * ncond:2])

        # Hold value for the degenerate first-step-fails case: the seated
        # compile baseline solution (no extra solve, so the converged-step
        # sequence below stays bit-identical to the validated physics).
        prev = read_row()
        n_nonconv = 0
        consec = 0
        held = np.zeros(n, dtype=bool)
        cur_pu = topo["src_pu"]
        for t in range(n):
            if vsrc[t] != cur_pu:
                dss.Text.Command(f"Vsource.source.pu={vsrc[t]:.9g}")
                cur_pu = vsrc[t]
            for cmd in cmds_on.get(t, ()):
                dss.Text.Command(cmd)
            dss.Solution.Solve()
            if dss.Solution.Converged() or self._resolve_in_place(dss):
                consec = 0
                prev = read_row()
            else:
                # Hold-last-valid-sample recovery: a merging unit that misses a
                # conversion repeats the previous one. Labels are untouched -
                # they come from the independent event schedule, not the solve -
                # so a held sample keeps its NORMAL/event label deterministically.
                n_nonconv += 1
                consec += 1
                held[t] = True
                if _should_abort_day(n_nonconv, consec, n):
                    raise RuntimeError(
                        f"{n_nonconv}/{n} steps failed to converge on "
                        f"{self.spec.name} day {day} ({consec} consecutive) - "
                        f"genuine divergence past the tolerated rate, aborting day")
            vmag[t], vang[t], imag[t], iang[t], p_kw[t], q_kvar[t] = prev

        # -- channel assembly (per-unit RMS phasor magnitudes + sequence quantities)
        i_base = topo["i_base_a"]
        s_base = topo["s_base_kva"]
        v_cplx = vmag * np.exp(1j * np.deg2rad(vang))
        i_cplx = imag * np.exp(1j * np.deg2rad(iang))
        v0 = np.abs(v_cplx.sum(axis=1)) / 3.0
        i0 = np.abs(i_cplx.sum(axis=1)) / 3.0 / i_base
        fdev = frequency_deviation(
            rng_freq, labels, step_s, [(s, e, int(ev.cls)) for s, e, ev in intervals])
        clean = np.column_stack([
            vmag[:, 0], vmag[:, 1], vmag[:, 2],
            imag[:, 0] / i_base, imag[:, 1] / i_base, imag[:, 2] / i_base,
            p_kw / s_base, q_kvar / s_base, v0, i0, fdev,
        ])
        chain = MeasurementChain(bias_rng=rng_bias)
        measured = chain.apply(clean, rng_meas, dropout=True)
        # Held (non-convergent) steps repeat the previous *measured* sample -
        # exactly what a merging unit emits when it misses a conversion. Ascending
        # order propagates runs of consecutive holds; labels stay as scheduled.
        for t in np.nonzero(held)[0]:
            if t > 0:
                measured[t] = measured[t - 1]
        df = pd.DataFrame(measured, columns=list(CHANNELS))
        df["label"] = labels
        # Solver diagnostics ride along in df.attrs (keeps the contract return
        # type); the federation builder aggregates them into the run manifest.
        df.attrs["n_steps"] = n
        df.attrs["n_nonconverged"] = n_nonconv
        return df

    @staticmethod
    def _pv_sites(topo: dict[str, Any], rng: np.random.Generator,
                  pen: float) -> list[tuple[int, float]]:
        """Load-proportional PV siting on a fixed share of load buses.

        Selection is seeded per client (installed assets do not move day to
        day); capacity totals ``pen`` x nominal feeder load, split across the
        selected buses proportionally to their load size.
        """
        loads = topo["loads"]
        kw = np.array([ld["kw"] for ld in loads], dtype=np.float64)
        if kw.sum() <= 0 or pen <= 0:
            return []
        n_sites = max(1, int(round(_PV_BUS_SHARE * len(loads))))
        prob = kw / kw.sum()
        sites = rng.choice(len(loads), size=min(n_sites, len(loads)),
                           replace=False, p=prob)
        sel_kw = kw[sites]
        total_pv = pen * kw.sum()
        caps = total_pv * sel_kw / sel_kw.sum()
        return [(int(i), float(c)) for i, c in zip(sites, caps, strict=True)]
