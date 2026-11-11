# Route B — EMT-domain validation (DPsim)

Companion design record for `experiments/emt_routeb.py`. This is the
reviewer-facing experiment that answers the Route B option in
`E:/paper7/annot_full.txt` (p. 14): *"a separate experiment on a timescale
compatible with fast dynamics, using EMT/HIL … 3–5 representative scenarios
(LG, LL, LLG, 3PH, high-impedance or equivalent), fast-timescale signals,
run CNN-LSTM/Mamba, measure latency — no need to repeat the 1,845 runs."*

Status: **implemented, smoke-validated locally, not yet run as a battery**
(the full battery is deliberately deferred to the remote box — see
[§7](#7-compute-budget)).

---

## 1. Why EMT and why DPsim

The OpenDSS generator is quasi-static phasor domain: every sample is a
steady-state power-flow snapshot. Raising its sampling rate does not change
the physics, so it cannot support the paper's protection-adjacent claims.
DPsim (`dpsimpy`, PyPI wheels for CPython 3.10–3.14) is a true EMT solver —
dynamic phasor / MNA-domain transient simulation at µs timestep — and is
small enough to run inside the existing pipeline. `villas-dataprocessing`
is intentionally **not** used: DPsim CSV logs are read with pandas directly
(one less remote dependency, documented in the script docstring).

## 2. Network model

- EMT replica of the IEEE 13-node test feeder (the same feeder behind
  federation clients 0 and 5), parameterised from the vendored
  `feeders/ieee13/IEEE13Nodeckt.dss`. With `--topology-source auto` the
  topology is extracted live from OpenDSS when `opendssdirect` is
  importable; otherwise an embedded snapshot (same extraction, pinned in
  the source) is used — the script never silently changes topology source;
  the provenance string is recorded in `manifest.json`.
- **Substitution transformer restored to nameplate**: XHL = 8 %,
  %r = 0.5 %/winding on 5 MVA (≈ 0.0346 + j0.2769 Ω at 4.16 kV) plus a
  small upstream Thevenin (X/R = 7). The published DSS case divides the
  transformer impedance by 1000 (a stiff benchmark); restoring nameplate is
  required for sags to propagate and for fault currents to be physical.
  With the restored impedance, pre-event head voltage ≈ 0.97 pu and head
  current ≈ 0.94 pu — correct order for 3.5 MW of connected load.
- The OpenDSS `Switch=y` tie `671692` is a node alias, not a branch:
  bus 692 is merged into 671 (a literal near-zero PiLine makes the MNA
  system singular).
- 1-phase voltage regulators 650→RG60 (XHL = 0.01 % on 1666 kVA) are
  skipped — O(1e-5) Ω. The 4.16/0.48 kV service transformer XFM1 is folded:
  its loads sit at bus 633 (same kW/kvar drawn from the MV system; the
  ≈2 % secondary drop is an accepted approximation).
- A 1e7 Ω bleed shunt on every node gives floating downstream nodes a
  reference without measurably loading the feeder.

### 2.1 Load model — RXLoad deliberately avoided

This was the hardest integration bug and is a known DPsim 1.4.0 sharp edge:

- `RXLoad` draws **no current** unless `do_init_from_nodes_and_terminals`
  is enabled, and that initialiser emits NaN across the whole network when
  any load phase carries zero power — which the unbalanced IEEE13 inventory
  (single-phase 611/634/675/692 loads) has everywhere.
- Instead every spot load is built as **explicit per-phase R||L shunts**
  (constant-Z wye-equivalent): `R = V_ln²/P`, `L = V_ln²/(2πf·Q)` per phase,
  phases with zero power get no branch. Verified numerically: a 485 kW /
  242 kvar phase-A load draws ≈ 533 A·(R||X equivalent seen at the source)
  with physically consistent phase voltages.
- The system starts from zero (`do_init_from_nodes_and_terminals(False)`)
  and the first `WARMUP_CYCLES = 5` cycles are discarded. The network ramps
  to steady state within ~2 cycles (observed).

### 2.2 Metering

Identical semantics to `simulation/grid.py`: PT = node voltage at the
substation bus `rg60`, CT = branch current of head line `650632`.
Bases: `V_LN = 4.16 kV/√3`, `I_BASE = 500 A`, `S_BASE = 5 MVA` (OpenDSS
`kva` nameplate) — the same constants `grid.py` uses.

## 3. Events

Parameter distributions are inherited verbatim from
`simulation/events.py`; only the realisation mechanism is EMT.

| class | mechanism | parameters |
|---|---|---|
| `fault_lg` | 3-ph `Switch` closed by `SwitchEvent3Ph`; `closed_resistance = G⁻¹` with G = phase→ground conductance | r ~ logU(0.05, 40) Ω (Dagenhart) |
| `fault_lg_hif` | same, upper band | r ~ logU(15, 40) Ω; label stays `FAULT_LG`, `arm="hif"` |
| `fault_ll` | G = 2×2 conductance block between the two phases (no ground) | r ~ logU(0.01, 1.5) Ω |
| `fault_llg` | G = per-phase conductances to ground (OpenDSS wye semantics) | r ~ logU(0.01, 1.5) Ω |
| `fault_3ph` | G = 3×3 conductance to ground | r ~ logU(0.01, 1.5) Ω |
| `sag` | upstream source switchover to a depth-scaled EMF (fast-transfer-switch analogue) | depth ~ U(0.75, 0.88) pu |
| `swell` | same, level 1.08–1.15 pu | level ~ U(1.08, 1.15) |
| `overload` | extra R||L load sized `(mult−1)×`total load, switched in **behind the head CT (bus 632)** | mult ~ U(1.4, 2.0) |
| `normal` | none | load_mult ~ U(0.7, 1.1), phase_g ±10 % |

Fault location is uniform over cumulative conductor length (as OpenDSS)
but the chosen line is **split into two PiLine segments at the fractional
point** — continuous siting, strictly finer than OpenDSS's bus-granular
fault placement. ~45 % of fault runs are sustained (duration 0 → held to
sim end), the rest self-clear after 6–36 cycles — realistic protective
relay/recloser timescale.

Fault `t_on` is drawn in *fractional* cycles (`U(WARMUP+12, WARMUP+35)`),
so inception lands at a random point on the waveform — the EMT solver sees
the true point-on-wave inception transient, which the phasor domain cannot
represent at all.

## 4. Channel extraction — the contract, verbatim, at cycle cadence

Per 60 Hz cycle (K = 320 samples/cycle, dt = 52.0833 µs — inside DPsim's
recommended EMT range):

- `va,vb,vc` = |phasor|/V_LN from the exact-bin DFT of the PT signals.
- `ia,ib,ic` = |phasor|/I_BASE from the CT signals.
- `p,q` = Re/Im of `−Σ V·conj(I)` / S_BASE (DPsim logs branch current in
  the node→ground direction; the sign flip restores the contract
  "power into the feeder" convention).
- `v0,i0` = |Σ phasors|/3 on the contract bases.
- `fdev` = measured per-cycle frequency deviation (angle drift of the
  phase-A phasor, resolved at cycle rate) **plus** the same AR(1) wander
  as `measurement.frequency_deviation` with the event-coupled synthetic
  term disabled — the event signature in `fdev` is *measured*, not
  injected. This is the deliberate difference from the OpenDSS pipeline:
  there `fdev` was fully synthetic; here inception/clearance transients
  produce real drift.

The clean 11×n_cycles frame then goes through the unmodified
`MeasurementChain` (IEC 61869-2 class-0.5 PT bias+noise, CT noise, 16-bit
ADC, NASPI-rate dropout with hold-last) and `windows_from_frame`
(T = 64, stride 16, ≥ 2-event-samples label rule) — both reused verbatim,
so the classifier sees the same `(11, 64)` tensor as the telemetry
pipeline.

**Cadence disclosure (must stay in the paper):** a T=64 window here is
**64 power-frequency cycles ≈ 1.07 s**, not 64 minutes. Outputs carry
`cadence: "cycle"` and the manifest records `window_t: 64` alongside
`samples_per_cycle: 320` so the two clocks can never be conflated.

## 5. Outputs

| file | content |
|---|---|
| `scenarios.jsonl` | full sampled plan (seed → params) before any simulation |
| `runs.jsonl` | per-run record: scenario + physics probes + wall time + window yield |
| `windows.npz` | `windows (N,11,64) f32`, `labels`, `run_ids` (split group), `run_cls`, `run_arm` |
| `manifest.json` | config echo, env fingerprint, git commit, versions, topology provenance, label histogram, SHA-256 of window bytes, wall budget |
| `eval_<model>.json` | `--train cnnlstm|ssm`: RunResult (acc, macro-F1, per-class, confusion, batch-1 latency p50/p95/p99), **split by run** (leakage-safe at the new cadence) |
| `xcheck_opendss.json` | `--xcheck-opendss`: OpenDSS snapshot-solve phasors vs EMT post-inception steady state for a grid of (class, r, loc) cells |

## 6. Validated so far (local smoke, `runs-per-class 2`, 100 cycles)

All nine scenario types produce physically plausible signatures; no NaNs
(anywhere in the window tensor); windows and labels are correct.

| scenario | preV | preI | ev v_min | ev i_max | ev i0 |
|---|---|---|---|---|---|
| normal | 0.97 | 0.83–0.97 | — | — | — |
| fault_lg bolted | 0.97 | 0.98 | 0.62 | 7.5 | 2.10 |
| fault_lg ~40 Ω | 0.97 | 0.95 | 0.97 | 1.0 | 0.04 |
| fault_lg_hif | 0.97 | 0.66–0.84 | 0.97–0.98 | 0.91 | 0.06–0.09 |
| fault_ll | 0.97 | 0.68–0.99 | 0.53–0.60 | 7.0–8.1 | 0.05–0.06 |
| fault_llg | 0.97 | 0.93–1.01 | 0.70 | 5.3–6.0 | 0.85–1.65 |
| fault_3ph | 0.97 | 0.97–0.99 | 0.65–0.70 | 5.7–5.9 | 0.07–0.09 |
| sag 0.80 | 0.97 | 0.88 | 0.84 → achieved 0.77–0.85 | | |
| swell 1.10 | 0.98 | 0.66 | v_max 1.11 | | |
| overload 1.7× | 0.97 | 0.93–1.03 | i_max 1.51–1.69 | | |

Zero-sequence behaves correctly: high for LG/LLG, near zero for LL/3PH.
`--selftest-sim` passes (with and without fedgrid on the path); the
train/eval path runs end-to-end (accuracy meaningless at 36 windows — the
mechanism is what is being smoke-tested).

## 7. Compute budget

Measured locally (single worker, Windows, i7-class CPU):

- ~1.7–2.0 s wall per run at 100 cycles (32 000 steps at dt = 52 µs).
- Full battery at `--runs-per-class 50` (450 runs, 100 cycles each):
  ≈ 13 min serial; ≈ 2 min at `--workers 8`. Negligible.
- The expensive knob is `n_cycles`, not topology: 150 cycles → ~2.5 s/run.
- `xcheck` adds ~2–4 s per cell × 16 cells.

The battery is cheap enough to run on the same remote box as the federated
experiments; no GPU required for simulation (only for `--train`).

## 8. Known limitations (disclose in the paper)

1. **Constant-Z loads.** The R||L equivalents reproduce the OpenDSS kW/kvar
   draw at nominal voltage exactly, but a real EMT study would want at
   least ZIP composition on the feeders. Inside a ±30 % sag the constant-Z
   response is within a few % of constant-power — noted, accepted.
2. **Sag/swell mechanism.** Source switchover produces the correct
   magnitude excursion at the meter but a different inception transient
   than e.g. an upstream remote fault would. It is the same mechanism the
   OpenDSS generator uses (source per-unit held), so the comparison is
   honest.
3. **XFM1 folded** — 634-bus loads sit at 633 (see §2). Regulators skipped.
4. **`fdev` at cycle cadence** resolves real transients, but the wander
   overlay inherits its AR(1) parameters from a model tuned at 60 s cadence;
   it is a small additive noise term and is documented as such, not as a
   measured frequency process.
5. **No secondary network** (240 V / 480 V behind XFM1 and the spot-load
   transformers is absent — same as OpenDSS aggregate loads).
6. **capacitor 675** is the only multi-kvar shunt; it is modelled at its
   nameplate kvar with the OpenDSS unit count — its breaker schedule is not
   replayed (it is static in the DSS case anyway).
7. **xcheck slack.** OpenDSS snaps the fault to a bus; EMT sites it at the
   fractional point — expect O(half-line-section) disagreement in `v`/`i`,
   which the report quantifies rather than hides.

## 9. Remote run recipe

```bash
python -m venv /venv/emt && /venv/emt/bin/pip install "dpsim==1.4.*" numpy pandas torch
# (opendssdirect only needed for --xcheck-opendss and live topology extraction)
cd fedgrid   # repo bundle
PYTHONPATH=$PWD/src python -m experiments.emt_routeb --selftest
PYTHONPATH=$PWD/src python -m experiments.emt_routeb --selftest-sim
PYTHONPATH=$PWD/src python -m experiments.emt_routeb --dry-run
PYTHONPATH=$PWD/src python -m experiments.emt_routeb \
    --out runs_emt/routeb_s0 --seed 0 --runs-per-class 50 --workers 8 \
    --train cnnlstm --train-epochs 15 --xcheck-opendss
```

`--selftest` needs no dpsimpy; `--selftest-sim` runs a 48-cycle fault and
verifies extraction end-to-end. Everything else is read-only against the
repo — no OpenDSS installation needed unless `--xcheck-opendss` or
`--topology-source opendss` is requested.
