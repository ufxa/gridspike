# REALISM.md — simulation realism parameters and their sources

Binding under CONTRACT.md amendment 2026-07-28b: every realism parameter used by
`src/fedgrid/simulation/*` carries a verifiable source (standard, dataset, or
paper). Rows marked **fitted-synthetic** are synthetic constructs *fitted to* the
cited published statistics — stated openly rather than passed off as measured
data (honesty over convenience). `tests/test_simulation.py` asserts that every
data row in the tables below has a non-empty Source cell.

Basis legend: **measured-data** = vendored real/measured-model data used directly;
**standard** = value taken from a normative standard; **fitted-synthetic** = synthetic
model fitted to the cited statistics; **modeling-choice** = documented engineering
approximation with its anchor.

## Feeder models and data provenance

| Parameter | Value / distribution | Basis | Source |
|---|---|---|---|
| Feeder models | IEEE 13-, 34-, 37-, 123-bus + IEEE EU LV Test Feeder, unmodified DSS files vendored under `src/fedgrid/simulation/feeders/` | measured-data | IEEE PES Distribution Test Feeders (Kersting, "Radial distribution test feeders", IEEE PES Winter Meeting 2001); files from the official DSS-Extensions distribution `dss-extensions/electricdss-tst` (BSD-3-Clause, EPRI/DSS-Extensions; licence copy at `feeders/LICENSE.txt`) |
| EU LV master variant | `eulv/fedgrid_master.dss` = pristine `Master.dss` minus the 906-monitor redirect and energymeter (state read via API instead) | modeling-choice | Header comment in the file documents the exact diff; all component files pristine |
| Solver engine | OpenDSSDirect.py 0.9.4 / DSS C-API 0.14.5 (AltDSS/DSS-Extensions engine, version recorded in generation manifest) | measured-data | DSS-Extensions project (docs/SOTA.md §pins); manuscript will state the alternative-engine caveat their README requests |
| Substation metering point | PT on feeder-head busbar, CT on first line section (rg60/650632, 800/l1, 799r/l35, 149/l115, 1/line1) | modeling-choice | IEC 61850 merging-unit placement at the substation bay; feeder-head elements taken from the published feeder one-lines |
| IEEE 37 grounding signature | Ungrounded delta feeder: ground faults appear as neutral-displacement (unfaulted phases toward 1.73 pu, `v0` large, `i0` ~ 0), not fault current | measured-data | Published IEEE 37-bus specification (delta, ungrounded); classic ungrounded-system behaviour per IEEE Std 142 ("Green Book") grounding practice |
| Quasi-static cadence | 60 s solver step, OpenDSS daily mode, regulator/control action via `controlmode=time` with the feeders' own regcontrol delays | modeling-choice | EPRI OpenDSS reference manual (daily solution mode, time-based control queue); cadence matches 1-min RMS trend telemetry of substation gateways |
| Load voltage sensitivity | Constant-PQ above 0.7 pu, constant-Z below (`vminpu=0.7`) | modeling-choice | OpenDSS load-model convention (EPRI OpenDSS reference manual); keeps deep-fault solves physical and convergent |
| Solver non-convergence handling | A step that fails the fixed-point power flow is re-solved in place with raised iteration caps (200) then Newton; if still non-convergent it **holds the last valid sample** (merging-unit repeat-last semantics) and is counted. A day aborts only past the tolerated rate: **> 5 % of its 1440 steps** (`NONCONV_ABORT_FRAC`) or **≥ 10 consecutive** (`NONCONV_ABORT_CONSEC`). Observed worst case across the 14 battery federations: ~1.04 % of a day (IEEE 37-bus, α=0.1, seed 1, day 171) and 0.28 % (IEEE 34-bus, seed 0, day 123); aggregate ≈ 0.005 %; most feeder-days are 0. The 5 % abort floor sits above this stiff-feeder ill-conditioning noise (holds gracefully) while still aborting a genuine topology-level divergence (≫ 5 % of a day, or a long consecutive run). Per-generation held-step count/rate recorded in `manifest.json → nonconvergence`. | modeling-choice | Merging units repeat the previous sampled value on a missed conversion (IEC 61850-9-2LE sampled-value stream semantics — hold-last); ill-conditioning of the long rural IEEE 34-bus feeder is a documented OpenDSS convergence challenge (EPRI OpenDSS reference manual, radial-feeder solution notes) |

## Load and DER

| Parameter | Value / distribution | Basis | Source |
|---|---|---|---|
| Residential base profiles | 100 measured-model 1-min daily household profiles (vendored with the EU LV feeder), used raw-kW for single homes and mean-normalised for aggregates | measured-data | IEEE PES EU LV Test Feeder load profile set; generated with the CREST high-resolution demand model — Richardson, Thomson, Infield, Clifford, *Energy and Buildings* 42(10):1878–1887, 2010 |
| MV spot-load aggregation | Spot load of P kW = mean of n = clip(P/2, 1, 30) distinct household profiles (diversity smoothing) | modeling-choice | Coincidence/diversity treatment per Kersting, *Distribution System Modeling and Analysis*, CRC Press (diversity factors chapter) |
| Commercial profile | Night base 0.35, sigmoid ramp 07–09 h to plateau 1.0, decay 17–20 h; weekend plateau 0.55; AR(1) texture sigma 3 % | fitted-synthetic | Fitted to DOE/NREL Commercial Reference Building (medium office) load data — Deru et al., NREL/TP-5500-46861, 2011 |
| Commercial/residential mix | IEEE 37 all-commercial (published as an underground urban delta feeder); mixed feeders: spot loads >= 100 kW commercial, rest residential aggregate; EU LV all-residential | modeling-choice | Feeder descriptions in the published IEEE test feeder documents |
| Seasonal structure | Annual sinusoid, amplitude 0.25 (winter:summer average-day energy ~1.67), winter peak mid-January | fitted-synthetic | Fitted to ELEXON Load Profile Class 1 (Domestic Unrestricted) seasonal profiles — ELEXON, "Load Profiles and their use in Electricity Settlement" |
| Weekday/weekend structure | Residential x1.05 weekend; commercial weekend plateau 0.55 of weekday | fitted-synthetic | ELEXON PC1 weekend uplift (domestic); NREL/TP-5500-46861 weekend office load |
| Per-phase unbalance | Fixed per-client phase-allocation factors, N(1, 0.05) clipped to +/-10 %, applied to single-phase loads (on top of the feeders' built-in unbalanced spot loads) | modeling-choice | Bounded so steady-state voltage unbalance stays within EN 50160:2010 (<= 2 %) / ANSI C84.1 (<= 3 %) limits; checked by test |
| PV penetration levels | Two levels: 15 % and 40 % of nominal feeder load, alternating across clients | standard/paper | Steady-state hosting-capacity range per Hoke, Butler, Hambrick, Kroposki, *IEEE Trans. Sustainable Energy* 4(2):350–357, 2013 (many feeders host 30–50 % without violations) |
| PV siting | 40 % of load buses, seeded per client (assets fixed across days), capacity proportional to host load | modeling-choice | Rooftop-PV ~ load-proportional adoption assumption, penetration anchored to Hoke et al. 2013 |
| PV electrical model | OpenDSS `PVSystem`, kVA = 1.1 Pmpp, unity pf, daily irradiance shape, default 25 °C (no temperature derating — stated simplification) | modeling-choice | EPRI OpenDSS PVSystem model documentation |
| Clear-sky irradiance | Haurwitz model GHI = 1098 cosZ exp(−0.057/cosZ); locale latitudes 34.5–38 °N (US feeders), 52.4 °N (EU LV) | standard/paper | Haurwitz, *J. Meteorology* 2:154–166, 1945; validated among simple clear-sky models in Reno, Hansen, Stein, SAND2012-2389 |
| Cloud transients | Two-state Markov clear/cloudy (mean sojourn 40/20 min), clear k_t ~ N(0.99, 0.02), cloudy episode depth ~ U(0.15, 0.60) + 8 % jitter, ~2-min edge smoothing | fitted-synthetic | Bimodal 1-min clear-sky-index statistics — Woyte, Belmans, Nijs, *Solar Energy* 81(2):195–206, 2007; minute-scale ramp depths 60 %+ — Mills & Wiser, LBNL-3884E (2010); Lave, Kleissl, Stein, *IEEE J. Photovoltaics* 3(1), 2013 |

## Events

| Parameter | Value / distribution | Basis | Source |
|---|---|---|---|
| Event rate | Poisson(1.4/day) per client; with T=64/stride-16 windowing this yields >= 90 % NORMAL windows (measured ~92–94 %) | modeling-choice | Contract realism block (NORMAL >= 90 % prior); rate tuned analytically and verified by test; rare-event operation consistent with distribution fault statistics (Short, *Electric Power Distribution Handbook*, CRC 2004: ~0.5–1 faults/mi/yr) with PQ events added |
| Event class mix | Categorical from the client's Dirichlet(alpha=0.5) draw over the 7 event classes (NORMAL share set by the rate) | protocol | Hsu, Qi, Brown, arXiv:1909.06335 (Dirichlet non-IID protocol; contract data section). Deliberately overrides the natural fault-type mix — the label skew *is* the federated protocol |
| LG fault resistance | Log-uniform 0.05–40 Ω (bolted → high-impedance) | paper | Dagenhart, "The 40-Ω ground-fault phenomenon", *IEEE Trans. Industry Applications* 36(1):30–32, 2000; typical values per De Andrade & Sorrentino, IEEE PES T&D-LA 2010 |
| LL/LLG/3PH fault resistance | Log-uniform 0.01–1.5 Ω (metallic to low-resistance arc) | paper | De Andrade & Sorrentino, "Typical expected values of the fault resistance in power systems", IEEE PES T&D-LA 2010 |
| EU LV fault-resistance floor | r >= 0.2 Ω on the LV feeder | standard | LV fault loop always includes cable/joint/earth-path impedance — earth-fault loop impedance scale per IEC 60364 / BS 7671 |
| Fault location | Line drawn with probability ∝ conductor length (exposure ~ length), position discretised to the nearer terminal bus | modeling-choice | Exposure-proportional fault incidence per Short 2004; bus-granular attach is an OpenDSS Fault-object constraint, stated openly |
| Faulted phase(s) | Uniform over the phases present at the fault bus | modeling-choice | No public per-phase asymmetry statistics adopted; uniform stated openly |
| Fault episode duration | LG: lognormal(med 3 min, sigma 0.6) clip [2, 15] min; LL/LLG/3PH: lognormal(med 2.5 min, sigma 0.5) clip [2, 10] min | fitted-synthetic (cadence-constrained) | Sub-minute breaker/fuse clearing (IEEE Std C37.104 reclosing practice) is below the 60-s cadence; modelled episodes represent what 1-min RMS telemetry does observe: high-impedance and repeated-reclose/lockout conditions persisting for minutes — HIF persistence per Ghaderi, Ginn, Mohammadpour, *Electric Power Systems Research* 143:376–388, 2017. See Known simplifications |
| SAG magnitude | Source-side target U(0.75, 0.88) pu so the observed metering-bus depth lands predominantly in the IEEE 1159 sustained-undervoltage band (0.8–0.9 pu) after regulator action | standard | IEEE Std 1159-2019 (sustained undervoltage category, > 1 min); regulator counteraction is the feeders' own regcontrol response |
| SAG origin | **Always** a source-side depression (remote/upstream fault or motor-start regime): the source per-unit is held at `depth_pu` for every sag episode. On PV-equipped feeders a 50 % `mech_pv` branch *superposes* on that same mechanism — irradiance x0.1 for the episode, scheduling confined to 09:30–15:30 — so local PV infeed also vanishes and imported power rises during the dip. No sag is PV-*caused*; PV only changes the signature | paper | Bollen, *Understanding Power Quality Problems*, IEEE Press 2000 (sag origins); PV-driven voltage excursions on high-penetration feeders — Katiraei & Agüero, *IEEE Power & Energy Magazine* 9(3):62–71, 2011 |
| SWELL magnitude | Source-side target U(1.08, 1.15) pu → observed within IEEE 1159 overvoltage band (1.1–1.2 pu); duration regime > 1 min | standard | IEEE Std 1159-2019 (overvoltage category); load-rejection/switching origin per Bollen 2000 |
| SAG/SWELL duration | U(2, 10) min | standard | IEEE 1159 sustained-variation regime (> 1 min) at the 1-min telemetry cadence |
| OVERLOAD | Feeder-wide load multiplier U(1.4, 2.0) for lognormal(med 15 min) clip [5, 45] min — the *sustained* CLP component (2–5x figures are seconds-scale inrush) | paper/standard | Cold-load pickup — Short 2004; IEEE Std C37.104 (CLP considerations for reclosing) |
| OVERLOAD feeder caps | Effective pickup clamped to what the feeder's voltage regime can serve at winter evening peak: ieee34 → 1.6, ieee123 → 1.9, others 2.0 (verified by worst-case solves) | modeling-choice | Sectionalized-restoration practice on capacity-constrained feeders (Short 2004, restoration/CLP chapter); caps determined empirically against the vendored feeders' solvable region |
| Class prior | NORMAL >= 90 % of windows (measured ~92–94 % at seed 0); primary metrics under this prior | contract | CONTRACT.md realism block (amendment 2026-07-28b); verified by `test_class_prior` |

## Frequency and measurement chain

| Parameter | Value / distribution | Basis | Source |
|---|---|---|---|
| Frequency wander | Stationary AR(1)/OU, sigma = 18 mHz, relaxation tau = 4 min, clipped +/-0.3 Hz | fitted-synthetic | Fitted to measured grid-frequency statistics: Rydin Gorjão et al., *Nature Communications* 11:6362, 2020 (open frequency database); ENTSO-E Continental Europe standard frequency range +/-50 mHz (ENTSO-E Operation Handbook, Policy 1) |
| Event-coupled frequency deviation | Class-dependent dips: faults −20…−120 mHz, overload −10…−50 mHz, sag −5…−20 mHz; swell (load rejection) +10…+50 mHz. Applied as a **step** at onset (held flat through the episode), then a **fixed 8-sample tail** `dev*exp(-(k+1)/2)`, k=0..7, cut to exactly zero after the eighth sample and truncated at the array end. Wander + coupling clipped to +/-0.3 Hz | fitted-synthetic | Scale anchored to the ENTSO-E +/-50 mHz normal band and the ~100 mHz scale of documented deterministic frequency deviations (ENTSO-E Operation Handbook; Rydin Gorjão et al. 2020) |
| CT accuracy | Class 0.5: per-instrument bias U(−0.3 %, +0.3 %) + per-sample noise N(0, 0.05 %) truncated at +/-0.2 % — total bounded by the 0.5 % class limit by construction | standard | IEC 61869-2 (current transformers, class 0.5 ratio-error limit); conformance over 1e4 samples asserted by test |
| PT/VT accuracy | Same class-0.5 envelope on voltage channels | standard | IEC 61869-3 (inductive voltage transformers, class 0.5) |
| P/Q metering accuracy | Same 0.5 % envelope on p, q (combined CT+VT+angle chain) | standard | IEC 62053-22 (class 0.5 S static meters for active energy) |
| Frequency measurement error | N(0, 1 mHz) truncated +/-5 mHz | standard | IEEE Std C37.118.1-2011 (PMU steady-state frequency error limit 0.005 Hz) |
| ADC quantisation | 16-bit uniform over fixed full-scale ranges: V 0–2 pu, I 0–40 pu (protective-core range, ~20x on a CT rated ~2x load), P/Q +/-16 pu, fdev +/-0.5 Hz | standard/modeling-choice | IEC 61869-9 / IEC 61850-9-2LE sampled-value practice (16-bit conservative); protective accuracy range per IEC 61869-2 (e.g. 5P20); clip = core saturation, stated openly |
| Sensor dropout | Poisson 0.5 episodes/day, geometric length (mean 3, max 10 samples), all channels hold-last-value (stale-data semantics) ≈ 0.1 % samples | fitted-synthetic | Real synchrophasor-network availability 99–99.9 % — NASPI Technical Report, "Synchrophasor Data Quality" (2017); hold-last-value per RTU/MU buffering behaviour |
| Nominal PF for per-unit bases | S_base = total load kW / 0.95; I_base = S_base/(3 V_LN) | modeling-choice | Typical distribution planning power factor (Short 2004) |

## Federation / consensus network

| Parameter | Value / distribution | Basis | Source |
|---|---|---|---|
| Non-IID protocol | Dirichlet(alpha) label skew (default 0.5, sweep {0.1, 0.5, 1.0}) x per-client feeder/load-shape identity | protocol | Hsu, Qi, Brown, arXiv:1909.06335; contract amendment 2026-07-28 §4 |
| Windowing | T = 64 samples @ 60 s, stride 16 → 87 windows/day, ~2.6k windows/client at 30 days (contract band 1.5–3k) | contract | CONTRACT.md data contract + the data-simulation module implementation note (derivation in `partition.py`) |
| Calendar sampling | Client-day d simulates calendar day 3 + 12d: the 30-day default spans the full year (seasonal structure present in the data) and cycles all weekday/weekend residues | modeling-choice | Deterministic mapping in `partition.py`; seasonal/day-type factors per the ELEXON/NREL rows above |
| Splits / normalisation | Per-client 70/15/15 seeded; per-channel affine normaliser fitted on train split only | contract | CONTRACT.md data contract; leakage bar asserted by test |
| PBFT network latency | One-way message latency ~ Lognormal(median 25 ms, sigma_ln 0.5) truncated [5, 200] ms; p99 < ~100 ms (`simulation/network.py`, consumed by the consensus/attacks module's PBFTSim) | fitted-synthetic | Lognormal RTT form per Aikat, Kaur, Smith, Jeffay, "Variability in TCP round-trip times", ACM IMC 2003; scale set inside IEC/TR 61850-90-12 WAN transfer-time class budgets |
| Scale-up path | Same pipeline runs unmodified on NREL SMART-DS feeders (declared scale-up reference, not used for the paper's numbers) | reference | NREL SMART-DS, OEDI submission 2981 (docs/SOTA.md §datasets) |

## Known simplifications (stated openly)

1. **Cadence vs protection speed.** At the fixed 60-s quasi-static cadence,
   sub-minute breaker/fuse clearing sequences are unobservable. Modelled fault
   episodes therefore represent the sustained conditions that 1-min RMS trend
   telemetry does capture (high-impedance faults, repeated-reclose/lockout,
   pre-isolation intervals). Point-on-wave transients and protection-speed
   dynamics are out of scope of this benchmark, and the manuscript will say so.
2. **Bus-granular fault location.** OpenDSS `Fault` objects attach to buses;
   the along-line position (uniform over conductor length) is discretised to
   the nearer terminal bus.
3. **Frequency is synthesised.** OpenDSS quasi-static solves fix system
   frequency; `fdev` is a stochastic process fitted to measured statistics
   (sources above), coupled to events, and is not a solver output.
4. **PV temperature derating omitted** (25 °C flat); irradiance is spatially
   uniform per feeder (one cloud field per feeder-day).
5. **Vendored EU LV master** omits the 906 monitors and the energymeter of the
   pristine `Master.dss` (state is read through the API); all electrical
   component files are unmodified.
6. **Windows do not cross day boundaries** (days are independent simulations;
   87 windows/day instead of 90).
7. **Bit-reproducibility is per-platform**: `make_federation(seed=0)` is
   bit-stable on a given OS/library build; cross-platform identity of solver
   floating point is not claimed.
