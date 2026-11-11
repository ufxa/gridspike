# REAL_DATA.md — real (non-simulated) measurement datasets for external validity

**Purpose.** Every number in the paper currently comes from OpenDSS quasi-static simulation
(`docs/REALISM.md`). The most damaging reviewer objection available is *"all of this is synthetic."*
This document records a live, link-verified survey of **real, publicly downloadable measurement
datasets** that could support an external-validity section, and states honestly what each one would
and would not buy us.

**Method / provenance.** Everything below was checked against the live web on **2026-07-28**.
Landing pages were fetched, download URLs probed with HTTP requests, and for the leading candidates
the **actual data was downloaded and parsed** — row counts, column names, channel fill rates, label
distributions and window yields quoted here are **measured by us**, not copied from the papers.
Where a claim could not be verified, it says so. Nothing here is asserted from memory.

**Contract we must map onto** (CONTRACT.md §Data contract):
`x: float32 (C=11, T=64)`, channels `va, vb, vc, ia, ib, ic, p, q, v0, i0, fdev`
(per-unit RMS phasor magnitudes, P/Q, zero-sequence V/I, frequency deviation), 60 s cadence;
`y ∈ {NORMAL, FAULT_LG, FAULT_LL, FAULT_LLG, FAULT_3PH, OVERLOAD, SAG, SWELL}`;
K = 10 clients, non-IID via feeder identity × Dirichlet(α); NORMAL ≥ 90 %.

**One-line summary of the survey.** There is **no** open dataset of real field measurements that
has all 11 channels, our 8 classes, and ~10 measurement sites. The realistic choice is between
*scale and federation with hardware-in-the-loop physics* (**MSU/ORNL**, ranked #1) and *genuine
field measurements at small scale* (**LBNL µPMU**, ranked #2). They are complementary and cheap
enough to do both.

---

## 1. Ranked table of verified candidates

Ranked by **external-validity gain per hour of integration work**, not by how real they are.

| # | Dataset | What it really is | Real vs simulated | URL | Licence | Size / format | Channels vs our 11 | Labels | Multi-point structure | Download status |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **MSU / ORNL Power System Attack Datasets** (Morris, Adhikari, Pan, Beaver, Borges, 2014) | 15 repeats × 37 scripted scenarios on a cyber-physical transmission testbed: 2 generators, 2 lines, 4 IED/PMU relays, 4 breakers, real OpenPDC / Snort / syslog stack | **Hardware-in-the-loop.** Power system modelled on an **RTDS real-time digital simulator**; relays, PMUs, PDC, protocol stack and Snort alerts are **real hardware/software**. *Not field data.* | [sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets](https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets) | ⚠️ **No explicit licence** on the host page (citation requests only). DHS **IMPACT** entry DS-1290 records *"Restriction Type: Unrestricted"*, *"Commercial Allowed: true"* | 3 × ~20 MB `.7z` → 15 CSV + 15 ARFF; **78,377 rows × 129 cols** | **11 / 11** (9 direct, `p`/`q` derived — §2.1) | 37 scenarios; 3-class {NoEvents, Natural, Attack}; binary | **4 relays (R1–R4) on 2 lines × 15 independent runs = 60 shards** | ✅ **Verified direct download, no login** (HTTP 200; 19.7 / 19.7 / 20.0 MB) |
| **2** | **LBNL Open µPMU** — raw archive **+** PMU Event Library | Micro-PMU recordings from the **LBNL campus 12.47 kV distribution grid** (real buses, real CTs/PTs) | **REAL field measurements** | archive [powerdata-download.lbl.gov/data/](https://powerdata-download.lbl.gov/data/) (HTTP 200) · project [secpriv.lbl.gov/project/powerdata/](https://secpriv.lbl.gov/project/powerdata/) · labels [github.com/LBNL-ETA/pmu_event_library](https://github.com/LBNL-ETA/pmu_event_library) | Event Library: **BSD-3-Clause-LBNL** — genuinely redistributable. ⚠️ Raw archive: **licence unstated**; README asks citation of Peisert et al., *LBNL Open Power Data* (DOI 10.21990/C21599) | archive ≈ **31 GB** gz CSV (`ts_ns,value`, one file per channel per site); Event Library **25 MB** repo, 65 CSVs | **11 / 11** — `L1/L2/L3 Mag+Ang` (Vrms), `C1/C2/C3 Mag+Ang` (Arms), **P and Q per phase precomputed**; `v0`,`i0` from symmetrical components; `fdev` from angle drift | Event Library ships per-sample booleans `Voltage-Sag [-]`, `Voltage-Swell [-]`. Raw archive **unlabelled** | **3 sites** (`B90`, `FG`, `FL`; archive: `a6_bus1`, `bank_514`, `grizzly_bus1_2`) | ✅ **Verified direct download, no login** (open Apache index; repo tarball 19.9 MB) |
| **3** | **OscGrid — Open Dataset of Real-World Oscillograms from Electric Power Grids** (Evdakov, Filatova, Yablokov et al., *Sci Data* 2026) | 50,765 relay-terminal fault/disturbance oscillograms from operational **0.4–35 kV networks**, anonymised | **REAL field recordings.** *"sourced exclusively from anonymised internal industrial databases and external collaborators"* | data [doi.org/10.6084/m9.figshare.28465427](https://doi.org/10.6084/m9.figshare.28465427) · paper [10.1038/s41597-026-06587-8](https://doi.org/10.1038/s41597-026-06587-8) · code [AIRI-Institute/oscgrid](https://github.com/AIRI-Institute/oscgrid) (MIT) | **CC BY 4.0** on the figshare record (API-verified). ⚠️ the *article* is CC BY-NC-ND 4.0 per Crossref — cite the **data record** | **10.00 GB**, 42 files, 7z → COMTRADE + flat CSV. Labelled subset `labeled.7z` 42.6 MB → 384 MB CSV | **~8 / 11.** `IN` (→ `i0`) only **2.4 % filled**; `fdev` unreliable on 0.4 s records | 31 sparse hierarchical binary `ML_*` columns | 990 series = 480 oscillograms × Bus 1/Bus 2. ⚠️ **No substation ID retained (anonymised)** | ✅ **Verified direct download, no login** (figshare API + `ndownloader`) |
| **4** | **RTE DFRDB — Digital Fault Recording Database** (French TSO) | 12,053 measured fault waveform records from the **French 90 kV transmission grid** | **REAL field recordings.** *"measured voltage and current waveform signals (phase-ground) on high voltage lines of the French electricity transmission grid during various faults"* | [dfrdb.rte-france.com](https://dfrdb.rte-france.com/) · [github.com/rte-france/digital-fault-recording-database](https://github.com/rte-france/digital-fault-recording-database) | **ODbL-1.0** (share-alike on derived databases — *not* CC-BY) | ~2 GB NumPy `.npz`; `DATA_S` = (12053, 6, 21000) | 6 shipped (`v1..3`, `i1..3`) @ **6400 Hz**, 3.28 s/record; rest derivable | ❌ **NONE** | ❌ **NONE** — no substation, line or timestamp metadata | ✅ Verified reachable, no login (both hosts HTTP 200) |
| **5** | **VSB terabyte partial-discharge dataset** (Klein, Krátký et al., VŠB-TU Ostrava) | Contact-galvanic PD monitoring of **23 real MV overhead lines** (22/35 kV, >750 km), Czech Republic | **REAL field measurements** | Harvard Dataverse [doi:10.7910/DVN/JYJJ5W](https://doi.org/10.7910/DVN/JYJJ5W) | **CC BY 4.0** per the Dataverse machine-readable record. ⚠️ the *Sci Data* article text says CC BY-NC-ND 4.0 — **unresolved discrepancy** | **~410 GB** (17 × ~24 GB `.tar.gz`) + `feature_vector.csv` 192.7 MB; 1,353,997 signal files | **~1 / 11.** 3-phase stray-field voltage proxy @ 40 MS/s, 20 ms records. **No currents, no P/Q, no frequency** | 7 PD classes; class 0 = 98.1 % | **23 power lines** — strong natural client split | ✅ Dataverse files public/unrestricted (API-verified). ⚠️ the paper's mirror `db.cs.vsb.cz/datacolls/pdp2020/` is **DEAD (404)** |
| **6** | **Kaggle "VSB Power Line Fault Detection"** | The ~3 % competition subset of #5 | **REAL field measurements** | [kaggle.com/competitions/vsb-power-line-fault-detection](https://www.kaggle.com/competitions/vsb-power-line-fault-detection) (HTTP 200) | ⚠️ **Unverified.** Rules page is JS-rendered and unreadable by automated fetch; Kaggle API returns `401 Unauthenticated` | ~9 GB parquet; 8,712 train + 20,337 test signals | Same as #5 | Binary PD / no-PD | 3 phases per group | ⚠️ **Free Kaggle account + rule acceptance required.** Superseded by #5 |

### Would have been ideal — not open

| Dataset | Why losing it hurts | Status |
|---|---|---|
| **Operational Data Set … Faults in Medium-Voltage Reclosers (ODCEAMPFMVR)** — [IEEE DataPort](https://ieee-dataport.org/documents/operational-data-set-cleaning-and-evaluation-ai-models-predicting-faults-medium-voltage) | **9 SEL reclosers in Puno, Peru**, Jan 2024–Jun 2025: 156,023 load records @15 min + **1,191 labelled 3φ/1φ faults**; channels = phase/line voltages, `IA/IB/IC/IG`, P, Q, PF, frequency, **sequence components**. Almost exactly our 11 channels with **9 natural federated clients**, from real field reclosers. This is the dataset we actually want. | ❌ *"This dataset requires an IEEE DataPort Subscription to access."* No Zenodo/Mendeley mirror found. **Worth emailing the authors** — highest-value outreach on this list. |
| **NI4AI "Sunshine" dataset** | 6 µPMUs (3 substations + PV array + 2 buildings), 12 channels @ 120 fps — a near-perfect 6-client real federated setup. | ☠️ **Platform offline.** `ni4ai.org` and `api.ni4ai.org` both fail to connect (HTTP 000, verified twice). ARPA-E funding ended. **Do not cite as available.** |
| **Real-life Power Quality Sags / Transients** — Univ. of Cádiz, [IEEE DataPort](https://ieee-dataport.org/documents/real-life-power-quality-sags) | Real 20 kHz PQ recordings | ❌ Subscription-only; single site anyway |

---

## 2. Integration plans

### 2.1 MSU / ORNL Power System Attack Datasets — **primary recommendation**

**Verified by parsing the data ourselves:**

* `triple.7z` → `data1.csv … data15.csv`; `multiclass.7z` → the same 15 as ARFF carrying a per-row
  **scenario id** (`marker ∈ {1..41}`, 37 distinct).
* **129 columns** = 4 relays × 29 measurements (116) + 12 control-panel/relay/Snort log columns + `marker`.
* **78,377 rows** total; per file 4,966–5,569 rows covering 36–37 scenarios.
* PMU reporting rate **120 samples/s**; the public release is **randomly sub-sampled at 1 %**, and
  **there is no timestamp column** — the ARFF header (`…Remove-R1-2`) shows two leading attributes
  were stripped before release.
* Hygiene: the only non-numeric cells are in `R*-PA:Z` (~3–4 % `Infinity` when current ≈ 0).
  **All 11 channels we need are clean.**

**Channel mapping — 11 / 11 available.** Per relay `R# ∈ {R1..R4}`:

| Contract channel | Source | Note |
|---|---|---|
| `va, vb, vc` | `R#-PM1:V`, `R#-PM2:V`, `R#-PM3:V` | phase A–C voltage magnitude (V) ÷ `V_base` |
| `ia, ib, ic` | `R#-PM4:I`, `R#-PM5:I`, `R#-PM6:I` | phase A–C current magnitude (A) ÷ `I_base` |
| `v0` | `R#-PM9:V` | zero-sequence voltage magnitude, already computed by the relay |
| `i0` | `R#-PM12:I` | zero-sequence current magnitude |
| `fdev` | `R#:F` − 60 | relay frequency (Hz) |
| `p`, `q` | **derived** from `PM1..3:V`, `PA1..3:VH`, `PM4..6:I`, `PA4..6:IH` | `p = Σ_φ V_φ I_φ cos(θ_Vφ − θ_Iφ)`, `q = Σ_φ V_φ I_φ sin(θ_Vφ − θ_Iφ)` |

We implemented and **sanity-checked the P/Q derivation**: on `NoEvents` rows it gives
**P ≈ 178 MW, Q ≈ 11 MVAr, pf = 0.998**, with `Va ≈ 131 kV` line-to-neutral — a healthy,
near-unity-PF 230 kV bus. The derivation is physically correct.

**Preprocessing steps:**

1. Fetch the three `.7z` archives (60 MB total) via a downloader script; extract with `py7zr`.
2. Read the **multiclass ARFF** (not the CSV) — it carries the scenario id needed for episodes.
3. **Reconstruct episodes.** Rows are grouped by scenario (verified: only 5 label runs per file;
   **553 contiguous scenario blocks** across all 15 files, median 123 rows). Treat each contiguous
   `(file, scenario)` block as one quasi-continuous episode and **never window across a boundary**.
4. Per relay build the 11-channel matrix; per-unit using that relay's own median `NoEvents`
   magnitudes as base; `fdev = F − 60`.
5. Window `T = 64` within episodes. **Measured yield:** stride 16 → **3,044 windows**;
   stride 8 → **5,842**; stride 1 → 45,076. Multiply ×4 for relay-as-sample.
6. Fit the existing per-channel `Normalizer` on each client's train split only (unchanged).

**Client partitioning (K = 10).** Two defensible axes, use both:

* **Vantage point:** 4 relays — R1/R2 on Line 1, R3/R4 on Line 2. A genuine measurement-location
  split, the direct analogue of our per-substation metering point.
* **Independent repetition:** 15 experiment sets, each an independent run of the scenario suite.

Cross them → **60 shards (4 relays × 15 sets)** → group into **K = 10 clients of 6 shards**.
At stride 8 that is **23,368 relay-windows ≈ 2,337 per client**, inside the contract's 1.5–3 k
band. Impose label skew with the same Dirichlet(α) protocol we already use, so the federated
protocol is **identical to the synthetic experiments** — that comparability is the point.

⚠️ **Honest caveat:** every set contains all 37 scenarios, so the natural split is close to IID.
The real heterogeneity comes from relay vantage point; the label skew is imposed, exactly as in the
synthetic setting. Say this plainly; do not claim naturally-occurring non-IID.

**Which of the 8 classes survive.** Measured window counts (stride 16):

| Our class | Source scenarios | Windows |
|---|---|---|
| `NORMAL` | 41 (normal operation, load changes) | **224** |
| `FAULT_LG` | 1–6 (SLG faults at 10–19 / 20–79 / 80–90 % of L1, L2) | **631** |
| — (outside taxonomy) | 13, 14 line maintenance | 108 |
| — (outside taxonomy) | 7–12, 15–30, 35–40 cyber attacks | 2,081 |

**Only 2 of our 8 classes physically occur: `NORMAL` and `FAULT_LG`.** No LL, LLG, 3PH, OVERLOAD,
SAG or SWELL events exist in this testbed. So the real-data section must run either
**(a) the dataset's own 3-class task** {No-event, Natural, Attack} — the standard formulation,
directly comparable to published baselines, **recommended** — or **(b)** a 2-class physical subset
{NORMAL, FAULT_LG}, honest but thin.

⚠️ **The class prior is inverted.** Measured on `data1.csv`: `NoEvents` 173, `Natural` 927,
`Attack` 3,866 — NORMAL ≈ 7 % of windows against our simulated ≥ 90 %. The rare-event regime our
metric design assumes **does not hold here**. Report it as a distinct operating point.

**Effort estimate: 8–14 hours.** Loader + ARFF parse + episode reconstruction ≈ 3 h; channel
mapping and P/Q derivation ≈ 2 h (already prototyped and validated here); partitioner +
`ClientDataset` adapter ≈ 3 h; run the existing FL/SNN pipeline and make figures ≈ 2 h; writing
≈ 2–4 h. **No model changes** — the tensor is already `(11, 64)`.

**What could go wrong:**
* The 1 % sub-sampling makes "consecutive" rows ~0.8 s apart on average and **irregularly spaced**;
  a T=64 window is a ragged ~50 s of wall-clock, not a uniform grid. Our LSTM and the SNN's leak
  dynamics implicitly assume uniform spacing. Mitigation: state it, and report a stride-1 vs
  stride-16 ablation showing the conclusion is not an artefact.
* Absolute per-unit bases are unknown, so we normalise per relay — a leakage risk if the base is
  fitted on anything but the train split. The existing `Normalizer` discipline covers this.
* NORMAL has only 224 windows at stride 16; macro-F1 will be noisy. Use stride 8 and report CIs.
* Expect the objection *"this is still RTDS simulation."* It is correct; concede it (§4).

**Prior art — this dataset is already used for federated learning.**
FedDiSC ([arXiv:2304.03640](https://arxiv.org/abs/2304.03640)); a Location-FedAvg IDS
([ScienceDirect S2590123025045621](https://www.sciencedirect.com/science/article/pii/S2590123025045621));
Chatzimiltis et al. (segmentation + FL). Good — established baselines exist — but we cannot present
federated learning on this dataset as novel.

---

### 2.2 LBNL Open µPMU — **cheapest genuine field-measurement check** (recommended add-on)

**Verified by downloading and parsing the repo ourselves** (`master` branch tarball, 19.9 MB):

* **65 `*_raw_data.csv`** across **30 event folders**, split `No_PV` (57) / `PV` (8).
* **3 sites:** `B90` 30 files (office building), `FL` 18, `FG` 17 (FlexGrid test facility).
* **24 columns**, exactly: ns timestamp index, `C1/C2/C3-Ang [Deg]` + `C1/C2/C3-Mag [Arms]`,
  `L1/L2/L3-Ang [Deg]` + `L1/L2/L3-Mag [Vrms]`, `Active-Power-{1,2,3} [W]`,
  `Reactive-Power-{1,2,3} [VAR]`, `Apparent-Power-{1,2,3} [VA]`,
  **`Voltage-Sag [-]`**, **`Voltage-Swell [-]`**.
* 9,097 samples total; median **135 rows/file** (≈1.1 s at 120 Hz). Timestamp deltas confirm 120 Hz.
* **Label reality check (measured across all 65 files):** **57 / 65 files contain ≥1
  `Voltage-Sag = TRUE`**; **0 / 65 contain any `Voltage-Swell = TRUE`**. Sample level: 482 sag
  samples out of 9,097 (5.3 %), **zero swells**.
* **Window yield: T=64 stride 16 → 346 windows; stride 8 → 649.**

**Channel mapping — 11 / 11.** `va,vb,vc ← L1/L2/L3-Mag`; `ia,ib,ic ← C1/C2/C3-Mag`;
`p,q ← Active/Reactive-Power` (summed over phases — **already computed, no derivation needed**);
`v0,i0 ← ` symmetrical components from the magnitude+angle pairs; `fdev ← ` angle drift
(marginal over ~1 s records — consider dropping it or setting it to 0 with a documented note).

**Client partitioning:** only **3 real sites**. Options: use K = 3 and say so, or split each site
by time block / event group into ~9–10 pseudo-clients and **state explicitly that these are
pseudo-clients**, not independent substations. Do not dress 3 sites up as 10.

**Which classes survive: effectively one — `SAG`** (plus `NORMAL` from the pre/post-event padding
and from the raw archive). `SWELL` is labelled in the schema but **never fires**. No faults, no
overload. The honest task is **binary NORMAL vs SAG on real distribution measurements**.

**Scaling up NORMAL:** the raw archive at [powerdata-download.lbl.gov/data/](https://powerdata-download.lbl.gov/data/)
(open Apache index, `Accept-Ranges: bytes`, no auth) has 3 sites × 13 streams
(`L1/L2/L3 MAG|ANG`, `C1/C2/C3 MAG|ANG`, `LSTATE`) at 120 Hz, plain `ts_ns,value` gzip CSV,
≈31 GB total, ~12 days of continuous coverage from 2015-10-01. That supplies effectively unlimited
NORMAL windows — but it is **unlabelled** and its **licence is unstated** (README asks citation of
Peisert et al., DOI 10.21990/C21599). Only the Event Library carries a real licence
(**BSD-3-Clause-LBNL**), so **ship derived data only from the Event Library**; for the raw archive,
ship a downloader.

**Effort estimate: 4–6 hours** for the Event Library alone (25 MB, one flat CSV schema, P/Q already
present, labels already present). **+6–10 hours** if we also ingest the 31 GB raw archive for
NORMAL windows and range-request handling.

**What could go wrong:** 346 windows is small enough that a single seed proves nothing — report
multi-seed CIs, and do not quote a headline accuracy. Three sites is a weak federation. `fdev` over
a 1 s record is close to meaningless. And the raw archive's unstated licence must not leak into our
released artifact.

---

### 2.3 OscGrid — real field oscillograms (do not attempt for this paper)

**Verified by downloading and parsing `labeled.7z`:**

* `labeled.csv`: **384 MB, 51 columns, 1,554,809 rows, 990 measurement series** from 480 distinct
  oscillograms (`file_name = <md5>_Bus N`).
* Series length: min 468, **median 641**, max 48,800. Labelled subset is 1,600 Hz on a 50 Hz grid
  → **32 samples/cycle**, so the median record is only **~20 cycles ≈ 0.4 s**.
* **Channel fill rates (measured):** `IA` 97.8 %, `IC` 97.8 %, `IB` **58.1 %**,
  `UA/UB/UC BB` 79.7 %, `UN BB` 64.1 %, **`IN` 2.4 %**.
* **Records with all 6 core channels: 297 / 990.** Records *also* carrying `IN` **and** `UN BB`: **1**.
* Amplitudes normalised to secondary values (V ÷ 3×100 V, I ÷ 25×5 A); per-unit is supported via
  the shipped `norm_coef` table, but absolute physical scale is deliberately removed.
* Labels are 31 sparse binary `ML_*` columns. **`ML_3_3_1` (single-phase earth fault) is not present
  in this release's `labeled.csv`** — only `ML_3_3`, `ML_3_3_2`, `ML_3_3_3` exist. Among the 297
  full-channel records: `ML_3` (fault) 52, `ML_1` 37, `ML_2_3`/`ML_2_3_1` (voltage drawdown,
  SAG-like) 37, `ML_2_4_1` 17 — but **`ML_3_3_2` (LL) in 2 records and `ML_3_3_3` (3PH) in 2**.

**The blocking problem.** A `T = 64` window of *per-cycle RMS phasors* — the quantity our contract
defines — needs ≥ 2,048 raw samples at 1,600 Hz. **Only 48 of the 297 full-channel records are that
long.** Forty-eight windows is not an experiment.

The workaround is to redefine the timestep as a raw sample or sliding sub-cycle RMS
(T=64 samples = 40 ms), yielding ~11 k windows — numerically fine, but the model is then
classifying *waveform shape*, not RMS-trend telemetry. That is **a different task**, and the paper
must not blur the two.

**Client partitioning:** anonymised — **no substation identifier is retained**. The most defensible
~10-client split is by **sampling-rate bucket** (19 distinct rates ≈ recorder/vendor families,
a real device-driven distribution shift), optionally crossed with 50/60 Hz. Splitting by oscillogram
hash is possible but is not a substation split and must not be described as one.

**Effort estimate: 25–40 hours** — COMTRADE ingestion, per-record channel-availability handling,
RMS/phasor extraction, a second windowing semantics, remapping a 31-node label hierarchy, and a
separate results narrative. Effectively a second paper section with its own pipeline.

---

### 2.4 RTE DFRDB — best-in-class realism, unusable labels

12,053 genuinely measured fault records from the French 90 kV grid, 6 channels @ 6,400 Hz,
ODbL-1.0, ~2 GB, no login. `v0, i0, p, q, fdev` all derivable from `v1..3` / `i1..3`.

**Two hard blockers.** (1) **No labels ship with it** — the papers that use it annotated their own
subsets and did not release the annotations. (2) **No substation, line or timestamp metadata**, so
there is no defensible federated partition; clients would have to be synthesised by clustering
signal statistics, which a reviewer will correctly call arbitrary.

**Verdict:** excellent as an *unlabelled* transfer/pre-training corpus or a qualitative
"does our encoder transfer to real waveforms" probe. Not a supervised benchmark without a labelling
campaign. **Effort to make it usable: 40+ hours**, most of it manual annotation.

---

## 3. Not usable, and why — do not revisit

**Simulated, despite being widely cited as "fault datasets":**

| Dataset | Verdict |
|---|---|
| **PSML** ([tamu-engineering-research/Open-source-power-dataset](https://github.com/tamu-engineering-research/Open-source-power-dataset)) | PMU streams are **T+D co-simulation output**; only the load/weather forcing is real. |
| **Power System Multi-source Events Dataset** ([IEEE DataPort](https://ieee-dataport.org/documents/power-system-multi-source-events-dataset)) | Superficially a perfect fit (IEEE 34-node, 120 Hz, 9 channels, 34 nodes) — **it is EMT simulation**. |
| **Transients and Faults in Power Transformers and Phase Angle Regulators** ([IEEE DataPort](https://ieee-dataport.org/open-access/transients-and-faults-power-transformers-and-phase-angle-regulators-dataset)) | Open Access, 88k+ files, *"simulated in PSCAD/EMTDC."* |
| **PMU measurements of IEEE 39-bus power system model** ([IEEE DataPort](https://ieee-dataport.org/open-access/pmu-measurements-ieee-39-bus-power-system-model)) | Open Access but **RTDS simulation of a model**. |
| **Realistic Labelled PMU Data … Real-Time Synchrophasor Testbed** ([IEEE DataPort](https://ieee-dataport.org/documents/realistic-labelled-pmu-data-cyber-power-anomaly-detection-using-real-time-synchrophasor)) | The SGSMA-2024 competition set. *"Hardware-in-the-Loop (HIL) … Real-Time Digital Simulator (RTDS)."* HIL, **not field data**, and subscription-only. Flag explicitly — it is often miscited as "real". |
| **Synthetic PQ Disturbances** family (several IEEE DataPort records) | Synthetic by construction (IEEE Std 1159 generators). |
| **Zenodo, systematically** | Four API sweeps (~100 records: power quality, synchrophasor, COMTRADE, power-system fault). Essentially everything fault-related is simulation — PROTECT-90 ([21109169](https://zenodo.org/records/21109169), 9,022 EMT sims), IEEE 39-bus transient-stability sets ([4521886](https://zenodo.org/records/4521886), [7350829](https://zenodo.org/records/7350829)), series-compensated line faults ([5643563](https://zenodo.org/records/5643563)), "Transmission Line Fault Data via PMU" ([8214226](https://zenodo.org/records/8214226) — CC0 but MATLAB/Simulink). **Zenodo has no real, labelled, multi-site power-fault dataset.** Note: `COMTRADE` as a Zenodo keyword returns UN *Comtrade trade statistics*. |

**Real but not openly downloadable:**

| Dataset | Verdict |
|---|---|
| **FNET / GridEye** (UTK/ORNL) | Real (300+ FDRs, frequency/voltage magnitude/angle @100 ms). [fnetpublic.utk.edu](https://fnetpublic.utk.edu/) is **live but is a visualisation only** — no download control, no archive link, no data-request form anywhere on the site (independently confirmed). The literature's operative sentence: *"historical data is accessible with further privilege authorization."* Access is via the FNET/GridEye Industry Consortium, which grants **event alert e-mails and an event website — not bulk time series**. **Not reproducible; drop it.** |
| **NGED LV Load Monitor Data** ([portal](https://connecteddata.nationalgrid.co.uk/dataset/lv-load-monitor-data)) | 1,431 real LV substations with per-phase P/Q — would have been a superb client partition. CKAN API returns `"isopen": false`, `"license_id": ""`, `"dataclassification": "Shared"`, and **every resource URL is literally `redacted`** to unauthenticated callers. **Not open.** |
| **IEEE DataPort subscription tier** | ODCEAMPFMVR (Peru reclosers), Real-life PQ Sags, Real-life PQ Transients, trains_PQ_by_GW, HF-CT-Fault — all real, all *"requires an IEEE DataPort Subscription."* |
| **Kaggle VSB competition** | Free account **and** rule acceptance required; rules text is JS-rendered and could not be read, so redistribution terms are **unverified**. Superseded by the Harvard Dataverse release. |

**Open and well-licensed, but the wrong shape — frequency-only or load-only:**

| Source | What it actually is | Why it fails |
|---|---|---|
| **Open Power System Data** ([open-power-system-data.org](https://open-power-system-data.org/)) | Load, day-ahead prices, wind/solar generation & capacity, 15/30/60-min, 32 countries, from ENTSO-E | **No frequency, no voltage, no current — 0 of our 11 channels.** Market/energy-balance data, not electrical measurement. Core packages **frozen since 2020-10**. Licence caveat: *"some data published here might be subject to copyright."* |
| **NESO (GB) system frequency** — [data portal](https://www.neso.energy/data-portal/system-frequency-data), CKAN API | 1 s frequency, Jan 2014 → Jun 2026, header `dtm,f`. **NESO Open Data Licence** (commercial OK). Genuinely open and excellent data | **1 channel (`fdev`), 1 site.** No labels. |
| **Fingrid (FI)** — [dataset 339](https://data.fingrid.fi/en/datasets/339) | **10 Hz** frequency, bulk `.7z` per month, no key needed. **CC-BY-4.0** | **1 channel, 1 site.** No labels. |
| **Elia (BE)** — `ods057` via [opendata.elia.be](https://opendata.elia.be/) | 10 s frequency + FCR demand, 3.4 M records | **1 channel, 1 site.** Licence page could not be read (portal rate-limiting). |
| **Energinet (DK)** ([energidataservice.dk](https://www.energidataservice.dk/)) | CO₂, market, consumption/production | **No frequency, no V/I.** |
| **ENTSO-E Transparency Platform** | Load/generation/transmission/outages/balancing per Reg. 543/2013 | **Aggregate, not electrical measurement.** Site fetch failed; API needs free registration. |
| **Power-Grid Frequency Database** (Rydin Gorjão et al., *Nat. Commun.* 11:6362) — [osf.io/by5hu](https://osf.io/by5hu/), [osf.io/m43tg](https://osf.io/m43tg/), index at [power-grid-frequency.org/database](https://power-grid-frequency.org/database/) | **Verified open**, both OSF nodes `public: true`, **CC-BY-4.0**, direct download no login. 31 site folders / 40+ recording locations | **Frequency only → 1 of 11 channels, and zero event labels.** Verified the hard way: even the Nordic *PMU* sub-dataset has been post-processed down to a single `Frequency` column — V/I phasors were stripped before publication. The `WaveformData/` folder is **single-phase voltage only**, 25 kHz, one hour per site, mostly recorded *"from conventional power sockets in office or hotel environments"* — no current, so `v0` is not even constructible. ⚠️ Also **per-dataset licences differ from the node licence** — notably **RTE France: redistribution "strictly forbidden."** |
| **Electrical Measurement Dataset from a University Laboratory** ([Zenodo 17107425](https://zenodo.org/records/17107425), CC-BY-4.0) | Real PCC measurement with 3-phase V, I, P, Q per phase, PF, frequency — our exact channel set | **10-minute sampling.** A T=64 window spans 10.7 hours — physically incapable of representing a fault, sag or swell. **1 site**, no labels. |

**Too weak to matter (real, open, wrong shape):**
Mesa Del Sol microgrid ([Zenodo 8339403](https://zenodo.org/records/8339403), CC0 — real, **no event labels**);
transformer inrush lab measurements ([Zenodo 7712948](https://zenodo.org/records/7712948), CC-BY-4.0 — **~945 GB** for one phenomenon);
MV/LV feeders Jabodetabek ([Zenodo 7954007](https://zenodo.org/records/7954007), CC-BY-4.0 — 7 days, no fault labels);
IEEE DataPort LFO/short-circuit PMU records ([open access](https://ieee-dataport.org/open-access/phasor-measurement-data-recorded-during-low-frequency-oscillation-and-short-circuit)) — real, free IEEE login, but **two `.mat` files totalling 13.7 MB**, no labels, no documented channels.

**Dead links / stale pointers:**

| Item | Verdict |
|---|---|
| `https://db.cs.vsb.cz/datacolls/pdp2020/` | Alternate mirror named in the VSB *Sci Data* paper. **404 / TLS failure.** Use the Harvard Dataverse DOI. |
| `ni4ai.org`, `api.ni4ai.org` | **Offline (HTTP 000).** Only `blog.ni4ai.org` survives. |
| [dasi.msstate.edu/projectAnalyticsDatasets.php](https://www.dasi.msstate.edu/projectAnalyticsDatasets.php) | No datasets listed; DASI merged into the Center for Cyber Innovation. Live distribution point is the UAH page in §1. |
| IEEE DataPort "Empowering a Proactive Grid with Power Quality Visibility" | The "dataset" is **9 PNG images**. Also subscription-only. |

---

## 4. Recommendation

> **Add the MSU / ORNL Power System Attack Dataset as the external-validity section — 8–14 hours.
> If budget allows, spend 4–6 more hours on the LBNL PMU Event Library, which is the cheapest
> genuine *field*-measurement data point available anywhere in this survey.**

**Why MSU/ORNL wins on gain-per-hour.** It is the only verified candidate where the tensor our
pipeline already consumes — `(11, 64)` — can be built **without dropping a single channel**. Nine of
`va,vb,vc,ia,ib,ic,v0,i0,fdev` are shipped directly by the relays and `p,q` follow exactly from the
phase magnitudes and angles (derivation validated: pf 0.998 at a 230 kV bus). It has a real
multi-point structure (4 relays × 15 runs → K=10 at ~2.3 k windows/client), downloads in 60 MB with
no login, and federated baselines already exist on it. Nothing else is close: OscGrid yields
48 usable windows under our contract's semantics, RTE DFRDB has no labels at all, and every open
TSO source is a single frequency scalar.

**Why LBNL is the right second, if there is a second.** MSU does not answer *"is any of this real?"*
— its physics is RTDS. LBNL Event Library is **actual field data from a real 12.47 kV distribution
grid**, ships all 11 channels (P and Q already computed), is **BSD-3-Clause-LBNL so we may
redistribute derived data**, and is a 25 MB download. Its weakness is scale — we measured
**346 windows at T=64/stride 16, 3 sites, and exactly one event class (57/65 files carry a
`Voltage-Sag`; zero files carry a `Voltage-Swell`)**. Presented honestly as a small binary
NORMAL-vs-SAG probe on real measurements, it is worth far more per hour than any attempt to force
OscGrid or DFRDB into the contract.

**What the paper could then honestly claim:**

* That the **full FedGrid stack** — windowing, non-IID federation, CNN-LSTM **and** the spiking
  classifier, fidelity scoring, robust aggregation, PBFT ledger — runs **unmodified** on
  measurements produced by **real protection relays, real PMUs, a real PDC and a real substation
  protocol stack**, not only on our OpenDSS output.
* That the contract's 11-channel representation is **not an artefact of our simulator**: an
  independent hardware testbed, built by a different group a decade earlier for a different purpose,
  emits exactly these quantities — and (with LBNL) so does a real distribution µPMU deployment.
* That our federated and Byzantine-robustness results **reproduce on a third-party benchmark with
  published baselines** (FedDiSC and successors), so the numbers are comparable outside our code.
* With LBNL: that the pipeline also ingests **genuine field measurements** end-to-end, at a small
  but real scale.

**What it would NOT prove — put this in the paper; do not let a reviewer find it:**

* **MSU is not field data.** The power system is modelled on an **RTDS real-time digital simulator**.
  What is real is the *measurement and communication chain*, not the electrical physics. It moves us
  from "one simulator" to "our simulator **plus** an independent hardware-in-the-loop testbed" — a
  real reduction in single-tool risk, **not** validation against grid measurements.
* **It does not validate our 8-class taxonomy.** Only `NORMAL` and `FAULT_LG` physically occur in
  MSU; LBNL contributes only `SAG`. Across *every* dataset in this survey, **no open real source
  provides labelled LL / LLG / 3PH faults at multiple sites.** Our 8-class taxonomy remains
  simulation-only, and the paper must say so.
* **It does not validate the rare-event regime.** NORMAL is ~7 % of MSU windows versus ≥ 90 % in our
  design; the class-prior story is a separate operating point, not a confirmation.
* **It does not validate the 60 s telemetry cadence.** MSU is 120 Hz sub-sampled at 1 % with no
  timestamps; LBNL is 120 Hz over ~1 s records. Different, irregular timescales.
* **It does not validate the distribution-feeder setting.** MSU is a 2-line, 230 kV *transmission*
  testbed; our simulation is distribution (IEEE 13/34/37/123 + EU LV). LBNL *is* distribution, which
  is another reason to include it.
* **The federation is partly constructed.** MSU's 10 clients come from 4 relays × 15 runs with
  imposed Dirichlet skew; LBNL has only 3 real sites. Neither is 10 independent substations.

**Suggested framing** — one honest sentence beats a paragraph of hedging:

> *"To test that our pipeline is not tied to a single simulator, we additionally evaluate it on the
> Mississippi State/ORNL power-system testbed dataset, in which a transmission system modelled on a
> real-time digital simulator is instrumented with commercial protection relays, PMUs and a phasor
> data concentrator. All eleven of our channels are recovered from the relay measurements without
> any modification to the model. This exercises the full federated stack on hardware-derived
> measurements and against published baselines; it does not constitute validation against field
> recordings, and the event taxonomy there covers only normal operation and single-line-to-ground
> faults."*

**Highest-value outreach.** The one dataset that would genuinely close the gap — real field
measurements, ~11 channels, labelled 1φ/3φ faults, 9 natural clients — is the **Puno recloser
dataset (ODCEAMPFMVR)**, which is IEEE DataPort subscription-only. **Emailing the authors for a
research copy is a cheap, high-upside action** and should be started now in parallel, since it has a
long lead time.
