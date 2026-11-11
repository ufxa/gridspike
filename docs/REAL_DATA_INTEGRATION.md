# REAL_DATA_INTEGRATION.md — external-validity section on real measurements

**Purpose.** Every number in the main results comes from OpenDSS quasi-static simulation
(`docs/REALISM.md`); the most damaging reviewer objection is *"all of this is synthetic."* This
document records how we integrate a **real measurement dataset** — the Mississippi State University /
Oak Ridge National Laboratory *Power System Attack Datasets* — and run the **same three classifiers**
(CNN-LSTM, spiking/SNN, SSM) under the **same federated protocol**, unmodified, so the paper can show
the method transfers to real relay/PMU measurements. It is the concrete follow-through on the survey
in `docs/REAL_DATA.md` (§2.1, the #1-ranked candidate).

Everything below is **measured by this repo** on the server, not copied from the source papers. The
loader is `src/fedgrid/data_real.py`; the run set is `experiments/manifest_real.py`; results live in
`runs_real/` (separate from the synthetic `runs/`).

---

## 1. What the dataset actually is — state it honestly

The MSU/ORNL dataset (Morris, Adhikari, Pan, Beaver, Borges, 2014) is a 2-generator, 2-line,
**4-relay transmission testbed** running 37 scripted scenarios repeated across 15 experiment sets.
Downloaded (no login, 3 × ~20 MB `.7z`) from
`http://www.ece.uah.edu/~thm0009/icsdatasets/{multiclass,triple,binaryAllNaturalPlusNormalVsAttacks}.7z`
(README PDF alongside). We use the **multiclass ARFF** set (it carries the per-row scenario id
`marker` needed to reconstruct episodes and labels).

> **It is hardware-in-the-loop, NOT field data.** The electrical physics is modelled on an **RTDS
> real-time digital simulator**. What is *real* is the **measurement and communication chain** —
> commercial protection relays, PMUs, an OpenPDC phasor data concentrator, a Snort IDS, and the
> substation protocol stack. Using it moves the paper from "one simulator" to "our simulator **plus**
> an independent hardware-in-the-loop testbed, built by a different group a decade earlier for a
> different purpose." That is a genuine reduction in single-tool risk — **not** validation against
> grid recordings. We say this plainly in the manuscript; a reviewer must not have to find it.

**Verified structure (parsed by us):** 15 ARFF files, 129 columns, 78,377 rows total; `marker` is the
last column with 37 distinct values; rows are contiguous per scenario — **553 episode blocks across
the 15 files** (37 per file), median 136 rows/block. The public release is a **1 % random sub-sample
with no timestamp column**, so a window is a ragged ~tens-of-seconds snapshot, not a uniform grid
(documented caveat, §8).

---

## 2. Channel mapping — all 11 recovered per relay (9 direct + 2 derived)

The data contract fixes `x: float32 (C=11, T=64)`, channel order
`va, vb, vc, ia, ib, ic, p, q, v0, i0, fdev`. For each relay `R# ∈ {R1..R4}` (PMU magnitude `PM*` and
phase-angle `PA*` columns):

| Contract channel | Source column(s)                       | Direct/derived |
|------------------|----------------------------------------|----------------|
| `va, vb, vc`     | `R#-PM1:V`, `R#-PM2:V`, `R#-PM3:V`      | direct (V)     |
| `ia, ib, ic`     | `R#-PM4:I`, `R#-PM5:I`, `R#-PM6:I`      | direct (A)     |
| `v0`             | `R#-PM9:V`                             | direct (zero-seq V) |
| `i0`             | `R#-PM12:I`                            | direct (zero-seq I) |
| `fdev`           | `R#:F` − 60                            | direct (Hz)    |
| `p`, `q`         | `PM1..3:V`,`PA1..3:VH`,`PM4..6:I`,`PA4..6:IH` | **derived** |

Derivation: `p = Σ_φ V_φ I_φ cos(θ_Vφ − θ_Iφ)`, `q = Σ_φ V_φ I_φ sin(θ_Vφ − θ_Iφ)` (phase angles in
degrees). **No channel is unavailable.** The unused columns (`PA:Z`/`PA:ZH` impedance, `S`, `DF`, the
log columns) are ignored; the `Infinity` cells that appear in `PA:Z` when current ≈ 0 never touch our
channels — **0 non-finite values across all used channels** (verified over one full file: 0 / 297,960).

**Physics validation (measured on `data1`, relay R1):**

| | NORMAL (marker 41) | FAULT_LG (markers 1–6) |
|---|---|---|
| Va, Vb, Vc | 131.2 / 131.2 / 131.3 kV (balanced) | 130.8 / 131.0 / 131.4 kV |
| Ia, Ib, Ic | 439 / 442 / 439 A (balanced) | 421 / 408 / 403 A |
| `i0` (zero-seq I) | mean 1.4, median 0.0 | **mean 11.5, median 4.0** |
| `v0` (zero-seq V) | 0.0 | **86.2** |
| derived P / Q / pf | 172.7 MW / 10.9 MVAr / **0.998** | 157.4 MW / 10.0 MVAr / 0.998 |

The derived P/Q give a healthy near-unity-pf 230 kV bus, matching the survey. The fault rows carry the
textbook **single-line-to-ground signature** — a rise in zero-sequence current *and* voltage — which
independently confirms that markers 1–6 are genuine ground faults (not merely a label we trust).

---

## 3. Label mapping — only the classes that physically occur

The testbed's 37 scenarios are (per the README): 1 **No-event**, 8 **Natural** (6 short-circuit faults
+ 2 line-maintenance), 28 **cyber-attack**. Our fixed 8-class taxonomy (`fedgrid.data.EventClass`)
has **no cyber-attack and no maintenance class**, so the honest physical subset is **binary**:

* `NORMAL` (`EventClass.NORMAL=0`) ← marker **41**
* `FAULT_LG` (`EventClass.FAULT_LG=1`) ← markers **1–6** (single-line-to-ground faults)
* **dropped** ← maintenance (13, 14) and every attack scenario (7–12, 15–30, 35–40): neither NORMAL
  nor a taxonomy fault, so they are removed rather than mislabelled.

**Only 2 of our 8 classes survive.** The models keep `n_classes=8` unchanged (the engine infers the
class count from the model head, and macro-F1 averages over the classes *present in `y_true`*), so the
**same** three classifiers run with **no architectural change** and the metric is defined exactly as in
the synthetic tables. This does **not** validate the LL / LLG / 3PH / OVERLOAD / SAG / SWELL classes —
no open real source provides those labelled at multiple sites (see `docs/REAL_DATA.md` §4). The
class prior here is **fault-heavy** (NORMAL ≈ 26.5 %), the opposite of our synthetic ≥ 90 % NORMAL — a
distinct operating point, reported as such.

---

## 4. Windowing and preprocessing

* **Episodes.** Each contiguous `(file, marker)` block is one quasi-continuous episode; `T=64` windows
  are cut with **stride 8** and **never cross an episode boundary**. A window inherits its block's
  single label (no mixed-label windows).
* **Yield (measured, binary subset, stride 8):** **6,628 windows** — NORMAL 1,756 / FAULT_LG 4,872 —
  across **60 shards** (4 relays × 15 files), median 110 windows/shard. (Stride 16 → 3,420 windows, i.e.
  855 window-positions × 4 relays, matching `docs/REAL_DATA.md`'s per-relay counts of 224 NORMAL + 631
  FAULT_LG; a stride ablation is available via the loader's `stride` argument.)
* **Per-unit / normalisation.** Absolute per-unit bases are **not shipped** with the release, so rather
  than divide by a fitted base (which would leak statistics across splits) we feed **physical SI units**
  and let the contract's per-channel `Normalizer` — fit on each client's **train split only** — 
  standardise every channel. This is the same leak-free discipline the synthetic path uses (verified
  in `tests/test_data_real.py::test_normalizer_is_train_only`).
* **Splits.** Per-client 70/15/15 train/val/test, seeded (unchanged `ClientDataset`). Global test =
  union of client tests.

---

## 5. Client partition — relay vantage point is the real heterogeneity

The natural unit is the **`(relay, file)` shard** (4 relays × 15 files = 60 shards). The loader offers:

* **`partition="relay"` → K = 4** (one client per relay): the *honest* measurement-location split, the
  direct analogue of our per-substation metering point. Pairs with a proportionate committee (m = 4,
  f = 1).
* **`partition="shard"` → K = 10** (default, used for the battery): the 60 shards, ordered by
  (relay, file), split into 10 contiguous groups of 6. This keeps the federated protocol — **K = 10,
  PBFT committee m = 10, Byzantine budget f = 3, τ = 0.5** — **identical to the synthetic experiments**;
  that comparability is the point. Each client is dominated by one relay (clients 0–1 ≈ R1, 3–4 ≈ R2,
  5–6 ≈ R3, 8–9 ≈ R4; 2 and 7 straddle a relay boundary).

> **Honest caveat (stated in the manuscript).** Every experiment file contains all 37 scenarios, so the
> split is close to IID in labels. The real heterogeneity is the **relay vantage point (feature skew)**;
> we do **not** impose the synthetic Dirichlet label skew (only two classes exist). Neither K=10 nor K=4
> is ten independent substations — the K=10 grouping is partly constructed. We do not dress this up as
> naturally-occurring non-IID.

---

## 6. The run set — a compact, honest external-validity battery

`experiments/manifest_real.py` emits the **clean reference-frame** only (this is a validation section,
not the 500-run synthetic battery): the six reference arms × the three classifiers × 3 seeds =
**54 runs**, all at K = 10 shard partition, 30 FL rounds, `optimizer=adam`,
`class_weighting=effective_num` — **byte-identical training wiring to the synthetic frame** (reused
`make_config`, only the `federation` block re-points to `source="msu"`). Arms:

`centralized` (pooled ceiling) · `local` (per-client floor) · `fedavg` · `fedprox` ·
`pipeline_krum` (fidelity → PBFT → Krum) · `pipeline_geomed` (… → geometric median).

Run on the server **CPU only** (10 workers, 2 threads each) so the GPU synthetic battery was never
touched. Reproduce with:

```bash
python -m experiments.manifest_real --out experiments/manifest_real.jsonl --seeds 3 --partition shard
# then, per line:  python -m experiments.run_one --manifest-line '<json>' \
#   --data-dir <extracted-msu-dir> --out-dir runs_real --device cpu
```

---

## 7. Results — measured macro-F1 on real relay measurements

<!-- RESULTS_TABLE -->

---

## 8. What the paper can honestly claim — and what it cannot

**Can claim:**

* The **full FedGrid stack** — windowing, non-IID federation, CNN-LSTM **and** the spiking classifier
  **and** the SSM, fidelity scoring, PBFT consensus, robust aggregation — runs **unmodified** on
  measurements produced by **real protection relays, real PMUs and a real phasor data concentrator**,
  recovering all eleven contract channels with **no change to any model**.
* The contract's 11-channel representation is **not an artefact of our simulator**: an independent
  hardware testbed emits exactly these quantities.
* The federated protocol (K, committee, Byzantine budget) is the **same** as the synthetic experiments,
  so the two are directly comparable.

**Cannot claim (put this in the paper):**

* **Not field data.** The physics is RTDS; only the instrumentation is real. This is "simulator + an
  independent HIL testbed," not "validated on real substations."
* **Does not validate the 8-class taxonomy.** Only NORMAL and FAULT_LG occur; LL/LLG/3PH/OVERLOAD/
  SAG/SWELL remain simulation-only.
* **Does not validate the rare-event regime.** The real prior is fault-heavy (~26 % NORMAL), not the
  synthetic ≥ 90 % NORMAL.
* **Does not validate the 60 s telemetry cadence.** MSU is 120 Hz sub-sampled at 1 % with no
  timestamps — an irregular, different timescale.
* **Transmission, not distribution.** MSU is a 2-line 230 kV transmission testbed; our simulation is
  distribution feeders.
* **The federation is partly constructed** (K=10 grouping + relay-vantage heterogeneity, no imposed
  label skew).

**Suggested one-sentence framing:** *"To test that our pipeline is not tied to a single simulator, we
additionally evaluate it on the Mississippi State/ORNL power-system testbed, in which a transmission
system modelled on a real-time digital simulator is instrumented with commercial protection relays,
PMUs and a phasor data concentrator; all eleven of our channels are recovered from the relay
measurements without any change to the models. This exercises the full federated stack on
hardware-derived measurements; it is not validation against field recordings, and the taxonomy there
covers only normal operation and single-line-to-ground faults."*

---

## 9. Provenance

* Loader `src/fedgrid/data_real.py`; battery generator `experiments/manifest_real.py`; wiring in
  `experiments/build.py` (`federation.source=="msu"` switch); tests `tests/test_data_real.py`
  (tiny synthetic ARFF fixture — no download needed).
* Data downloaded + parsed + all runs executed on the server (WSL2, `/root/fedgrid`); **zero training on
  the local machine**. sha256 of the archives and the parsed-window digest are recorded in the run log.
