# Validation Standard — phase 1 (2026-07-28)

Scope: (§1) the *formal* bar of the target journal — Engineering Applications of
Artificial Intelligence (EAAI, Elsevier, ISSN 0952-1976); (§2) the *de-facto empirical*
bar measured on a sample of 14 EAAI papers 2023–2026 close to our topic; (§3) our
protocol set at **double** that bar, as a binding checklist for the phase 4 battery;
(§4) exact CONTRACT.md amendment text; (§5) verifiable sources.

Method and honesty notes. ScienceDirect blocks robotic access (HTTP 403 on every
direct fetch, including open-access articles). Evidence tiers used:

- **Tier F (full text)** — 6 papers read in full from legitimate copies: 1 publisher
  accepted-manuscript PDF (University of Glasgow eprints), 1 published OA PDF was not
  needed, and 4 arXiv author versions + 1 arXiv version of a published OA paper.
  arXiv versions may differ in minor ways from the journal versions; DOIs and arXiv
  IDs are both listed so every claim is checkable.
- **Tier A (abstract/metadata only)** — 8 closed-access papers with no open copy
  anywhere (verified via OpenAlex locations, Crossref, Semantic Scholar, Unpaywall
  data as exposed by OpenAlex). For these, cells that require full text are marked
  **n.r.** (not retrievable) rather than guessed.
- Guide-for-Authors content was captured from the **live ScienceDirect page on
  2026-07-28 via the r.jina.ai reader proxy** (not an archive snapshot); the Elsevier
  generative-AI policy page was fetched directly the same day.
- Citation counts are OpenAlex values retrieved 2026-07-28 (journal source id
  S900972176).

---

## §1 Formal requirements (EAAI Guide for Authors + Elsevier policies)

Source: EAAI Guide for Authors, sciencedirect.com/journal/engineering-applications-of-artificial-intelligence/publish/guide-for-authors (live page, fetched 2026-07-28 via reader proxy); Elsevier generative-AI policy page (fetched 2026-07-28).

| Item | Requirement (quoted or paraphrased from the Guide) |
|---|---|
| Review model | **Double-anonymized** peer review: separate title page with author info; anonymized manuscript body (no names/affiliations/self-identifying text). Consequence for us: the released artifact must have an **anonymized link during review** (e.g., Anonymous GitHub / anonymized Zenodo), de-anonymized on acceptance. |
| Length | "All submissions must not exceed **50 pages**"; manuscript file **< 100 MB**. |
| Abstract | "Concise and factual abstract which does not exceed **250 words**"; no references or non-standard abbreviations in it. |
| Highlights | **3 to 5 bullet points, max 85 characters each including spaces** (submission checklist item; Elsevier lists highlights as strongly encouraged — we treat them as mandatory). |
| Title | "Concise and informative. Please avoid abbreviations and formulae" unless established and widely understood ("IEC 61850" qualifies as established; do not put model acronyms in the title). |
| Keywords | 1–6 keywords. |
| References | Any style at submission "as long as the style is consistent"; must contain authors, titles, year, volume; journal restyles at proof. |
| Research data | Elsevier research-data policy applies: authors "are required to **state the availability of any data** at submission" (data availability statement is a mandatory submission element); depositing in a repository and **citing/linking the dataset in the article** is the expected route. |
| Code/software | Sharing is encouraged and interlinked (optional *Software Impacts* co-submission exists for open-source software). No hard mandate to release code — releasing it exceeds the bar. |
| Declarations | Required at end of manuscript: (i) **Declaration of competing interests** ("All authors must disclose any financial and personal relationships…"), (ii) **funding sources**, (iii) **CRediT authorship contribution statement** (14 roles; corresponding author responsibility), (iv) generative-AI declaration (below). |
| Generative AI in writing | Elsevier policy (exact template, to be placed in a separate section titled "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process", immediately before the references): *"During the preparation of this work, the author(s) used [NAME OF TOOL / SERVICE] in order to [REASON]. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the published article."* AI must not be listed as author; AI-generated primary research images are prohibited; basic grammar/spell checking needs no disclosure. AI use **in the research methodology itself** is disclosed in Methods, not in this statement. |
| Experimental rigor (explicit journal policy) | The Guide contains an explicit rigor clause (written for metaheuristics but it is the journal's stated benchmark ethos): papers proposing "novel" methods are rejected unless they "present a **fair comparison with other state-of-the-art methods using state-of-the-art practices for benchmarking algorithms**". No further EAAI-specific rigor editorial/checklist was found beyond this clause (searched 2026-07-28); the de-facto bar in §2 is therefore the operative standard. |

---

## §2 The de-facto empirical standard (what published EAAI papers actually do)

Sample: 14 EAAI papers 2023–2026 adjacent to our topic (federated learning; smart
grid / power system protection; DL/SNN fault diagnosis; edge AI; IDS). Includes
top-cited in-topic papers (50, 38, 37 cites) and very recent items (2025–2026).

Column key — (a) independent runs/seeds & mean±std; (b) significance tests & effect
sizes; (c) # comparison methods & centralized/upper-bound reference; (d) ablations;
(e) metrics; (f) hyperparameter discipline; (g) sensitivity analyses; (h) dataset
scale & CV scheme; (i) code/data released. Tier F = full text read; Tier A =
abstract-level only, unverifiable cells marked n.r.

| # | Paper (year, cites, DOI 10.1016/…) | Tier | (a) runs | (b) stats | (c) baselines | (d) ablations | (e) metrics | (f) HPO | (g) sensitivity | (h) data & CV | (i) release |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | FedraTrees — FL for smart-grid energy forecasting (2023, 16, j.engappai.2023.106654) | F | 1 run; no ± | none | 2 (FedAvg-LSTM, Persistence); **no centralized ceiling** | 2 shallow (feature count; stopping-window size) | MAE, computation time, communication rounds | random search stated; final values listed | feature-count sweep only | Tetouan (3 substations = 3 clients, 10-min, 1 yr); single 80/20 split | code: no; dataset public (cited) |
| 2 | DA-LSTM — drift-adaptive load forecasting (2023, 78, j.engappai.2023.106480; arXiv:2305.08767) | F | 1 run per household; no ± | none between methods (p-values are internal to the drift detector) | 4 (LSTM, Rolling ARIMA, Bagging Regr., RNN) + passive variant | 2–3 (drift-significance levels; passive vs active; granularity) | MAPE, RMSE + compute cost (CPU/GPU util., AWS $) | **Bayesian optimization, search space tabulated**, per-household; selected values tabulated | significance-level sweep {0.07, 0.10, 0.15}; per-household variability (≈10 households) | real consumption, ~1 yr, ~44k samples/household; single chronological split | none found |
| 3 | AnoFed — federated ECG anomaly detection (2023, 50, j.engappai.2023.106051) | A | n.r. | n.r. | "a number of state-of-the-art methods" | adaptive-vs-fixed detection | detection performance + computational cost (edge-oriented, parameter count) | n.r. | n.r. | ECG benchmarks | n.r. |
| 4 | FDDG — federated distillation domain generalization, machinery fault diagnosis (2023→v130 2024, 38, j.engappai.2023.107765) | A | n.r. | n.r. | multiple SOTA (claim: +3% acc.) | knowledge-sharing components | accuracy | n.r. | unseen-client (leave-domain-out) protocol | 2 rotating-machine testbeds | **code on GitHub (CHAOZHAO-1/FDDG)** |
| 5 | DL anomaly detection prior to state estimation, smart grid (2023, 35, j.engappai.2023.107368) | A | n.r. | n.r. | n.r. | n.r. | detection metrics | n.r. | n.r. | grid measurements (simulated) | n.r. |
| 6 | PB-fdGAN — permissioned blockchain + decentralized federated distillation GAN IDS for industrial CPS (2024, 29, j.engappai.2024.108862) | A | n.r. | n.r. | n.r. | component-wise (LDP-ecGAN, DFD) | detection metrics | n.r. | class imbalance addressed | ICS IDS datasets | n.r. |
| 7 | Data-driven SNN fault detection, EV Li-ion batteries (2024, 37, j.engappai.2024.109756) | A | n.r. | n.r. | n.r. | n.r. | detection metrics (+ low-power argument) | n.r. | real-vehicle operating conditions | vehicle operation data | n.r. |
| 8 | Multi-modal multi-sensor fusion SNN, early bearing weak faults (2024, 24, j.engappai.2024.109845) | A | n.r. | n.r. | "other feature-extraction and diagnostic algorithms" | fusion/denoising components | accuracy | n.r. | n.r. | bearing test data | n.r. |
| 9 | Optimized adaptive FL, pumping-unit diagnosis under label heterogeneity (2025, 19, j.engappai.2025.110724) | A | n.r. | n.r. | n.r. | init/aggregation components | accuracy + communication cost | n.r. | label-heterogeneity scenarios | pumping-unit data | n.r. |
| 10 | Privacy-preserving FL compatible with robust aggregators (2025, 13, j.engappai.2025.110078) | A | n.r. | n.r. | robust aggregators × privacy methods grid | privacy-mechanism placement | model accuracy + membership-inference attack accuracy | n.r. | attack settings | FL benchmarks | n.r. |
| 11 | Recurrence-plot + change-quantile protection for PV-connected transmission lines (2025, j.engappai.2025.113034; arXiv:2511.02084) | F | 1 run, single split; no seeds | **McNemar's test** between the two proposed classifiers (p = 1.0); no effect sizes | 2 proposed variants (ICT vs LST) + literature table (not re-run) | 2–3 (feature count, SSL paradigms/kernels) | Accuracy, P, R, F1, AUROC, AUPRC, confusion matrices + **runtime vs protection-cycle budget (12.82 ms half-cycle)** | fixed; SSL kernel params tabulated | **strong**: SNR {20,30,40 dB}, sampling freq., PV units {300,500}, series compensation, high-impedance, cross-country (1,188 cases) & evolving faults (396), unlabeled fraction 0.2–0.8 | simulated: 5,280 base cases (2,400 no-fault + 2,160 fault + extensions); single 70/30 split | **dataset released** on public platform; code not stated |
| 12 | SGDA — signature-guided augmentation, induction-motor diagnostics (2026, j.engappai.2026.114137; arXiv:2506.08412) | F | mean **±std reported**, but variance is across test conditions/segments — **not across seeds** | none | 5 classifiers (LR, SVM, MLP, CNN, ResNet18) | 3–5 (batch size, dropout, attention mask, configs — appendix) | macro-F1, accuracy, per-class, confusion matrices (binary + multiclass) | appendix sweeps (partial) | load {0–100%} × phase grid; leave-load-out; **cross-motor validation** (2nd industrial motor) | real 17 kW industrial motor, 10 kHz; held-out test | none found |
| 13 | Edge-computing methods for indoor-parameter estimation (2023, 17, j.engappai.2023.107149; arXiv:2403.08810) | F | informal repeats "until convergence"; no seeds/± | none | 7 ML models (MLP-ANN, SVM, kNN, RF, DT, XGB, ELM) × 2 architectures + related-work table | architecture ablation (central GPU vs distributed ARM) | MAE/RMSE + **measured node power consumption** + timing | manual topology selection (3-6-6-1) | 2 experiments (normal vs extreme temperature profiles) | 6,460 samples × 2 experiments; single split | none |
| 14 | EDGMAT — edge-directed graph multi-head attention NIDS (2026, j.engappai.2026.115284; arXiv:2310.17348) | F | 1 run; no ± | none ("significantly better" used informally, no test) | 4 SOTA (KNN, E-GraphSAGE, …) on 4 datasets | protocol grid (transductive/inductive × static/dynamic) | weighted F1, weighted accuracy/recall, per-class P/R/F1 | minimal (Adam + LR stated) | topology settings | 4 public NIDS datasets (BoT-IoT, ToN-IoT, NF-variants) | none found |

### Computed median / typical EAAI standard (over the 6 Tier-F papers; Tier-A abstracts corroborate — none mentions seeds or significance tests; 1 of 8 released code)

| Dimension | Median / typical practice | Best-in-sample |
|---|---|---|
| (a) Independent runs/seeds | **1 run, no seed protocol, no mean±std over runs** (0/6 seeded; 1/6 shows ± but over data conditions) | ±std heatmaps per condition (paper 12) |
| (b) Statistical testing | **None** (1/6 uses any test — McNemar; 0/6 Wilcoxon/Friedman; 0/6 effect sizes; 0/6 CIs) | single McNemar (paper 11) |
| (c) Baselines | **≈4** re-run comparison methods (range 2–7); FL papers: **no centralized ceiling, no local-only floor** re-run (0/1 Tier-F FL paper) | 7 models × 2 architectures (paper 13) |
| (d) Ablations | **2–3 shallow axes**, often appendix-level | 3–5 axes (paper 12) |
| (e) Metrics | task metric + per-class/confusion (4/6); AUROC 1/6; **an application cost number (latency/power/compute cost) in 4/6** — EAAI visibly rewards this | full P/R/F1/AUROC/AUPRC + cycle-budget latency (paper 11) |
| (f) HPO discipline | final values listed; systematic search stated in 2/6 (1 documented search space) | Bayesian HPO with tabulated space + outcomes (paper 2) |
| (g) Sensitivity analyses | **1–2 sweeps** | 6+ sweep families (paper 11) |
| (h) Data & CV | 5k–135k samples; **single hold-out split (6/6); k-fold 0/6** | multi-condition + cross-rig validation (papers 11, 12) |
| (i) Code/data release | **no code (0/6 Tier-F; 1/8 Tier-A)**; dataset release 1/6 | GitHub code (paper 4); public dataset (paper 11) |

Reading: the EAAI de-facto bar for applied AI papers in our neighborhood is a
single-seed, single-split study with ~4 baselines, informal use of the word
"significant", 2 shallow ablations, good application-cost reporting, and no
artifact. The best papers add one statistical test, rich sensitivity sweeps, or a
released dataset — never all at once. **Doubling the standard therefore means:
doubling the strongest single practice observed in the sample, on every axis
simultaneously.**

---

## §3 Our protocol — the doubled standard (binding checklist for the phase 4 battery)

Baseline for doubling: median = 1 seed/0 tests/4 baselines/2 ablations; best
observed = ±std, one McNemar, 7 comparisons, 6 sweeps, dataset release. Adjacent
Byzantine-FL literature (NDSS/USENIX canon) typically uses 3–5 repetitions. We
double the best: **10 seeds, a full non-parametric battery with effect sizes and
CIs, 12 named comparison arms, 8 ablation axes, 6 sensitivity families, and a
tested, released artifact.**

### 3.1 Seeds and repetitions
- [ ] **Tier 1 (headline tables/figures): 10 seeds** (s = 0…9) per configuration.
- [ ] **Tier 2 (sweeps/ablations): 5 seeds** per grid point (still ≥ adjacent-field norm).
- [ ] One seed drives the *entire* stochastic pipeline: federation generation
      (feeder load-shape seeds, Dirichlet draw, event plans), model init, batching,
      LDP noise, attack RNG, PBFT network latency. Re-partitioning per seed makes the
      10 repetitions also 10 dataset re-draws — stronger than k-fold on a frozen split
      (which the EAAI sample never does anyway).
- [ ] Every table cell = mean ± std over seeds with n stated; every figure shows CI
      bands (§3.2). Per-seed raw values ship in the artifact (RunResult files).

### 3.2 Statistical battery (chosen to match what this community accepts — Demšar
protocol for multi-method comparison, non-parametric throughout; doubled in rigor
by adding effect sizes + CIs + pre-registration, which 0/14 sampled papers have)
- [ ] Pairwise headline claims (SNN vs CNN-LSTM at matched config; defense vs
      no-defense; LDP on vs off): **Wilcoxon signed-rank** across the 10 paired seeds,
      two-sided, **Holm-corrected** within each hypothesis family; report W and exact p.
- [ ] ≥3 methods across conditions (5 defense arms over the attack grid):
      **Friedman test** with blocks = (attack × model) conditions on seed-means, then
      **Nemenyi critical-difference diagram + Holm-adjusted comparisons vs our
      pipeline** (Demšar, JMLR 2006).
- [ ] **Effect sizes** for every Wilcoxon: Cliff's delta with magnitude bands, plus
      the absolute Δmacro-F1.
- [ ] **95% confidence intervals**: BCa bootstrap, 10,000 resamples, over seeds, for
      every reported mean; CI bars/bands on all result figures.
- [ ] **McNemar's test** (the one test the EAAI sample itself uses) between
      CNN-LSTM and SNN predictions on the pooled global test set at matched
      configuration — the "matched accuracy" claim gets its own significance test.
- [ ] No parametric tests, no normality assumptions; α = 0.05 two-sided.
- [ ] **Pre-registration**: hypothesis list committed to `experiments/HYPOTHESES.md`
      *before* Tier-1 runs start; deviations documented, never silently dropped.

### 3.3 Baseline / comparison arms (12 named training arms; median found: 4, no
ceiling/floor — we run the full reference frame)
- [ ] B1/B2 **Centralized ceiling** — CNN-LSTM and SNN trained on pooled data (upper bound).
- [ ] B3/B4 **Local-only floor** — per-client training, no federation; report mean/min/max over the K=10 clients (both models).
- [ ] B5/B6 **FedAvg plain** (both models).
- [ ] B7/B8 **FedProx** (μ = 0.01 default; μ ablated in §3.4) (both models).
- [ ] B9/B10 **Full pipeline** (fidelity → PBFT → robust agg) with **Krum** (both models).
- [ ] B11/B12 **Full pipeline with geometric median** (both models) — satisfies the
      contract's Krum-vs-geomedian requirement and the blockchain-vs-Krum-alone ablation arm.
- [ ] Under attack, additionally: **no-defense (FedAvg) under each attack** and
      **Krum-alone / geomedian-alone (no fidelity, no PBFT)** — isolates what the
      consensus layer buys (accuracy vs auditability; the paper's honesty point).
- [ ] Excluded with stated reason (pre-empting reviewers): FLTrust-style root-dataset
      trust — assumes a clean server dataset, unavailable by design in a multi-utility
      setting; cited and discussed instead of re-run.

### 3.4 Ablation matrix (8 axes; median found: 2–3 shallow)
- [ ] A1 Defense decomposition: none → fidelity-only → +Krum → +PBFT+Krum → +PBFT+geomedian (under AdaptiveCosine).
- [ ] A2 Aggregator: Krum vs multi-Krum vs geometric median (under attack).
- [ ] A3 SNN coding: rate vs delta (clean + under attack).
- [ ] A4 SNN T_steps ∈ {4, 8, 16} (accuracy–energy–latency trade).
- [ ] A5 Fidelity threshold τ sweep (drives the operating curve, §3.5).
- [ ] A6 LDP × defense 2×2 (on/off × on/off) — the tension made visible.
- [ ] A7 FedProx μ ∈ {0.001, 0.01, 0.1} vs FedAvg (heterogeneity handling).
- [ ] A8 Blockchain-vs-Krum-alone (accuracy identical or not + overhead measured) — already implied by A1 but reported as its own named table.

### 3.5 Sensitivity sweeps (6 families; median found: 1–2)
- [ ] S1 Non-IID severity: Dirichlet **α ∈ {0.1, 0.5, 1.0} + IID control** (uniform), per contract amendment §4 (Hsu et al. protocol).
- [ ] S2 Attack fraction: **{0, 1, 2, 3}/10 clients** (0 = clean reference; 3 = f_max at K=10).
- [ ] S3 LDP noise: **σ ∈ {0.25, 0.5, 1.0, 2.0} × fidelity threshold τ ∈ {3 values}** → the empirical σ×τ operating curve (honest-client false-rejection rate vs adaptive-attacker admission rate) — pillar-4 deliverable.
- [ ] S4 Validators: **M ∈ {4, 7, 10}** with f = ⌊(M−1)/3⌋ — consensus overhead vs robustness.
- [ ] S5 Attack type: SignFlip, GaussianNoise, LabelFlip, AdaptiveCosine (defense-aware) — full grid in Tier 1.
- [ ] S6 Precision: fp32 vs INT8 (accuracy delta, latency, estimated energy) — evaluation-time, no retraining (PTQ).
- [ ] K (client count) stays fixed at 10 per the data contract; a K sweep is declared out of scope in the manuscript (cross-silo setting, K set by the co-simulation design) — honesty note rather than a silent omission.

### 3.6 Metrics set (every run logs all of these into RunResult)
- [ ] Classification: accuracy, **macro-F1** (headline, 8 classes), weighted-F1, per-class F1, full confusion matrix, one-vs-rest macro ROC-AUC.
- [ ] Robustness: accuracy degradation under attack, attacker admission rate, honest false-rejection rate (per round and final).
- [ ] Consensus: message count (O(M²) verified), simulated latency ms, rejected-set trace, ledger verification.
- [ ] Privacy: ε (RDP composition) per LDP configuration.
- [ ] Energy: SynOps, dense-MACs, spike rate, **estimated** µJ/inference (Horowitz constants, fp32 + INT8 rows) and SNN:ANN ratio — "estimated via SynOps accounting", never "measured" (contract amendment §1).
- [ ] Latency: **measured** p50/p95/p99 (n = 1000, warm-up 50) per model × precision × device, reported against the 4 ms GOOSE transfer-time budget.
- [ ] Cost: rounds-to-target-accuracy, bytes/round communication, wall-clock/run.

### 3.7 Hyperparameter discipline (median found: final values only)
- [ ] Grid search documented in the paper (space + selection metric + budget): LR ∈ {1e-2, 1e-3, 1e-4}, batch ∈ {32, 64}, local epochs ∈ {1, 2}, hidden sizes fixed by contract; selection on **validation** macro-F1 only, never test.
- [ ] Identical tuning budget for both models and all baselines (fair-comparison clause of the Guide, §1) — tuned once on clean FedAvg, then **frozen** for all attack/defense/sweep configurations.
- [ ] All final values + search logs in the artifact (`experiments/configs/`).

### 3.8 Software-testing bar for the released artifact (0/14 sampled papers release tested code — this axis is our clearest differentiator)
- [ ] Unit tests: every contract correctness bar (tests/test_simulation.py, test_models.py, test_fl.py, test_consensus.py) green from repo root.
- [ ] **Property-based tests** (Hypothesis) per module: aggregation permutation-invariance and weighting exactness ∀ random client sets; clip_and_noise post-clip norm ≤ clip ∀ inputs; fidelity scores ∈ [0, 2] ∀ update sets; ledger verify() fails ∀ single-field tampers; PBFT never commits with > f faulty ∀ sampled topologies.
- [ ] **Determinism tests**: same-seed engine run on CPU produces byte-identical serialized RunResult (sha256-stable); CUDA nondeterminism documented with observed tolerance.
- [ ] Golden regression fixture: a tiny fixed-seed 2-round federation whose exact metrics are asserted, catching silent numeric drift.
- [ ] Coverage gate: **≥ 85% line coverage** on `src/fedgrid/{simulation,fl,consensus,attacks,metrics,models}`, **≥ 75% overall** (api included); frontend `npm run build && npm run lint` green.
- [ ] Battery manifest dry-run: `experiments/run_all.py --dry-run` enumerates every planned run (config hash + seed) without training; the manifest is committed before Tier-1 starts.
- [ ] Release: public GitHub + **Zenodo DOI archive** (code, configs, all RunResult JSONs, figure scripts); data availability statement = generator + one-command regeneration (+ optional deposit of generated windows). During review: anonymized artifact link (double-anonymized model, §1).
- [ ] RunResult must carry: seed, full config, git commit hash, environment fingerprint (torch/CUDA versions, GPU name).

### 3.9 Run budget and feasibility on one shared RTX 3090 (≤ 16 GB VRAM cap)

Planned training runs:

| Block | Grid | Runs |
|---|---|---|
| T1a clean reference frame | 12 arms (B1–B12) × 10 seeds | 120 |
| T1b attack × defense (f_att = 3/10, α = 0.5) | 4 attacks × 5 defense arms × 2 models × 10 seeds | 400 |
| T2a α sweep | {0.1, 1.0, IID} × {FedAvg, pipeline-Krum} × 2 models × {clean, adaptive} × 5 seeds | 120 |
| T2b attack-fraction sweep | {1, 2}/10 × {SignFlip, Adaptive} × {no-defense, pipeline-Krum} × 2 models × 5 | 80 |
| T2c LDP σ×τ operating curve | 4σ × 3τ × pipeline-Krum × 2 models × 5 (attack + LDP-honest in same run) | 120 |
| T2d validators M | {4, 10} × pipeline-Krum × 2 models × 5 | 20 |
| T2e model/strategy ablations | coding (2 arms), T_steps (2), FedProx μ (2 × 2 models), fidelity-only (2 models) × 5 | ~60 |
| Margin (failures/re-runs, ~8%) | | ~75 |
| **Total** | | **≈ 995 ≈ 1,000 runs** |

Cost model (estimate — to be replaced by a **measured phase 3 pilot**): dataset per
client ≈ 1.5–3k windows of (11×64) floats; models < 1 M params; 30 FL rounds ×
10 clients × ~30 batches ⇒ CNN-LSTM ≈ 3–5 min/run, SNN (T = 8 unroll) ≈ 10–15
min/run on the 3090 ⇒ average ≈ 8 min/run ⇒ **≈ 130 GPU-hours ≈ 5.5 days of pure
compute; plan 2–3 calendar weeks** on the shared GPU (headroom for the co-tenant,
nvidia-smi checks, sequential queue). Latency measurement (n = 1000 × 2 models × 2
precisions × 2 devices) and INT8 PTQ are minutes, not hours. **Verdict: feasible;
not computationally unrealistic — but only with the phase 3 timing gate.**

Timing gate + trim ladder (honesty over bravado — applied in order, each step
recorded as a dated amendment, never silently):
1. If measured average run cost > 15 min: T1b drops 10 → 5 seeds (−200 runs; Wilcoxon n = 5 pairs stated openly as the floor).
2. If still over budget: T1b drops the geomedian defense arm under Gaussian/LabelFlip (kept under SignFlip/Adaptive where it matters) (−80).
3. If still over budget: T2c runs CNN-LSTM only, SNN spot-checked at the 4 curve corners (−50).
4. Never trimmed: the 12-arm reference frame at 10 seeds, the AdaptiveCosine column, the statistical battery, the pre-registration, the artifact tests.

---

## §4 Recommended CONTRACT.md amendments (exact text, ready to prepend)

> **Amendment 2026-07-28 (validation standard, evidence in `docs/VALIDATION_STANDARD.md`):**
> 1. **phase 4 battery is bound by `docs/VALIDATION_STANDARD.md` §3** (checklist):
>    Tier-1 = 10 seeds, Tier-2 sweeps = 5 seeds; ≈ 1,000 training runs total;
>    a phase 3 pilot must *measure* per-run cost and, if the estimate is exceeded,
>    apply the §3.9 trim ladder via dated amendment — never silently.
> 2. **Statistics**: Wilcoxon signed-rank + Holm for pairwise claims; Friedman +
>    Nemenyi/Holm across defense arms; Cliff's delta + BCa 95% CIs (10k resamples)
>    on every headline mean; McNemar for the CNN-LSTM-vs-SNN matched-accuracy claim;
>    non-parametric only; hypotheses pre-registered in `experiments/HYPOTHESES.md`
>    before Tier-1 runs start.
> 3. **Baseline frame fixed (12 arms)**: centralized ceiling ×2 models, local-only
>    floor ×2, FedAvg ×2, FedProx ×2, full pipeline (Krum) ×2, full pipeline
>    (geometric median) ×2; plus no-defense-under-attack and Krum/geomedian-alone
>    arms in the attack grid. No arm dropped without a dated amendment.
> 4. **Artifact bar**: coverage ≥ 85% on core `src/fedgrid` modules (≥ 75% overall);
>    property-based tests (Hypothesis) on aggregation/LDP/fidelity/ledger/PBFT
>    invariants; same-seed CPU determinism test asserting a stable RunResult hash;
>    golden tiny-run regression fixture; `experiments/run_all.py --dry-run` manifest
>    committed before Tier-1. `RunResult` gains mandatory `git_commit` and
>    `env_fingerprint` fields (the models/metrics module; backward-compatible addition).
> 5. **Manuscript compliance (EAAI)**: abstract ≤ 250 words; 3–5 highlights ≤ 85
>    chars each; double-anonymized submission with an anonymized artifact link
>    during review (public GitHub + Zenodo DOI on acceptance); CRediT statement;
>    data availability statement (generator + regeneration command); Elsevier
>    generative-AI declaration section with the exact template wording (tool +
>    reason + responsibility sentence), placed before the references; methodology
>    AI use disclosed in Methods, not in that statement; title without
>    non-established acronyms; ≤ 50 pages.

(No change to the data contract, module ownership, or wave plan; the α sweep of
amendment 2026-07-28 §4 is subsumed by §3.5-S1 above with an added IID control.)

---

## §5 Sources (all verified 2026-07-28)

Journal / policy:
1. EAAI Guide for Authors — https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence/publish/guide-for-authors (live page fetched 2026-07-28 via r.jina.ai reader proxy; ScienceDirect returns 403 to direct robotic fetches).
2. Elsevier, "Generative AI policies for journals" — https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals (fetched 2026-07-28; exact declaration template quoted in §1).
3. OpenAlex journal record for EAAI (citation counts, OA locations) — https://api.openalex.org/sources/S900972176.

Sampled EAAI papers (DOI; full-text copy used where Tier F):
4. FedraTrees — DOI 10.1016/j.engappai.2023.106654; accepted MS: https://eprints.gla.ac.uk/300846/1/300846.pdf.
5. DA-LSTM — DOI 10.1016/j.engappai.2023.106480; arXiv:2305.08767.
6. AnoFed — DOI 10.1016/j.engappai.2023.106051 (abstract via Semantic Scholar; repository copies bot-blocked).
7. FDDG — DOI 10.1016/j.engappai.2023.107765; code: https://github.com/CHAOZHAO-1/FDDG.
8. Smart-grid resilience anomaly detection — DOI 10.1016/j.engappai.2023.107368.
9. PB-fdGAN blockchain + decentralized FL IDS — DOI 10.1016/j.engappai.2024.108862.
10. SNN Li-ion battery fault detection — DOI 10.1016/j.engappai.2024.109756.
11. Multi-modal multi-sensor SNN bearing diagnosis — DOI 10.1016/j.engappai.2024.109845.
12. Adaptive FL pumping-unit diagnosis — DOI 10.1016/j.engappai.2025.110724.
13. Privacy-preserving FL with robust aggregators — DOI 10.1016/j.engappai.2025.110078.
14. PV transmission-line protection — DOI 10.1016/j.engappai.2025.113034; arXiv:2511.02084.
15. SGDA induction-motor diagnostics — DOI 10.1016/j.engappai.2026.114137; arXiv:2506.08412.
16. Edge-computing IoT estimation — DOI 10.1016/j.engappai.2023.107149; arXiv:2403.08810.
17. EDGMAT graph NIDS — DOI 10.1016/j.engappai.2026.115284; arXiv:2310.17348.
18. (context) Backdoor attacks & defenses in FL survey, EAAI — DOI 10.1016/j.engappai.2023.107166; arXiv:2303.02213.

Methodology references for §3 (community-standard, verifiable):
19. Demšar, "Statistical Comparisons of Classifiers over Multiple Data Sets", JMLR 7:1–30, 2006 (Friedman + Nemenyi/Holm protocol).
20. Holm, "A simple sequentially rejective multiple test procedure", Scand. J. Statist. 6(2):65–70, 1979.
21. Cliff, "Dominance statistics: Ordinal analyses to answer ordinal questions", Psychological Bulletin 114(3):494–509, 1993 (Cliff's delta).
22. Hsu, Qi, Brown — arXiv:1909.06335 (Dirichlet non-IID protocol; already pinned in CONTRACT.md).
23. Horowitz, ISSCC 2014 — DOI 10.1109/ISSCC.2014.6757323 (energy constants; already pinned).

Access caveat repeated for the record: Tier-A cells marked n.r. could not be
verified because the papers are paywalled with no open copy; nothing was inferred
for those cells. arXiv author versions (5, 14–17) may differ from the journal
versions in minor editorial ways.
