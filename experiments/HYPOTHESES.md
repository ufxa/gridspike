# Pre-registered hypotheses — FedGrid phase 4 battery

## Amendment — X3 coverage extension (more seeds on the same cells; no hypothesis registered)

Added **after** the registered analysis, to retire the seed-budget caveat that was
flooring every Wilcoxon family. The X3 batteries (`runs_x3`, `runs_ssm_x3`,
`runs_gap_x3`, plus the `runs_snn_fixed` recipe reference) re-ran every shared
seed and extended each cell: Tier-1 and the temporal controls now run seeds
0–14 (15), and every Tier-2 sweep, the T3tau threshold/LDP grid and the T4mag
magnitude control run seeds 0–8 (9).

1. **What it changes.** Only sample sizes. Every cell keeps its registered
   attack, defense, model, sweep value and decision rule; the tests, families
   and margins below are unchanged. On `run_id` or semantic-config collisions
   the expansion copy wins by convention (documented in
   `docs/X3_ANALYSIS.md` Part 2 / P0), and T3tau's emitted triplicates are
   deduplicated on their semantic key because the expansion manifests carry
   the seed line's `config_hash` (a documented staleness, not drift).
2. **What this licenses.** Reporting the battery at its realized seed counts:
   an exact one-sided Wilcoxon now reaches $2^{-15}$ at Tier-1, so every
   pre-registered family is decidable at $\alpha = 0.05$ after Holm
   correction. Families that were "undecidable at this seed budget" are
   decided; verdicts are reported against the same registered rules.
3. **What this does not license.** No null, alternative, metric, test,
   correction family, margin or decision rule is registered or changed. The
   Tier-3 blocks remain post-hoc controls with no hypothesis attached, and no
   verdict in H1–H5 may be rescued by an unregistered test even at the larger
   $n$.

No hypothesis, no metric, no test, no margin and no direction registered below is
changed by this amendment; it adds runs, not claims.

---

## Amendment 2026-07-31a — label-flip is realized as *dynamic per-batch label noise*

The registration lists label-flip among the four attacks of the H2 family without saying
how it is realized, and the natural reading — a static, one-time relabelling of the
Byzantine client's training set — is **not** what the executed battery ran. Recorded here
because it changes what the label-flip cells of T1b and T2b measure, not because any test
changed. Verified against the code, not the docstrings: `src/fedgrid/attacks/suite.py`
(`LabelFlip`) and `src/fedgrid/fl/engine.py` (`_flip_labels`, `_fold_rng`, the client
training loop).

1. **What the implementation actually does.** `LabelFlip` is a data-level *config* object:
   it exposes `frac` and `seed` and deliberately has no `poison` method, so the engine
   duck-types it (`not hasattr(atk, "poison")` → `data_level`) and never routes it through
   the model-poisoning hook. Instead, inside the Byzantine client's training loop, every
   batch of every local epoch of every round is passed through `_flip_labels`, which moves
   `round(frac * len(batch))` of that batch's labels to a uniformly random **different**
   class (`offsets = rng.integers(1, n_classes)` then `(y + offsets) % n_classes`, so a
   flip can never land back on the true label). The generator is
   `_fold_rng(attack_stream, LabelFlip.seed)`, derived per (round, client) from the run
   seed, so the flip pattern is fresh at every batch and still deterministic under a fixed
   seed. Only training labels are touched — validation and test labels are never flipped.
2. **The realized threat model.** Dynamic per-batch label noise at rate `frac`, not a
   one-time dataset corruption. Averaged over the 30 rounds it approaches symmetric label
   noise: the poisoning direction is re-randomized every batch instead of being
   consistently reinforced by a fixed corrupted set, which makes it a **weaker and
   higher-variance adversary** than static poisoning. Label-flip results must be read
   against that adversary.
3. **What this does not license.** No null, alternative, metric, test, correction family
   or decision rule changes; label-flip remains one of the four attacks in the H2 family
   and its cells are corrected inside it exactly as registered. It does **not** license
   describing these results as static data poisoning, and no claim about static poisoning
   is made anywhere in the manuscript or the artifact.
4. **Scope note.** `LabelFlip.flip()` in `attacks/suite.py` *is* a static reference
   implementation (one deterministic relabelling from `(seed, cid, len(labels))`). It is
   retained as the basis of a future static-poisoning variant and is exercised only by the
   unit tests; it is **not** on the federated training path and produced no result in this
   battery.

No hypothesis, no metric, no test, no margin and no direction is changed by this amendment.

---

## Amendment 2026-07-31b — Tier-3 control battery (post-hoc controls, no hypothesis registered)

Added **after** the pre-registered analysis was complete. Review identified two threats to
validity that the registered design does not cover, neither of which has a registered
hypothesis: (a) under the random client split, windows adjacent in time can fall on both
sides of the split, so the clean reference frame may be optimistic; (b) the attacker's
update *magnitude* is the one lever a scale-free (cosine) fidelity gate cannot see, and
the registered grid holds it fixed. Rather than leave both unaddressed or quietly fold
them into the registered blocks, they are run as an explicitly post-registration control
battery and reported as such.

1. **What was added — 534 runs, all executed after the registered analysis.**
   - `T4mag` (144): attacker magnitude `s ∈ {1,2,5,10}` × 4 defense arms × 3 models,
     adaptive-cosine at `f_att = 3/10`, 3 seeds (the Tier-2 budget it controls).
   - `T1a_temporal` (90) and `T1b_temporal` (300): blocks T1a and T1b re-run under a
     contiguous **chronological** client split, 5 seeds (the Tier-1 budget they control).
     Each temporal run copies its T1 twin's manifest line and changes one field, so twin
     and control are matched on every other axis and pair by seed.
   - Emitted by the same manifest machinery, the same configuration hashing and the same
     run-id discipline as the pre-registered blocks. 534 records, 534 distinct
     configurations, none diverged.
2. **What this licenses.** Reporting these blocks as **post-hoc controls on the validity
   of the pre-registered results**, labelled Tier-3 / post-registration everywhere they
   appear — battery table, manifests, run ids and the results sections that discuss them.
   Descriptive statistics, effect sizes and BCa intervals are reported for them on the
   same footing as elsewhere, as description.
3. **What this does not license.** These are **not pre-registered hypotheses.** No null,
   no alternative, no direction, no correction family and no decision rule is registered
   for them, so no p-value computed on these blocks may be presented as a pre-registered
   test outcome, and none of them may be used to rescue, overturn or re-decide a verdict
   in H1–H5. Where a control disagrees with a registered result, the disagreement is
   reported as a threat to validity of that result — not as a revised verdict, and not as
   a new finding with registered status.

No hypothesis, no metric, no test, no margin and no direction registered below is changed
by this amendment; it adds runs, not claims.

---

## Amendment 2026-07-30a — analysis corrections (no hypothesis, metric or test changed)

Four defects in the *analysis code*, not in this registration, were found by review and
fixed in `experiments/report.py`. Recorded here because the numbers they produce change.

1. **The Mamba (`ssm`) was missing from the H2, H3 and H5 test loops.** The registration
   says "per model" and the battery ran three models (T1b and T2a carry `ssm` in full),
   but the implementation iterated only `cnn_lstm` and `snn`. Family sizes were therefore
   under-counted: H2 is **24** cells (2 pipelines × 3 models × 4 attacks), not 16; H3a and
   H3b are **6** each; H5a is **6** trends; H5b is **3** models. Restored to three models.
   *Consequence:* H5a falls from 4/4 supported to **2/6** (both Mamba trends are flat,
   Jonckheere p ≈ 0.5), so H5a — and with it H5 — is **not supported**. Reported as found.
2. **The Friedman was pseudo-replicated.** The "Statistical conventions" section above
   fixes blocks = (attack × model) **seed-means**; the implementation used one block per
   *seed row*, multiplying the block count by 5 and shrinking the Nemenyi critical
   difference accordingly. The pre-registered blocking (N=12 with three models) is now the
   **primary** result and the per-seed variant is reported beside it, labelled as a
   **sensitivity analysis**. Both appear in `tab_stats` and in `runs/summary.json`.
3. **Power ceiling, stated rather than implied.** An exact one-sided Wilcoxon signed-rank
   test on n paired seeds cannot return p < 2⁻ⁿ, and Holm within a family of m members
   multiplies the smallest p by m. At the seed budget this battery actually ran, the H2
   (m=24, n=5), H3a, H3b (m=6, n=5) and H5b (m=3, n=3) families have Holm floors of 0.75,
   0.1875 and 0.375 — all above α=0.05. **No configuration of the data could have declared
   a single cell in those families significant.** The "not supported" verdicts there report
   the seed budget of this design, not a refuted defense; the per-cell Cliff's δ and BCa
   intervals carry what evidence exists. `tab_hypotheses` and `tab_stats` now say so in a
   power note. This is a disclosure, not a changed decision rule: the rules below stand.
4. **Evidence strings inverted.** `tab_hypotheses` read "0/16 cells decided (16 planned)"
   where 16 was the number *decided*, not undecided. Now "0/24 support (24/24 cells
   decided)".

No hypothesis, no metric, no test, no margin and no direction is changed by this amendment.

---

Registered on **2026-07-28, before any Tier-1 run** (CONTRACT
amendment 2026-07-28d §2). Changes after this point only by dated amendment at the top
of this file — never silently. Battery definition: `docs/VALIDATION_STANDARD.md` §3;
arms and seeds: CONTRACT amendments d§3 (12-arm frame) and e/h (Compact battery,
Tier-1 = 5 seeds, Tier-2 = 3 seeds). The battery grid is enumerated in
`experiments/manifest.py` (`experiments/run_all.py --dry-run` → `experiments/manifest.jsonl`)
and every test named below is implemented in `experiments/analyze.py`.

This file is **frozen once the first Tier-1 run starts**. No result appears here; only
directions, metrics, tests and decision rules, fixed in advance.

---

## Statistical conventions (bind every hypothesis below)

Chosen to match what this community accepts (Demšar, JMLR 2006), non-parametric
throughout, doubled in rigor by adding effect sizes + CIs + this pre-registration
(0/14 sampled EAAI papers had any of the three — `docs/VALIDATION_STANDARD.md` §2).

- **Unit of analysis**: one seed = one independent repetition (federation re-draw +
  model init + batching + LDP/attack/PBFT RNG all from that seed, §3.1). Paired tests
  pair by seed; the two arms in a pair share the seed, hence the same federation draw.
- **Significance**: α = 0.05, **two-sided**, unless a one-sided direction is stated.
- **Pairwise claims**: **Wilcoxon signed-rank** on the per-seed paired differences,
  **Holm-corrected within the hypothesis family** named in each entry; report W, exact
  p, Holm-adjusted p, and n (pairs).
- **≥3-arm comparisons**: **Friedman** across arms with blocks = (attack × model)
  seed-means, then **Nemenyi critical-difference** and Holm-adjusted post-hoc vs the
  full pipeline.
- **Effect size**: **Cliff's δ** (magnitude bands: |δ|<0.147 negligible, <0.33 small,
  <0.474 medium, else large; Romano et al. 2006) for every Wilcoxon, plus the absolute
  Δ in the metric.
- **Confidence intervals**: **BCa bootstrap, 10,000 resamples** over seeds, on every
  reported mean and on every headline ratio/difference (implemented in numpy —
  `scipy` is not a dependency; the numpy path is asserted in the tests).
- **Headline metric**: **macro-F1** on the pooled global test set under the realistic
  NORMAL≥90% prior (§3.6); accuracy and per-class F1 reported alongside, never instead.
- **No parametric test, no normality assumption** anywhere.

The reference frame (not a hypothesis, the interpretive envelope): every federated
result is read **between the centralized ceiling (B1/B2)** — the best a single model on
pooled data reaches — **and the local-only floor (B3/B4, mean/min/max over K=10)** — what
a substation gets training alone. A federated arm that beats the floor buys something
over going it alone; the gap to the ceiling is the price of decentralization + privacy +
Byzantine tolerance. Both bounds are reported with BCa CIs; neither is a significance test.

---

## H1 — Spiking energy ≪ CNN-LSTM at matched accuracy

The headline P6 claim: the SNN classifies substation events at accuracy statistically
indistinguishable from the CNN-LSTM while spending far less estimated energy per
inference.

- **Two linked sub-claims** (both must hold for H1):
  - **H1a (parity)** — SNN and CNN-LSTM predictions on the pooled global test set are
    *not* distinguishable in accuracy at the matched clean-FedAvg configuration.
    - H0: the two classifiers have equal error rates (McNemar symmetry: b = c).
    - H1a-alt: they differ (two-sided). **We seek to *retain* H0** (parity is the
      premise of the energy claim), so we also pre-register an equivalence check.
    - Metric: paired correct/incorrect on identical pooled-test samples, pooled per seed.
    - Test: **McNemar's test** (exact binomial on discordant pairs; the one test the EAAI
      sample itself uses). Reported per seed and pooled; plus a **TOST equivalence** on
      Δaccuracy with margin ±0.02 macro-F1 (pre-set) so "no difference" is a positive
      finding, not merely a non-rejection.
    - Decision rule: parity **supported** iff McNemar p > 0.05 (fail to reject) **and**
      the BCa 95% CI of Δmacro-F1 lies within ±0.02.
  - **H1b (energy)** — SNN estimated energy per inference is far below the CNN-LSTM's.
    - H0: energy ratio ρ = E_SNN / E_ANN ≥ 1.
    - H1b-alt: ρ < 1 (one-sided; the paper's direction).
    - Metric: ρ from SynOps accounting — `synops·E_AC` vs `macs·E_MAC` (Horowitz 2014
      constants), per `experiments/build.py` energy block; fp32 headline, INT8 row from
      the `*_INT8_PJ` constants (eval-time PTQ).
    - Test: **BCa 95% CI of ρ over seeds**; H1b supported iff the CI upper bound < 1.
      Wilcoxon signed-rank (one-sided) on per-seed ρ vs 1 as a secondary check.
    - Effect: report ρ and 1/ρ ("× cheaper") with the CI; Cliff's δ of SynOps vs MACs.
- **H1 verdict**: supported iff **H1a parity holds AND H1b CI upper bound < 1**.
- **Honesty note**: energy is *estimated via SynOps accounting* (arithmetic-only, memory
  excluded), never "measured" (CONTRACT amendment §1); latency is measured separately
  (§H-latency below) against the 4 ms GOOSE budget.

## H2 — The full pipeline beats no-defense under attack

The defense actually defends: under each attack, fidelity→PBFT→robust-agg recovers
accuracy that plain FedAvg loses.

- H0: under attack, macro-F1(pipeline) ≤ macro-F1(no-defense), per model.
- H1-alt: macro-F1(pipeline) > macro-F1(no-defense) (one-sided), for both Krum and
  geometric-median pipelines.
- Metric: final global-test macro-F1; paired by seed within each (attack × model) cell.
- Test: **Wilcoxon signed-rank** per (attack × model × {pipeline-Krum, pipeline-geomed}),
  **Holm-corrected across the H2 family** (all attack×model×pipeline cells). Also
  **Friedman + Nemenyi** across the five defense arms (no-defense, Krum-alone,
  geomed-alone, pipeline-Krum, pipeline-geomed) blocked by (attack × model).
- Effect: Cliff's δ + ΔmacroF1 with BCa CI per cell.
- Decision rule: H2 **supported** for a cell iff Holm-adjusted p < 0.05 **and** δ is at
  least *small* in the predicted direction. The **auditability honesty point** is
  reported even when Krum-alone matches the pipeline on accuracy: the pipeline additionally
  yields the tamper-evident ledger + measured consensus overhead, which Krum-alone does not
  (this is a stated qualitative contribution, not a p-value).

## H3 — AdaptiveCosine evades the fidelity gate yet still does more damage than the rejected SignFlip

The novelty lever (P3): a defense-aware attacker keeps its fidelity score under the
admission threshold (so it is *admitted* where SignFlip is *rejected*), and precisely
because it is admitted it shifts the aggregate / costs more accuracy than the loud attack
that the gate catches.

- **H3a (evasion)** — realised fidelity score of AdaptiveCosine < τ ≤ realised score of
  SignFlip, per defense arm.
  - H0: score(Adaptive) ≥ score(SignFlip).
  - H1-alt: score(Adaptive) < score(SignFlip) (one-sided), and admission-rate(Adaptive) >
    admission-rate(SignFlip).
  - Metric: per-round realised fidelity score of attacker updates; attacker admission rate.
  - Test: **Wilcoxon signed-rank** on per-seed mean attacker score (Adaptive vs SignFlip),
    paired by seed, per pipeline defense; Holm within the H3 family.
- **H3b (damage-given-evasion)** — the accuracy drop under AdaptiveCosine ≥ the drop under
  SignFlip, *under the same pipeline defense*.
  - H0: ΔmacroF1(Adaptive) ≤ ΔmacroF1(SignFlip) (drop = clean − attacked, same seed).
  - H1-alt: ΔmacroF1(Adaptive) > ΔmacroF1(SignFlip) (one-sided).
  - Metric: macro-F1 degradation vs the clean reference (paired by seed).
  - Test: **Wilcoxon signed-rank** paired by seed per (model × pipeline arm), Holm-corrected
    in the H3 family; Cliff's δ + BCa CI of the drop difference.
- **H3 verdict**: supported iff H3a (evasion) **and** H3b (≥ damage) both hold for at least
  the geometric-median pipeline (the averaging aggregator the scale lever targets). Reported
  against **both** Krum and geometric median (CONTRACT amendment 2026-07-28 §3).

## H4 — LDP σ×τ operating curve: honest false-reject rate rises with σ

The P4 empirical contribution (the Guerraoui et al. PODC-2021 tension made measurable):
stronger local DP noise makes honest clients look Byzantine, so the honest-client
false-rejection rate is monotone increasing in σ at fixed τ, and the σ×τ grid traces the
operating curve between honest-rejection and adaptive-admission.

- H0: honest false-reject rate (FRR) is independent of σ at fixed τ (no monotone trend).
- H1-alt: FRR increases with σ (one-sided, monotone) at each τ.
- Metric: **honest-client false-rejection rate** (honest clients scoring > τ, per round and
  final) and, jointly, the **adaptive-attacker admission rate** — the two axes of the curve.
- Test: **Jonckheere–Terpstra** ordered-trend test across σ ∈ {0.25,0.5,1.0,2.0} at each τ
  (a numpy implementation with a normal approximation; cited in `analyze.py`); **Spearman**
  ρ(σ, FRR) as a secondary trend statistic; **BCa 95% CI** of FRR at each (σ,τ) grid point.
  Where JT is unavailable the pre-registered fallback is the per-adjacent-σ **Wilcoxon**
  (Holm-corrected across the three σ steps) requiring monotone increase.
- Decision rule: H4 **supported** at a τ iff the ordered-trend p < 0.05 with FRR increasing,
  and the per-(σ,τ) FRR CIs are reported for the curve figure regardless of significance.
- Note: privacy budget ε per σ is reported (RDP accountant, `fl/ldp.epsilon`) so the curve
  carries its ε-axis; ε is a computed guarantee, not a tested quantity.

## H5 — Non-IID severity degrades accuracy, and hurts FedAvg more than the robust pipeline

Heterogeneity (P1's non-IID-by-construction design) costs accuracy, and the plain
average suffers more from it than the pipeline — the robust aggregator's selection also
damps the heterogeneity-driven client drift.

- **H5a (degradation)** — macro-F1 falls as α falls (0.5 → 0.1 more skewed; IID control
  is the near-uniform upper anchor).
  - H0: macro-F1 independent of α.
  - H1-alt: macro-F1 monotone increasing in α (IID ≥ 1.0 ≥ 0.5 ≥ 0.1), one-sided.
  - Test: **Jonckheere–Terpstra** ordered-trend across α ∈ {0.1, 0.5, 1.0, IID} per (model ×
    arm); Spearman secondary; BCa CI per α.
- **H5b (FedAvg suffers more)** — the α-induced drop is larger for FedAvg than for the
  pipeline.
  - H0: drop(FedAvg, α=0.1 vs IID) ≤ drop(pipeline-Krum, α=0.1 vs IID).
  - H1-alt: drop(FedAvg) > drop(pipeline) (one-sided), per model.
  - Test: **Wilcoxon signed-rank** on the per-seed paired drop difference (FedAvg − pipeline),
    Holm within the H5 family; Cliff's δ + BCa CI.
- **H5 verdict**: supported iff H5a trend holds **and** H5b shows FedAvg's degradation
  strictly larger with at least *small* δ. α = 0.5 is the reference-frame value (in T1a);
  {0.1, 1.0, IID} are the T2a sweep points.

---

## Supporting measured claims (reported with numbers, not hypothesis tests)

- **H-latency (GOOSE envelope)** — p50/p95/p99 inference latency (n=1000, warm-up 50) per
  model × precision (fp32, INT8) × device, reported against the **4 ms GOOSE transfer-time
  budget** (§3.6, `metrics/latency`). Measured on the single reference 3090, stated as such.
  A descriptive comparison to 4 ms, not a significance test.
- **H-consensus (overhead is a lower bound)** — PBFT message count = 2M(M−1) for a
  committing round (verified exactly by draw count) and simulated latency from the sourced
  utility-WAN lognormal (`simulation.network`, never the placeholder), swept over M∈{4,7,10}
  (T2d). Reported as a lower bound (honest primary, no view change; CONTRACT amendment i §3).
- **H-ledger (auditability)** — every committed round appends to the SHA-256 hash chain and
  `Ledger.verify()` holds end-to-end; a single-field tamper fails verification (property test).
  Qualitative contribution, not a p-value.

## Family / correction map (Holm applied within each family)

| Family | Members corrected together | Test |
|---|---|---|
| H1a parity | McNemar per seed + pooled | McNemar (exact) + TOST |
| H2 defense | every (attack × model × pipeline) cell vs no-defense | Wilcoxon + Holm; Friedman/Nemenyi over 5 arms |
| H3 evasion+damage | (model × pipeline) cells, Adaptive vs SignFlip | Wilcoxon + Holm |
| H4 LDP trend | the 3 τ curves | Jonckheere–Terpstra (Wilcoxon fallback) |
| H5 non-IID | (model × arm) trends + FedAvg-vs-pipeline drop | Jonckheere–Terpstra + Wilcoxon + Holm |

Deviations from this registration (any test swapped, margin changed, arm dropped) will be
recorded as a dated amendment at the top of this file with its reason — never a silent edit.
