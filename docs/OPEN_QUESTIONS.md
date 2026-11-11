# Open questions

**Superseded by the manuscript as of 2026-07-31; see sections 6, 7, 9 (as
applicable).**

Nearly everything this file tracked has been answered by the completed
experiment battery and the finished manuscript, and the answers now live in
`paper/sections/`. The stale body has been removed rather than left to
contradict the paper. What follows is only what is *still open*, each with a
pointer to the manuscript section that discusses it (or a note that the
manuscript does not, which is itself the open question). Manuscript-scoped items
live in `paper/OPEN_QUESTIONS_PAPER.md`. Per the contract's house rules, remove
an entry only when it is resolved, and link the resolution.

---

## Federated learning / privacy (the federated-learning module)

- **NEW, 2026-07-31 — the parameter exchange is `model.parameters()` only, and
  the CNN-LSTM is *not* buffer-free.** `fl/engine.py` exchanges
  `[p for p in model.parameters()]` (`_get_params` / `_set_params`, engine.py
  ~328–345) and the old note here asserted "contract models are buffer-free;
  fine for current models". That is **false for `CNNLSTM`**, which has two
  `nn.BatchNorm1d` layers and therefore six buffers
  (`conv.1/conv.5.running_mean`, `.running_var`, `.num_batches_tracked`).
  Consequence: BatchNorm running statistics are **never aggregated** — each
  client evaluates the shared global weights through its own local normalization
  statistics, which under the non-IID partition are genuinely different per
  client. This is not a crash and does not invalidate any run; it is an
  undisclosed methodological detail. Section 4 describes the block as "1-D
  convolution, batch normalization, ReLU" and says nothing about what is
  exchanged. **Decide: disclose it (Section 4 and/or the Section 9 limitation
  list), or extend the exchange to a `state_dict` form.** The spiking and
  state-space models are unaffected.

## Consensus / ledger (the consensus/attacks module)

- **NEW, 2026-07-31 — `Ledger.verify()` does not detect tail truncation.**
  Confirmed by construction against `src/fedgrid/consensus/ledger.py`: an edit,
  a reordering and an *interior* deletion each break recomputation, but dropping
  the last k blocks (up to and including all of them) leaves a valid shorter
  chain and `verify()` returns `True`. So does truncate-then-re-append. Nothing
  stored commits to the chain's length or head. The docstrings in `ledger.py` now
  state this precisely and name the missing ingredient (a signed head committing
  to record count + final digest); `verify()` accepts optional `expected_len` /
  `expected_head` (defaults preserve the old behaviour) and
  `tests/test_consensus.py` pins both the gap and the optional check.
  **Still open:** (a) the reproducibility paragraph of
  `paper/sections/06_experimental_design.tex` (~line 279; search the text, the
  file is under active edit) still says "a deletion anywhere in the chain fails
  recomputation" and must be replaced; (b) a real
  fix — validator-signed heads or an external anchor — is future work, and
  Section 8 already names it as such.
- **End-to-end ledger verification is not instrumented in any `RunResult`.**
  Registered as a supporting claim in Section 6, reported as *not measured*: the
  Ledger column of `tab_defense_decomposition` reads `--` throughout, and
  Section 7 (“Consensus overhead”), 8 and 9 all say auditability is a design property with
  property-test coverage rather than a battery result. Instrumenting it is a
  small runner change plus a re-run.

## API / frontend (the API module)

- **The console's "CHAIN VERIFIED" badge is a view-side FNV-64 chain**
  (`frontend/src/lib/ledger.ts`, `frontend/src/components/Ledger.tsx`), not the
  authoritative backend SHA-256 ledger. The component's own caption says so, and
  Section 4 claims no more than "the ledger with a verification badge", so the
  manuscript is not wrong — but the released artifact's badge does not display
  the real chain. Wire the real ledger head into `RoundEvent` or add a ledger
  endpoint. Note also that the badge, like `Ledger.verify()` itself, shows only
  structural self-consistency and cannot see a truncated tail.
- **The fidelity threshold τ is not on the wire.** The attack panel infers it
  from the accepted/rejected score split. Adding τ to `RoundEvent.consensus`
  would make it exact.

## Artifact polish (before release)

- **`RunResult.method` records the internal strategy class name.**
  `fl/engine.py:417` sets `method = type(strategy).__name__.lower()`, so the
  released run records contain `_krumstrategy` (120 runs) and `_geomedianstrategy`
  (156 runs) instead of the arm's public label (`krum`, `geomedian`). Cosmetic,
  but visible in the released artifact and in any table built naively from
  `method`. No table is affected: `experiments/report.py:353` reads the arm from
  the manifest (`manifest["arm"]`) and only falls back to `method`. Still,
  `experiments/build.py` should set a clean label at the source before the code
  is published.
- **Ops note — OpenDSS on Windows raises a caught Free-Pascal SEH exception**
  (code `0xe0465043`) on import / first solve, which `faulthandler` would dump to
  stderr. It is **non-fatal**: the library works and all tests pass.
  `tests/test_simulation.py` disables `faulthandler` around the import and
  restores it, so test output stays clean. Kept here because it will otherwise
  alarm the next person who sees it. Linux is unaffected, and the battery ran
  against pre-generated federations so `run_one` never touched OpenDSS in-process.
- **Standing caveat — matched-accuracy McNemar (H1a) pairing relies on stable
  global-test ordering.** `RunResult` carries no per-sample predictions (a schema
  gap, not a bug), so `experiments/build.py` attaches `final["correct"]` — a
  per-sample 0/1 correctness vector over the deterministically-ordered global
  test — and two models at a shared seed pair sample-for-sample. This relies on
  `Federation.global_test()` yielding a stable order across models at a fixed
  seed (true for the current `ConcatDataset` construction). If a future model
  introduces nondeterministic eval ordering, this pairing must switch to explicit
  sample ids.

---

## Resolved since the last revision (kept as a pointer, not as an open item)

- **SNN fp32 inference latency** — measured, not pending: p99 **16.07 ms** CPU
  and **16.93 ms** CUDA at batch 1 (`runs/latency-snn-fp32.json`), i.e. ~4× the
  4 ms IEC 61850 fast-message envelope on either device. Section 7 (“Latency against the GOOSE envelope”). The old
  "≈ 4.38 ms CPU / 13.5 ms CUDA, sits at the GOOSE edge" figures here were wrong
  and are gone. **INT8 was attempted and could not be produced**: torch's dynamic
  INT8 kernels reject the recurrent and state-space packed parameters, so
  `runs/latency-*-int8.json` carry empty `latency` blocks, no model in the paper
  is quantized, and the INT8 energy rows are an analytic re-costing. Sections 6,
  7 (“Latency against the GOOSE envelope”) and 9.
- **Delta coding threshold** — 0.2, swept on validation; now the model default
  (`models/snn.py:173`) and reported as the single swept hyper-parameter in
  Section 6. The `FAULT_LG` honesty flag was honoured: `tab_per_class` reports
  support 25 with SNN recall 0.196.
- **Macro-F1 definition** — confirmed and stated in the paper: Section 6 says macro-F1 "averages over classes *present in the evaluation labels*",
  which is the definition the tables report.
- **PBFT network-latency placeholders** — every pipeline arm builds its
  validator with the sourced utility-WAN lognormal
  (`make_validator(..., net_ms=utility_wan_net_ms)`), never `pbft.default_net_ms`.
- **PBFT overhead is a lower bound** — now stated in the manuscript itself
  (Sections 4, 6, the table caption, 8, 9), not just here.
- **`LabelFlip` handshake between the consensus/attacks module and the federated-learning module** — reconciled. The
  realized threat model is dynamic per-batch label noise at rate `frac`;
  `attacks/suite.py` documents it, `LabelFlip.flip` is labelled a static
  reference implementation retained for unit tests and off the federated path,
  and the manuscript describes the dynamic attack in Section 4 ("Label-flip is
  applied dynamically, per batch"), as a dated pre-registration amendment in
  Section 6, in the attack-grid footnote, and as a Section 9 limitation.
- **phase 4 data staging** — multi-federation layout implemented
  (`experiments/stage_data.py` pre-generates each battery federation into
  `<root>/<fed_id>/`; `load_or_make_federation` picks the matching subdir), so
  workers do zero in-process generation.
- **Published-system name** — `GridSpike` (`paper/macros.tex:33`); the repo and
  package stay `fedgrid`. Residual pre-submission re-check tracked in
  `paper/OPEN_QUESTIONS_PAPER.md`.
- **Local-floor evaluation bug** — fixed; `_SingleClientFederation.global_test()`
  now evaluates on the federation-wide test set, the 10 stale runs are
  quarantined in `runs_invalid/`, and the 15 post-fix `t1a-local-*` runs (three
  models × 5 seeds) are in `runs/`. The floor now sits below every federated arm
  (CNN-LSTM 0.298, spiking 0.130, Mamba 0.202).
- **"Centralized ceiling does not dominate"** — resolved by the re-run: the
  ceiling now leads its federated arms on all three models (CNN-LSTM 0.666 vs
  FedAvg 0.620; spiking 0.700 vs 0.161; Mamba 0.916 vs 0.758), and Section 7 (“Clean reference frame”)
  qualifies the ceiling row explicitly (the centralized arms use a more generous
  recipe; the matched-recipe ceilings are also reported, and the spiking ceiling
  is further shown to be largely leakage under the temporal control).
- **"Federated SNN collapses"** — no longer an open question but a reported
  finding: it is one of the paper's headline results, quantified in Section 7 (“Clean reference frame”)
  and interpreted in Section 8, and the spiking model's failure under parameter
  averaging is stated rather than smoothed.
