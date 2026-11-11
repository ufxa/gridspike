# SOTA Report — phase 1 (2026-07-28)

Scope: adversarial novelty check of the six CONTRACT.md pillars against 2023–2026
literature; technical pins (SNN implementation, encoding, energy constants, OpenDSS
binding, FL engine, non-IID protocol, datasets); recommended contract amendments.
Method: web search (arXiv, IEEE, ACM, Elsevier, NDSS/USENIX, NREL), verification of
IDs/DOIs where retrievable. Every reference below was surfaced and cross-checked in
at least one independent source; where author lists could not be verified, they are
omitted rather than guessed.

**Bottom line**: the headline (pillar 6: federated SNN for substation-edge fault/anomaly
classification, inside a Byzantine-robust + LDP + PBFT + GOOSE-latency stack) survives.
No paper combines FL + SNN with power-system fault classification. But three pillars
need re-phrasing: the DP×Byzantine tension is theoretically known (PODC 2021), the
blockchain/PBFT+robust-aggregation combination is crowded (2023–2025), and the word
"measured" for SynOps energy is indefensible (it is analytic estimation). One branding
hazard: a 2023 MDPI Energies paper is already titled "FedGrid".

---

## (i) Per-pillar verdicts

### Pillar 1 — OpenDSS physics → non-IID federation by construction · **ADJUST**

No paper builds a *reproducible physics co-simulation* (OpenDSS feeders + labelled event
injection) whose per-client data is non-IID by feeder identity **and** Dirichlet skew, as
a shipped artifact. But "FL for power-system fault/disturbance classification on non-IID
data" as a sentence is taken; the claim must be the *benchmark generator*, not the task.

Closest references:
- **FeDiSa: A Semi-asynchronous Federated Learning Framework for Power System Fault and
  Cyberattack Discrimination** — Husnoo et al., arXiv:2303.16956 (2023). FL across grid
  zones distinguishing faults from cyberattacks on ICS/PMU-style datasets. *Differs*: no
  physics co-sim, no SNN, no Byzantine defense stack, data not generated per-substation.
- **FedDiSC: A Computation-efficient Federated Learning Framework for Power Systems
  Disturbance and Cyber Attack Discrimination** — Husnoo et al., arXiv:2304.03640 (2023).
  DP-SignSGD gradient privacy, deep autoencoder. *Differs*: same as above; DP is
  communication-oriented, no robustness/consensus.
- **Clustered Federated Learning for Generalizable FDIA Detection in Smart Grids with
  Heterogeneous Data** — arXiv:2507.14999 (2025). Handles grid data heterogeneity via
  clustering on IEEE bus systems. *Differs*: FDIA (measurement tampering) not physical
  fault classes; heterogeneity handled, not *constructed and released*.
- **Federated Learning for Smart Grid: A Survey on Applications and Potential
  Vulnerabilities** — arXiv:2409.10764 (2024). Maps the crowded FL-for-grid space;
  useful to cite for positioning.

Adjustment: claim = "first reproducible OpenDSS-based federated benchmark with
sample-accurate event labels and controllable two-axis non-IID (feeder identity ×
Dirichlet)". Do not claim "first FL for grid fault classification".

### Pillar 2 — Spiking classifier with SynOps energy accounting at matched accuracy · **ADJUST**

SNNs for grid disturbance classification exist (centralized, transmission-level, PMU);
SNN energy ratios via MAC/AC counting are standard practice. The novelty is *federated*
SNN on distribution-substation waveforms inside the robustness stack — i.e. pillar 2
only survives fused with pillar 6. Two wording problems: (a) "energy is measured" —
SynOps accounting is an analytic estimate (constants exclude memory traffic; recent
papers say so explicitly); say "estimated via SynOps accounting", reserve "measured"
for latency. (b) The ANN→SNN accuracy gap in this domain is small (SNNs reach parity
on PMU disturbance data), so frame "matched accuracy" as expected, not surprising.

Closest references:
- **Power System Disturbance Classification with Online Event-Driven Neuromorphic
  Computing** — Mahapatra, Lu, Sengupta, Ray Chaudhuri, IEEE Trans. Smart Grid (2021),
  arXiv:2006.06682. SNN on PMU signals, 16-machine NE-NY system, ~96.8% accuracy,
  energy argument via reduced MAC→AC operations. *Differs*: centralized, transmission
  PMU not substation waveforms, no FL, no formal pJ budget vs a latency envelope.
- **Exploring Spiking Neural Networks for Binary Classification in Multivariate Time
  Series at the Edge** — Ghawaly et al., arXiv:2510.20997 (2025). ORNL; evolved SNNs on
  multivariate streams (gamma-ray, EEG), 2 mW on microCaspian. *Differs*: not grid, not
  federated; useful as edge-SNN-for-time-series precedent.
- **A Survey of Encoding Techniques for Signal Processing in Spiking Neural Networks** —
  Auge et al., Neural Processing Letters (2021), DOI 10.1007/s11063-021-10562-2.
  Canonical encoding taxonomy (rate / latency / temporal-contrast a.k.a. delta).
- Note: the sizeable "spiking neural P systems" literature for power-grid fault
  *diagnosis* (fuzzy-reasoning membrane systems, 2014–2025, e.g. DOI 10.3390/en17225742)
  is symbolic reasoning, **not** gradient-trained SNNs — cite one and explicitly
  distinguish, or a careless reviewer will conflate them.

### Pillar 3 — Fidelity scoring → PBFT → Krum/geo-median, measured overhead · **CROWDED**

Blockchain/BFT + robust aggregation + FL is a dense 2023–2025 space, including
grid-flavored variants. An EAAI reviewer *will* call "we combined PBFT with Krum"
incremental. The defensible levers, none of which the competitors have together:
(1) an **adaptive attacker that knows the fidelity defense** (contract's
AdaptiveCosine) evaluated against the full pipeline; (2) consensus overhead *measured*
(messages, simulated latency) inside the FL loop and reported next to the GOOSE budget;
(3) the blockchain-vs-Krum-alone **ablation** answering "does the ledger buy accuracy
or only auditability?" — reviewers reward that honesty; (4) coupling with the LDP
tension curve (pillar 4).

Closest references:
- **FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping** — Cao, Fang,
  Liu, Gong, NDSS 2021, arXiv:2012.13995. Cosine-similarity trust scores vs a
  server-side root dataset; evaluated under adaptive attacks. *Differs*: our fidelity
  is cosine-to-coordinate-wise-median (no clean root dataset assumption — important in
  a multi-utility setting); no consensus layer, no grid domain. **Must cite.**
- **Krum Federated Chain (KFC)** — arXiv:2502.06917 (2025). Krum + blockchain (PoFL)
  against Byzantine and backdoor attacks. *Differs*: no PBFT overhead accounting, no
  fidelity pre-filter, no adaptive defense-aware attacker, not grid.
- **BRFL: A Blockchain-based Byzantine-Robust Federated Learning Model** — Li, Xia, Li,
  Wang, J. Parallel & Distributed Computing (2024), arXiv:2310.13403. Spectral
  clustering + correlation-based aggregator selection. *Differs*: no PBFT, no adaptive
  attacker, no application domain.
- **Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for
  Federated Learning** — Shejwalkar & Houmansadr, NDSS 2021. Origin of MinMax and the
  omniscient/non-omniscient attacker taxonomy our AdaptiveCosine must be placed in.
  Also **Local Model Poisoning Attacks to Byzantine-Robust Federated Learning** — Fang,
  Cao, Jia, Gong, USENIX Security 2020, arXiv:1911.11815 (first defense-aware
  optimization attacks on Krum/median). **Both must be cited**; our attack should be
  described as an instance of this family specialized to cosine-median fidelity.
- Foundations to cite, not claim: Krum — Blanchard et al., NeurIPS 2017; geometric
  median RFA — Pillutla, Kakade, Harchaoui, IEEE Trans. Signal Processing (2022),
  arXiv:1912.13445.

### Pillar 4 — LDP noise makes honest clients look Byzantine (trade-off curve) · **ADJUST**

The tension itself is *theoretically established* — claiming discovery is a desk-reject
risk. What is genuinely open: an **empirical LDP-σ × fidelity-threshold operating
curve** (false-rejection rate of honest clients vs attacker admission) in a concrete
application system. That framing is a real contribution; "the trade-off exists" is not.

Key references:
- **Differential Privacy and Byzantine Resilience in SGD: Do They Add Up?** —
  Guerraoui, Gupta, Pinot, Rouault, Stephan, PODC 2021, DOI 10.1145/3465084.3467919,
  arXiv:2102.08166. Proves classical DP + Byzantine-robust composition degrades with
  model dimension; the theoretical anchor for our pillar. **The paper our related-work
  section must lead with.**
- **Combining Differential Privacy and Byzantine Resilience in Distributed SGD** —
  same group, arXiv:2110.03991 (2021/22). Constructive follow-up.
- **DP-BREM: Differentially-Private and Byzantine-Robust Federated Learning with Client
  Momentum** — arXiv:2306.12608 (PoPETs 2024). Engineering both properties jointly via
  momentum; evidence the space is active. *Differs*: no per-client accept/reject
  consensus, no application, no operating-curve characterization.
- **PRECAD: Privacy-Preserving and Robust Federated Learning via Crypto-Aided
  Differential Privacy** — arXiv:2110.11578. Crypto-aided route around the tension;
  cite as alternative design point.

### Pillar 5 — Inference latency (fp32/INT8) vs the 4 ms GOOSE envelope · **NOVEL (narrow), cite adjacent IDS line**

Comparing ML inference latency against the 4 ms GOOSE figure exists — but only for
*packet-level intrusion detection*, never for waveform-level event classifiers, never
INT8-quantized, never SNN, never federated. Reporting p50/p95/p99 of fp32 vs INT8 vs
SNN classifiers against the protection envelope on edge-class hardware is unclaimed
territory. Nuance to handle in the manuscript: 4 ms is the commonly quoted GOOSE
transfer-time budget (IEC 61850-5 fast-message classes; stricter classes go to 3 ms) —
quote the standard's class table once, then use 4 ms as the envelope, as the IDS
literature does. Inference is one component of the end-to-end chain; say so explicitly
(we bound the *classifier's* share of the budget).

Closest references:
- **Anomaly Detection in IEC-61850 GOOSE Networks: Evaluating Unsupervised and Temporal
  Learning for Real-Time Intrusion Detection** — Moore, arXiv:2604.14233 (2026).
  RF/AE/RNN/LSTM/GRU on the ERENO GOOSE dataset; explicitly benchmarks per-prediction
  latency against 4 ms (RF fails at 21.8 ms; GRU 1.118 ms). *Differs*: network packets
  not electrical waveforms; no quantization, no SNN, no FL. **Most dangerous adjacency —
  cite and differentiate.**
- **Machine Learning-Based Intrusion Detection for Achieving Cybersecurity in Smart
  Grids Using IEC 61850 GOOSE Messages** — Symmetry 13(5):826 (2021),
  DOI 10.3390/sym13050826. Earlier GOOSE-message ML IDS; establishes the 4 ms framing.
- **Federated learning for cyber attack detection in protection schemes of
  cyber-physical energy systems** — ScienceDirect S2667305325001164 (2025). FL + IEC
  61850 protection context without latency-envelope accounting; shows the gap.

### Pillar 6 — Federated SNN for substation fault/anomaly classification (headline) · **NOVEL**

Adversarial searches ("federated spiking" × {smart grid, substation, power, smart
meter, load forecasting, electricity theft, photovoltaic}; FedSNN surveys; 2025–2026
FL-SNN papers) found **no FL+SNN work in the power/energy domain**. FL+SNN exists on
vision/neuromorphic benchmarks, audio, UAVs, V2X, and IIoT *network traffic* — not on
electrical measurements or grid events. The composite claim (FL + SNN + physics-based
non-IID grid data + Byzantine/PBFT stack + LDP curve + GOOSE latency + SynOps energy)
has no near neighbor. Framing rule: claim the *system in the domain*, never "first
FL+SNN" in general (2021 took that) and never "first robust FL+SNN" (2025 took that).

Closest references (the lineage to cite):
- **Federated Learning with Spiking Neural Networks** — Venkatesha, Kim, Tassiulas,
  Panda, IEEE Trans. Signal Processing 69:6183–6194 (2021),
  DOI 10.1109/TSP.2021.3121632, arXiv:2106.06579. First systematic FedSNN (BNTT,
  CIFAR, up to 100 clients; ~5.3× energy advantage claim). The origin paper.
- **Federated Neuromorphic Learning of Spiking Neural Networks for Low-Power Edge
  Intelligence** — Skatchkovsky, Jang, Simeone, arXiv:1910.09594 (ICASSP 2020).
  Online FL-SNN; the "federated neuromorphic" term.
- **The Robustness of Spiking Neural Networks in Federated Learning with Compression
  Against Non-omniscient Byzantine Attacks** — Nguyen, Zhao, Deng, Wu,
  arXiv:2501.03306 (2025). FL-SNN + top-κ sparsification vs MinMax; ~40% accuracy gain
  over FL-ANN under attack. *Differs*: image benchmarks; explicitly **non-omniscient**
  attackers (our adaptive attacker is defense-aware); no consensus/ledger, no DP, no
  domain, no latency/energy envelope.
- **Exploiting Label Skewness for Spiking Neural Networks in Federated Learning
  (FedLEC)** — Yu, Du, Jiang, Zhang, Deng, IJCAI 2025, arXiv:2412.17305. FL-SNN under
  label skew (+11.6% avg). Evidence non-IID FL-SNN is active; *differs*: algorithmic
  fix for skew, not a domain system; cite when defending Dirichlet design.
- **HFedSNN: Efficient Hierarchical Federated Learning using Spiking Neural Networks** —
  ACM MobiWac 2023, DOI 10.1145/3616390.3618288. 4.3× energy reduction vs FedANN;
  the energy-ratio-in-FL precedent to compare our numbers against.
- **A Robust and Energy-Efficient Federated IDS for IIoT Using SNNs** (Int. J. of
  Computer Science and Mathematics, 2026; Edge-IIoTset, Dirichlet α=0.3, differential
  evolution). Closest *spirit* competitor: FL+SNN for critical-infrastructure security —
  but on IIoT *network traffic*, no physics, no PBFT/ledger, no defense-aware attacker,
  no GOOSE/energy envelope. Verify the journal's indexing before citing prominently.

---

## (ii) Threats to novelty — single most dangerous paper per pillar

| Pillar | Paper | Why dangerous | Survival line |
|---|---|---|---|
| 1 | FeDiSa (arXiv:2303.16956) | "FL for power fault discrimination" sentence already exists | We ship the physics benchmark generator; they consume fixed ICS datasets |
| 2 | Mahapatra et al., IEEE TSG 2021 (arXiv:2006.06682) | SNN + grid disturbance + energy argument, in a top grid venue | Centralized, transmission PMU; no FL, no substation waveforms, no budgeted pJ/latency envelope |
| 3 | KFC (arXiv:2502.06917) | Krum + blockchain vs Byzantine/backdoor, 2025 | No PBFT cost accounting, no fidelity pre-filter, no defense-aware attacker, no domain |
| 4 | Guerraoui et al., PODC 2021 (DOI 10.1145/3465084.3467919) | The tension is *their theorem* | We contribute the empirical operating curve in a deployed-style grid system |
| 5 | Moore, arXiv:2604.14233 | Already benchmarks ML latency against 4 ms GOOSE | Packet IDS vs our waveform classifiers; no INT8/SNN/FL |
| 6 | Nguyen et al., arXiv:2501.03306 | FL + SNN + Byzantine robustness triple exists | Not grid, non-omniscient attacks only, no consensus/DP/energy-latency envelope |

Branding threat (not a pillar): **"FedGrid: A Secure Framework with Federated Learning
for Energy Optimization in the Smart Grid"**, Energies 16(24):8097 (2023),
DOI 10.3390/en16248097. Different task (energy/load optimization, no SNN, no
consensus), but the *name is taken in our exact venue space* (Elsevier/MDPI energy+FL).
Recommend retitling the paper's system name (repo name is harmless).

---

## (iii) Recommended pins

**SNN implementation — pure PyTorch LIF, arctan surrogate: VALIDATED.**
snnTorch's own reference (Eshraghian et al., *Training Spiking Neural Networks Using
Lessons From Deep Learning*, Proc. IEEE 111(9):1016–1054, 2023, arXiv:2109.12894)
documents exactly this recipe (hard-reset LIF + atan surrogate is snnTorch's default);
SpikingJelly (Fang et al., Science Advances 9(40):eadi1480, 2023,
DOI 10.1126/sciadv.adi1480) is the throughput-optimized alternative. For a ~2-layer,
hidden-256, T=8 classifier, a framework buys nothing and costs a dependency; our
custom SynOps counters and INT8 path are easier against raw modules. Papers are not
criticized for framework-free SNNs; they are criticized for *unvalidated gradients* —
keep the correctness bar (non-zero surrogate grads, hand-computed SynOps check), and
cite Eshraghian (method) + SpikingJelly (tooling context) in the implementation section.

**Encoding — rate + delta (temporal-contrast), ablate both: KEEP.**
Literature consensus (Auge et al. 2021; comm-systems SNN survey arXiv:2601.16550):
latency/TTFS coding minimizes SynOps but is noise-fragile; rate coding is noise-robust
but spike-hungry; delta/temporal-contrast suits slowly-varying signals with salient
transients — precisely RMS phasor windows around sags/faults. Contract's
`coding: {"rate","delta"}` matches the state of the art; report the ablation (energy
ratio will differ visibly between codings — that is a figure, not a risk).

**Energy constants — CONFIRMED; drop the "[SOTA to confirm]" tags.**
E_MAC = 4.6 pJ (32-bit float multiply 3.7 pJ + add 0.9 pJ), E_AC = 0.9 pJ, 45 nm CMOS:
M. Horowitz, "1.1 Computing's energy problem (and what we can do about it)", IEEE
ISSCC 2014, **DOI 10.1109/ISSCC.2014.6757323**. Still the standard constants in
2024–2026 SNN papers (LitE-SNN arXiv:2401.14652; TS-SNN arXiv:2505.04165; ANN2SNN
arXiv:2407.01645; many others). 7 nm variants circulate but are not the community
default; switching would *hurt* comparability. Two mandatory manuscript caveats used
by current papers: (a) constants cover arithmetic only — memory access and peripheral
energy excluded; (b) it is an estimate, not a hardware measurement. If INT8 energy is
also reported, use the same Horowitz table's 8-bit integer entries (add 0.03 pJ,
multiply 0.2 pJ → INT8 MAC ≈ 0.23 pJ) and label the table rows by precision.

**OpenDSS binding — OpenDSSDirect.py (DSS-Extensions): PIN.**
- `OpenDSSDirect.py` (dss-extensions): cross-platform (Windows/Linux/ARM — required:
  our experiment server is WSL2 Ubuntu), pip-installable, active (v0.9.x, releases
  through mid-2025), no COM. IEEE 13/34/37/123 and the IEEE PES European LV Test
  Feeder are available in the dss-extensions test-case tree (`electricdss-tst`).
- `py-dss-interface` (EPRI-adjacent, v2.3.0 2026): actively maintained but
  Windows/official-engine oriented — wrong fit for a Linux CI/server artifact.
- `dss-python`: same engine/team as OpenDSSDirect.py, lower-level API.
Manuscript note (their README requests it): results computed with the DSS-Extensions
alternative engine, not the EPRI-supported build; state the engine version.

**FL engine — native in-process: VALIDATED, with one sentence of justification.**
The Byzantine-FL canon (Fang USENIX'20, FLTrust NDSS'21, Shejwalkar NDSS'21 artifacts)
runs custom PyTorch loops, not frameworks; EAAI papers routinely do the same. Flower
(Beutel et al., arXiv:2007.14390) should be *cited* as the ecosystem standard with the
stated reason for not using it: bit-level determinism of the attack/consensus pipeline
under seeded RNG, which a distributed runtime cannot guarantee. Keep the optional
`flwr` adapter promise out of the paper (mention in repo README only) — promising it
in the manuscript invites a reviewer request to deliver it.

**Non-IID protocol — Dirichlet α=0.5 over K=10: DEFENSIBLE; add an α sweep.**
Dirichlet label skew is the standard protocol (Hsu, Qi, Brown, arXiv:1909.06335,
2019); α=0.5 is a conventional "moderately severe" setting. Our two-axis design
(feeder/load-shape identity = natural feature skew, + Dirichlet = label skew) is
*stronger* realism than Dirichlet-only and worth a sentence. Recommended: report
α ∈ {0.1, 0.5, 1.0} in the ablation battery (the closest FL-SNN IDS competitor used
α=0.3; a sweep pre-empts "why 0.5?"). K=10 substations is within community norms for
cross-silo FL (5–20); justify as cross-silo (utilities), not cross-device.

**Datasets — simulation is justified; cite SMART-DS to pre-empt the reviewer.**
No public dataset provides per-substation labelled fault waveforms suitable for a
K-client federation — that gap *is* the justification for pillar 1. Expect the
question "why not SMART-DS?": answer by citing **NREL SMART-DS** (Synthetic Models for
Advanced, Realistic Testing: Distribution Systems and Scenarios; OpenDSS models for
SFO/GSO/AUS, timeseries loads 2016–2018; OEDI submission 2981 / data.openei.org) as
(a) the realism reference for synthetic feeders and (b) the declared scale-up path —
our pipeline runs unmodified on SMART-DS feeders. IEEE 13/34/37/123 + EU LV remain the
paper's feeders (universally recognized, small enough for multi-seed batteries).
Cite ERENO (the IEC 61850 dataset generator used by the GOOSE IDS line) when
differentiating pillar 5.

---

## (iv) Recommended contract amendments

1. **Energy wording (Architectural thesis §2, metrics/energy.py)** — replace "energy
   per inference is *measured* via SynOps accounting" with "*estimated* via SynOps
   accounting (Horowitz ISSCC 2014 constants; arithmetic-only, excludes memory)".
   Delete both `[SOTA to confirm]` tags; constants confirmed (DOI
   10.1109/ISSCC.2014.6757323). Optionally add INT8 constants (0.03/0.2 pJ add/mult)
   to `metrics/energy.py` for the INT8 row of the energy table.
2. **Pillar 4 phrasing** — from "the LDP-noise × fidelity-threshold trade-off curve is
   a contribution" to "the *empirical* LDP-σ × fidelity-threshold operating curve
   (honest-client false-rejection vs attacker admission) in a grid FL system is a
   contribution; the tension itself is established theory (Guerraoui et al., PODC
   2021 — cite)".
3. **Pillar 3 phrasing** — do not claim novelty for PBFT+Krum as a combination; the
   claims are (a) defense-aware AdaptiveCosine attacker evaluated end-to-end, (b)
   measured consensus overhead vs the latency envelope, (c) the blockchain-vs-
   Krum-alone ablation. Require AdaptiveCosine to be evaluated against *both* Krum and
   geometric median (Shejwalkar-style per-defense tailoring), else reviewers will.
4. **System/paper name** — "FedGrid" is taken (Energies 2023, DOI 10.3390/en16248097).
   Keep the repo name if convenient, but retitle the *published system* (candidates:
   SpikeGrid-FL, NeuroSubstation, FedSpike-61850 — venue fit decision).
5. **Pillar 6 phrasing** — claim scope "first federated *spiking* framework for
   substation fault/anomaly classification with Byzantine robustness and an IEC 61850
   latency/energy envelope"; explicitly cite Venkatesha 2021 (first FedSNN) and
   Nguyen 2025 (first Byzantine-robust FedSNN) as the lineage being specialized.
6. **Pins into contract** — OpenDSSDirect.py (DSS-Extensions engine, version stated);
   pure-PyTorch LIF confirmed (house precedent validated); native FL engine confirmed
   (cite Flower as the declined alternative); α sweep {0.1, 0.5, 1.0} added to the
   phase 4 battery; SMART-DS citation added as scale-up path.
7. **No change** needed to the data contract, module boundaries, or wave plan.

---

## Reference list (verified identifiers)

1. Horowitz, "1.1 Computing's energy problem (and what we can do about it)", IEEE ISSCC 2014. DOI 10.1109/ISSCC.2014.6757323
2. Venkatesha, Kim, Tassiulas, Panda, "Federated Learning with Spiking Neural Networks", IEEE TSP 69:6183–6194, 2021. DOI 10.1109/TSP.2021.3121632; arXiv:2106.06579
3. Skatchkovsky, Jang, Simeone, "Federated Neuromorphic Learning of Spiking Neural Networks for Low-Power Edge Intelligence", 2019/2020. arXiv:1910.09594
4. "HFedSNN: Efficient Hierarchical Federated Learning using Spiking Neural Networks", ACM MobiWac 2023. DOI 10.1145/3616390.3618288
5. Yu, Du, Jiang, Zhang, Deng, "Exploiting Label Skewness for Spiking Neural Networks in Federated Learning" (FedLEC), IJCAI 2025. arXiv:2412.17305
6. Nguyen, Zhao, Deng, Wu, "The Robustness of Spiking Neural Networks in Federated Learning with Compression Against Non-omniscient Byzantine Attacks", 2025. arXiv:2501.03306
7. Mahapatra, Lu, Sengupta, Ray Chaudhuri, "Power System Disturbance Classification with Online Event-Driven Neuromorphic Computing", IEEE Trans. Smart Grid, 2021. arXiv:2006.06682
8. Husnoo et al., "FedDiSC: A Computation-efficient Federated Learning Framework for Power Systems Disturbance and Cyber Attack Discrimination", 2023. arXiv:2304.03640
9. Husnoo et al., "FeDiSa: A Semi-asynchronous Federated Learning Framework for Power System Fault and Cyberattack Discrimination", 2023. arXiv:2303.16956
10. Cao, Fang, Liu, Gong, "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping", NDSS 2021. arXiv:2012.13995
11. Fang, Cao, Jia, Gong, "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning", USENIX Security 2020. arXiv:1911.11815
12. Shejwalkar, Houmansadr, "Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning", NDSS 2021 (ndss-symposium.org paper 24498)
13. Blanchard, El Mhamdi, Guerraoui, Stainer, "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent" (Krum), NeurIPS 2017
14. Pillutla, Kakade, Harchaoui, "Robust Aggregation for Federated Learning" (RFA, geometric median), IEEE TSP 70:1142–1154, 2022. arXiv:1912.13445
15. Guerraoui, Gupta, Pinot, Rouault, Stephan, "Differential Privacy and Byzantine Resilience in SGD: Do They Add Up?", PODC 2021. DOI 10.1145/3465084.3467919; arXiv:2102.08166
16. "Combining Differential Privacy and Byzantine Resilience in Distributed SGD", 2021/22. arXiv:2110.03991
17. "DP-BREM: Differentially-Private and Byzantine-Robust Federated Learning with Client Momentum", PoPETs 2024. arXiv:2306.12608
18. "Krum Federated Chain (KFC): Using blockchain to defend against adversarial attacks in Federated Learning", 2025. arXiv:2502.06917
19. Li, Xia, Li, Wang, "BRFL: A Blockchain-based Byzantine-Robust Federated Learning Model", J. Parallel Distrib. Comput. arXiv:2310.13403
20. Moore, "Anomaly Detection in IEC-61850 GOOSE Networks: Evaluating Unsupervised and Temporal Learning for Real-Time Intrusion Detection", 2026. arXiv:2604.14233
21. "Machine Learning-Based Intrusion Detection for Achieving Cybersecurity in Smart Grids Using IEC 61850 GOOSE Messages", Symmetry 13(5):826, 2021. DOI 10.3390/sym13050826
22. Auge et al., "A Survey of Encoding Techniques for Signal Processing in Spiking Neural Networks", Neural Processing Letters, 2021. DOI 10.1007/s11063-021-10562-2
23. Eshraghian et al., "Training Spiking Neural Networks Using Lessons From Deep Learning", Proc. IEEE 111(9):1016–1054, 2023. arXiv:2109.12894 (snnTorch)
24. Fang et al., "SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence", Science Advances 9(40):eadi1480, 2023. DOI 10.1126/sciadv.adi1480
25. Beutel et al., "Flower: A Friendly Federated Learning Research Framework", 2020. arXiv:2007.14390
26. Hsu, Qi, Brown, "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification", 2019. arXiv:1909.06335
27. NREL, "SMART-DS: Synthetic Models for Advanced, Realistic Testing — Distribution Systems and Scenarios" (OpenDSS models, SFO/GSO/AUS). OEDI submission 2981, data.openei.org/submissions/2981
28. "FedGrid: A Secure Framework with Federated Learning for Energy Optimization in the Smart Grid", Energies 16(24):8097, 2023. DOI 10.3390/en16248097 (name collision)
29. Ghawaly et al., "Exploring Spiking Neural Networks for Binary Classification in Multivariate Time Series at the Edge", 2025. arXiv:2510.20997
30. "Clustered Federated Learning for Generalizable FDIA Detection in Smart Grids with Heterogeneous Data", 2025. arXiv:2507.14999
