# GridSpike

Code, data and every experimental result behind the manuscript
**"GridSpike: Defense-aware stress-testing of federated edge intelligence for event classification in IEC 61850
substations"** (submitted to *Engineering Applications of Artificial Intelligence*).

GridSpike is a federated-learning framework and open benchmark for event classification on substation telemetry.
It has four parts:

- an OpenDSS quasi-static generator for non-IID, labelled per-substation windows: five feeders, ten clients,
  feeder × Dirichlet label skew;
- three edge classifiers (CNN-LSTM, a surrogate-gradient spiking network, a selective state-space/Mamba model);
- a fail-closed admission pipeline: cosine fidelity gate, simulated PBFT agreement, Krum or geometric-median
  aggregation, SHA-256 hash-chained ledger;
- a pre-registered experiment battery with non-parametric statistics.

The classifier provides event awareness only and takes no part in protection (trip) decisions. All synthetic data
are simulated.

## Repository layout

| Path | Content |
|---|---|
| `paper/` | Manuscript (`main.tex`, `main.pdf`), supplementary material (`supplementary.tex`, `supplementary.pdf`), separate title page, the 21 generated tables (`tables/`), figures (`figures/`) and the script that builds Fig. 1 (`scripts/`) |
| `src/fedgrid/` | Python package: simulation (OpenDSS feeders in `simulation/feeders/`), models, federated engine, attacks, consensus/ledger, metrics |
| `experiments/` | Single-run executor (`run_one.py`), table generator (`report.py`), statistics (`analyze.py`), figures, manifests (`manifest*.jsonl`, one line per pre-registered run), manifest generators, EMT validation (`emt_routeb.py`), distributed runner (`fleet/`) |
| `tests/` | Test suite |
| `data_battery/` | The pre-generated OpenDSS federations used by the battery (`fed_*/client_*.npz` + `manifest.json`) |
| `runs*/` | One JSON `RunResult` per run (configuration, seed, per-round events, metrics, environment fingerprint) |
| `docs/` | Design notes, realism audit, validation standard, statistical analysis (`X3_ANALYSIS.md`), EMT design (`EMT_ROUTE_B.md`) |

### Which run folders are the paper corpus

The paper's tables come from five folders, merged in **increasing precedence**. On the same `run_id`, or on the same
seeded configuration, the later folder wins (`CORPUS_DIRS` in `experiments/report.py`):

```
runs_snn_fixed  →  runs  →  runs_x3  →  runs_ssm_x3  →  runs_gap_x3
```

After two-pass deduplication this gives 4,851 distinct seeded runs from 6,836 files. 15 seeds per Tier-1 and temporal
cell; 9 per Tier-2/Tier-3 cell. Only the blocks `t1a_temporal` and `t1b_temporal` use the chronological
(leakage-controlled, headline) split; every other block uses the pre-registered random split.

Other folders:

- `runs_real/`, `runs_real_v2/`: sanity check on public MSU/ORNL traces (results only).
- `runs_emt/`: EMT-domain validation with DPsim (450 simulations, IEEE 13-node replica); see `docs/EMT_ROUTE_B.md`.
- `runs_ceiling_fixed/`, `runs_rate_fixed/`: auxiliary probes. The centralized ceilings in `runs_ceiling_fixed/`
  (and seeds 0–4 of `t1a-centralized-*` in `runs/`) use the earlier unweighted loss. The paper uses the
  `effective_num` re-runs in `runs_x3/` and `runs_ssm_x3/`, which take precedence.

## Reproducing

Python ≥ 3.10 (the battery ran on 3.10–3.12). OpenDSS through `OpenDSSDirect.py`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch            # or the CUDA wheel for your GPU
pip install -e ".[dev]"
python -m pytest tests/ -q
```

Regenerate the result tables and `runs/summary.json` from the corpus:

```bash
python -m experiments.report --runs-dir runs_snn_fixed runs runs_x3 runs_ssm_x3 runs_gap_x3
```

Run a single configuration from a manifest line with `experiments/run_one.py` (see its docstring). Bit-reproducibility
is claimed on CPU only; GPU runs are seeded but not bit-identical across drivers.

The table files in `paper/tables/` were produced by `report.py` and then given a uniform layout by hand. Regenerating
them reproduces every value, but not that layout.

Build the PDFs (TeX Live or MiKTeX with `elsarticle`):

```bash
cd paper && make        # supplementary -> supplabels -> main -> titlepage
```

Fig. 1 is built by `paper/scripts/figura1_drawio.py`, which reads every number from `paper/scripts/numeros_fig1.csv`
(source and evidence label per value). It writes `figures/fig1_overview.drawio`; export it to PNG with diagrams.net.

## Licence

- Code (`src/`, `experiments/`, `tests/`, `paper/scripts/`): MIT, see [`LICENSE`](LICENSE).
- Manuscript, figures, tables, documentation, generated federations and run results: CC BY 4.0, see
  [`LICENSE-CONTENT.md`](LICENSE-CONTENT.md).
- Third-party material keeps its own terms: the IEEE PES test feeders in `src/fedgrid/simulation/feeders/`, and the
  photo and icons of Fig. 1 (`paper/figures/fig1_parts/fontes/CREDITOS.md`).

## Citation

Citation details will be added after review.
