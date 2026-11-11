# FedGrid — Architecture

Authoritative interfaces live in [`CONTRACT.md`](../CONTRACT.md); this file is the
orientation map. Root package: `fedgrid` (src layout).

## Four layers

```
┌────────────────────────────────────────────────────────────────────────────┐
│ 4. Observability   fedgrid.api (FastAPI + WS)  ·  fedgrid.metrics          │
│                    frontend/ (React console)      (energy, latency, runs)  │
├────────────────────────────────────────────────────────────────────────────┤
│ 3. Robust layer    fedgrid.consensus (fidelity → PBFT → Krum/geo-median,   │
│                    ledger)  ·  fedgrid.attacks (threat models)             │
├────────────────────────────────────────────────────────────────────────────┤
│ 2. Learning        fedgrid.models (CNN-LSTM, spiking LIF)                  │
│                    fedgrid.fl (engine, FedAvg/FedProx, LDP)                │
├────────────────────────────────────────────────────────────────────────────┤
│ 1. Grid physics    fedgrid.simulation (OpenDSS feeders, event injection,   │
│                    non-IID partition)  ·  fedgrid.data (data contract)     │
└────────────────────────────────────────────────────────────────────────────┘
```

Dataflow: OpenDSS feeders produce labelled windows `(C=11, T=64)` per substation;
K=10 non-IID clients train locally (CNN-LSTM or SNN, optional LDP noise); updates
pass fidelity scoring, PBFT consensus and robust aggregation into the new global
model; a hash-chained ledger records each round; the API streams `RoundEvent`s to
the console.

## Module ownership (one owner per file — never edit another owner's file)

| Files | Owner | Purpose |
|---|---|---|
| `src/fedgrid/simulation/*`, `src/fedgrid/data.py` | the data-simulation module (phase 2) | OpenDSS co-sim, event injection, windowing, non-IID federation |
| `src/fedgrid/models/*`, `src/fedgrid/metrics/*` | the models/metrics module (phase 2) | CNN-LSTM, spiking classifier, SynOps/energy, latency stats, RunResult |
| `src/fedgrid/fl/*` | the federated-learning module (phase 2) | federated engine, FedAvg/FedProx, LDP |
| `src/fedgrid/consensus/*`, `src/fedgrid/attacks/*` | the consensus/attacks module (phase 2) | fidelity, Krum/geo-median, PBFT sim, ledger, threat models |
| `src/fedgrid/api/*`, `frontend/*` | the API module (phase 2) | FastAPI + WebSocket, React console |
| `experiments/`, integration glue | experiment runner | experiment runner, figures |

## The six claims (what the paper argues)

1. **Grid physics**: OpenDSS feeders generate labelled multivariate electrical time
   series per substation, non-IID by construction (distinct feeders/load shapes +
   Dirichlet class skew).
2. **Two classifiers, federated**: a CNN-LSTM reference and a spiking classifier
   (LIF, pure PyTorch) whose energy per inference is *measured* via SynOps
   accounting — headline claim is the energy ratio at matched accuracy.
3. **Validation before aggregation**: fidelity scoring → simulated PBFT consensus →
   robust aggregation (Krum / geometric median), overhead *measured*, tolerating
   `f = (M-1)/3` Byzantine clients including an adaptive attacker that knows the
   defense.
4. **Privacy tension quantified**: local DP makes honest clients look Byzantine;
   the LDP-noise x fidelity-threshold trade-off curve is a contribution.
5. **IEC 61850 envelope with numbers**: inference latency p50/p95/p99 (float32 and
   INT8) against the 4 ms GOOSE transfer-time budget.
6. **The framework is the artifact**: backend + live console + experiment runner
   ship open-source; every paper number is regenerable from this repo.

Nothing in Results may come from citation.

## Wave plan

1. **phase 1** — repo scaffold (stubs + green smoke tests + `/health` + frontend
   shell builds); server env; SOTA verification → contract amendments.
2. **phase 2** — five module owners implement + tests.
3. **Phase 3** — experiment-runner integration: experiment runner, live console E2E,
   first end-to-end federated run with attacks on the server.
4. **phase 4** — experiment battery (multi-seed, baselines, ablations, LDP sweep,
   adaptive attacker) on the server.
5. **phase 5** — INT8 + latency vs GOOSE budget, energy tables, figures, manuscript,
   reviewer panel.
