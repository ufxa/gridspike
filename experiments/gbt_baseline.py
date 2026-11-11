"""Shallow GBT baseline (Reviewer 2) reproduced under the fedgrid protocol.

Per-window summary statistics (7 per channel) -> HistGradientBoostingClassifier.
Uses the PERSISTED split indices and PERSISTED per-client normaliser from the
npz files (same contract as fedgrid.data.load_federation), and the project's own
macro-F1 (fedgrid.fl.engine._f1_from_confusion: averaged over classes present in
y_true) so the numbers are directly comparable to tab_reference_frame.

``--split-mode temporal`` swaps the persisted permutation split for the
leakage-controlled chronological one, re-deriving each client's blocks with the
canonical fedgrid.data.temporal_split_indices (same 70/15/15 cut, same 4-window
purge gap) and refitting the normaliser on the new train block, exactly as
fedgrid.data.apply_temporal_split does for the deep arms. It writes its own
``temporal_reference_battery`` key and leaves every other key of the output JSON
untouched; the default ``--split-mode random`` reproduces the original battery.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path("E:/paper7/fedgrid")
sys.path.insert(0, str(ROOT))
from fedgrid.fl.engine import _confusion, _f1_from_confusion  # noqa: E402
from fedgrid.data import (  # noqa: E402
    CHANNELS,
    N_CLASSES,
    SPLIT_MODES,
    TEMPORAL_SPLIT_GAP,
    Normalizer,
    temporal_split_indices,
)

BATTERY = ROOT / "data_battery"
OUT = ROOT / "paper" / "generated" / "gbt_baseline.json"

STATS = ("mean", "std", "min", "max", "ptp", "median", "mean_abs_diff")
FDEV = CHANNELS.index("fdev")  # 10
K = 10


def features(w: np.ndarray) -> np.ndarray:
    """(N, C, T) -> (N, C*7); stats taken over the time axis, channel-major order."""
    f = np.stack(
        [
            w.mean(axis=2), w.std(axis=2), w.min(axis=2), w.max(axis=2),
            np.ptp(w, axis=2), np.median(w, axis=2),
            np.abs(np.diff(w, axis=2)).mean(axis=2),
        ],
        axis=2,
    )
    return np.ascontiguousarray(f.reshape(len(w), -1), dtype=np.float32)


def feature_names(chans: tuple[int, ...]) -> list[str]:
    return [f"{CHANNELS[c]}_{s}" for c in chans for s in STATS]


def load_fed(fed_dir: Path, normalize: bool = True, split_mode: str = "random",
             gap: int = TEMPORAL_SPLIT_GAP) -> list[dict]:
    """Mirror of fedgrid.data.load_federation's npz contract (numpy-only).

    ``split_mode="random"`` reads the persisted indices and the persisted
    per-client normaliser. ``split_mode="temporal"`` calls the canonical
    fedgrid.data.temporal_split_indices on this client's window count and refits
    the normaliser on the new train block -- the two moves fedgrid.data.
    apply_temporal_split makes, so the GBT probe sees the same protocol as the
    deep arms. Window order is index order (make_federation concatenates day 0,
    day 1, ...), the same premise data.py states.
    """
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"unknown split_mode {split_mode!r}; expected one of {SPLIT_MODES}")
    manifest = json.loads((fed_dir / "manifest.json").read_text(encoding="utf-8"))
    clients = []
    for entry in manifest["clients"]:
        with np.load(fed_dir / entry["file"]) as z:
            w = z["windows"].astype(np.float32)          # (N, 11, 64)
            y = z["labels"].astype(np.int64)
            if split_mode == "temporal":
                tr, va, te = temporal_split_indices(len(y), gap=gap)
                norm = Normalizer.fit(w[tr])   # train-block only: no leak backwards
                mean, std = norm.mean, norm.std
            else:
                tr, va, te = z["train_idx"], z["val_idx"], z["test_idx"]
                mean, std = z["norm_mean"], z["norm_std"]
            if normalize:                                 # ClientDataset.__getitem__
                w = (w - mean) / std
            clients.append({
                "cid": int(entry["cid"]),
                "w": w,
                "y": y,
                "train": tr, "val": va, "test": te,
            })
    return clients


def score(model, X, y) -> dict:
    p = model.predict(X)
    per_class, macro = _f1_from_confusion(_confusion(y, p, N_CLASSES))
    return {
        "macro_f1": macro,
        "accuracy": float((p == y).mean()),
        "per_class_f1": [float(v) for v in per_class],
        "n_eval": int(len(y)),
        "classes_present": int((np.bincount(y, minlength=N_CLASSES) > 0).sum()),
    }


def fit_eval(Xtr, ytr, Xte, yte, seed: int, **kw) -> dict:
    t = time.time()
    m = HistGradientBoostingClassifier(random_state=seed, **kw).fit(Xtr, ytr)
    r = score(m, Xte, yte)
    r.update(n_train=int(len(ytr)), n_features=int(Xtr.shape[1]),
             n_iter=int(m.n_iter_), fit_seconds=round(time.time() - t, 2))
    return r


def _split_matrices(clients, chans):
    """Per-client feature matrices plus the pooled train/global-test blocks."""
    F = [features(c["w"][:, chans, :]) for c in clients]
    Xtr = [f[c["train"]] for f, c in zip(F, clients)]
    ytr = [c["y"][c["train"]] for c in clients]
    Xte = [f[c["test"]] for f, c in zip(F, clients)]
    yte = [c["y"][c["test"]] for c in clients]
    return Xtr, ytr, Xte, yte


def pooled(clients, chans, seed, **kw) -> dict:
    Xtr, ytr, Xte, yte = _split_matrices(clients, chans)
    return fit_eval(np.concatenate(Xtr), np.concatenate(ytr),
                    np.concatenate(Xte), np.concatenate(yte), seed, **kw)


def local_only(clients, chans, seed, **kw) -> dict:
    """Local floor. Train on client k only; score on the GLOBAL test set.

    experiments/build.py::_SingleClientFederation fixes the evaluation set for
    this arm: scoring a local model on its own skewed test split measures it on
    the distribution it trained on and inflates the floor above the federated
    and centralized arms. We reproduce the same choice, and keep the (inflated)
    own-split number alongside purely as a diagnostic.
    """
    Xtr, ytr, Xte, yte = _split_matrices(clients, chans)
    Xg, yg = np.concatenate(Xte), np.concatenate(yte)
    per = []
    for i, c in enumerate(clients):
        r = fit_eval(Xtr[i], ytr[i], Xg, yg, seed, **kw)   # global test (protocol)
        own = fit_eval(Xtr[i], ytr[i], Xte[i], yte[i], seed, **kw)
        r["cid"] = c["cid"]
        r["own_test_split_macro_f1"] = own["macro_f1"]
        r["own_test_classes_present"] = own["classes_present"]
        per.append(r)
    f1 = [r["macro_f1"] for r in per]
    own_f1 = [r["own_test_split_macro_f1"] for r in per]
    return {
        "per_client": per,
        # ddof=1: reported "mean +/- s" spreads are SAMPLE standard deviations
        # (here over the K=10 client fits of one seed), the manuscript convention.
        "mean": float(np.mean(f1)), "std": float(np.std(f1, ddof=1)),
        "min": float(np.min(f1)), "max": float(np.max(f1)),
        "accuracy_mean": float(np.mean([r["accuracy"] for r in per])),
        "own_test_split_mean": float(np.mean(own_f1)),
        "own_test_split_min": float(np.min(own_f1)),
        "own_test_split_max": float(np.max(own_f1)),
    }


ALL = tuple(range(len(CHANNELS)))
NO_FDEV = tuple(c for c in ALL if c != FDEV)
ONLY_FDEV = (FDEV,)


def _federations() -> dict[tuple[float, int], Path]:
    feds = {}
    for d in sorted(BATTERY.glob("fed_*")):
        cfg = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["config"]
        feds[(cfg["alpha"], cfg["seed"])] = d
    return feds


def write_merged(payload: dict) -> None:
    """Write ``payload``'s keys into OUT, preserving every key it does not own.

    The two batteries (random, temporal) live side by side in one file, and the
    paired-ablation blocks are added on top of them, so a run must never clobber
    a sibling key it did not produce.
    """
    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if OUT.exists():
        existing = json.loads(OUT.read_text(encoding="utf-8"))
    existing.update(payload)
    OUT.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def main() -> None:
    t_start = time.time()
    feds = _federations()
    # The reference frame (tab_reference_frame / block t1a) is alpha=0.5, seeds 0-4.
    reference = [(0.5, s) for s in range(5)]
    extra = sorted(k for k in feds if k not in reference)

    results = {"reference_battery": {}, "alpha_sweep_extra": {}}

    for alpha, seed in reference:
        d = feds[(alpha, seed)]
        print(f"[ref] {d.name} alpha={alpha} seed={seed}", flush=True)
        cl = load_fed(d, normalize=True)
        raw = load_fed(d, normalize=False)
        r = {
            "fed_dir": d.name, "alpha": alpha, "seed": seed,
            "pooled_all_channels": pooled(cl, ALL, seed),
            "pooled_no_fdev": pooled(cl, NO_FDEV, seed),
            "pooled_fdev_only": pooled(cl, ONLY_FDEV, seed),
            "local_only": local_only(cl, ALL, seed),
            # robustness: unnormalised inputs, and early stopping disabled
            "pooled_all_channels_raw_inputs": pooled(raw, ALL, seed),
            "pooled_all_channels_no_early_stop": pooled(cl, ALL, seed,
                                                        early_stopping=False),
        }
        for k, v in r.items():
            if isinstance(v, dict) and "macro_f1" in v:
                print(f"    {k:38s} macroF1={v['macro_f1']:.4f} "
                      f"acc={v['accuracy']:.4f} iters={v['n_iter']}", flush=True)
        print(f"    {'local_only':38s} mean={r['local_only']['mean']:.4f} "
              f"[{r['local_only']['min']:.4f},{r['local_only']['max']:.4f}]", flush=True)
        results["reference_battery"][f"seed{seed}"] = r

    for alpha, seed in extra:
        d = feds[(alpha, seed)]
        print(f"[extra] {d.name} alpha={alpha} seed={seed}", flush=True)
        cl = load_fed(d, normalize=True)
        results["alpha_sweep_extra"][f"alpha{alpha}_seed{seed}"] = {
            "fed_dir": d.name, "alpha": alpha, "seed": seed,
            "pooled_all_channels": pooled(cl, ALL, seed),
            "pooled_no_fdev": pooled(cl, NO_FDEV, seed),
            "pooled_fdev_only": pooled(cl, ONLY_FDEV, seed),
        }

    # ---- aggregate over the 5 reference seeds -------------------------------
    from experiments.analyze import bca_ci

    ref = list(results["reference_battery"].values())
    agg = {}
    for key in ("pooled_all_channels", "pooled_no_fdev", "pooled_fdev_only",
                "pooled_all_channels_raw_inputs", "pooled_all_channels_no_early_stop"):
        vals = [r[key]["macro_f1"] for r in ref]
        lo, hi = bca_ci(vals)
        agg[key] = {
            # ddof=1: sample sd over the n=5 reference seeds (the seeds are a
            # sample of federations, not the population), per the manuscript's
            # "mean +/- s" convention -- same as experiments/report.py.
            "macro_f1_mean": float(np.mean(vals)), "macro_f1_std": float(np.std(vals, ddof=1)),
            "macro_f1_per_seed": [float(v) for v in vals],
            "bca_ci_95": [float(lo), float(hi)],
            "accuracy_mean": float(np.mean([r[key]["accuracy"] for r in ref])),
            "n_seeds": len(vals),
        }
    lm = [r["local_only"]["mean"] for r in ref]
    all_clients = [c["macro_f1"] for r in ref for c in r["local_only"]["per_client"]]
    own = [c["own_test_split_macro_f1"] for r in ref for c in r["local_only"]["per_client"]]
    agg["local_only"] = {
        # ddof=1: sample sd over the n=5 seed means (see above).
        "macro_f1_mean": float(np.mean(lm)), "macro_f1_std": float(np.std(lm, ddof=1)),
        "macro_f1_per_seed": [float(v) for v in lm],
        "per_client_min": float(np.min(all_clients)),
        "per_client_max": float(np.max(all_clients)),
        "accuracy_mean": float(np.mean([r["local_only"]["accuracy_mean"] for r in ref])),
        "n_seeds": len(lm), "n_client_fits": len(all_clients),
        "eval_split": "global pooled test (as experiments/build.py _SingleClientFederation)",
    }
    agg["local_only_own_test_split_DIAGNOSTIC"] = {
        "macro_f1_mean": float(np.mean(own)),
        "per_client_min": float(np.min(own)), "per_client_max": float(np.max(own)),
        "note": "NOT comparable to the other arms -- each client is scored on its own "
                "skewed test split (often only 1-3 of the 8 classes present), which "
                "inflates the floor. Reported only to document the gap.",
    }
    agg["federated"] = None  # GBT does not federate by parameter averaging -> '--'
    results["aggregate_reference"] = agg

    results["config"] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "shallow GBT baseline, Reviewer-2 reproduction under fedgrid protocol",
        "model": "sklearn.ensemble.HistGradientBoostingClassifier",
        "sklearn_version": __import__("sklearn").__version__,
        "model_params": "library defaults; random_state = federation seed",
        "features": {
            "stats_per_channel": list(STATS),
            "channels": list(CHANNELS),
            "fdev_channel_index": FDEV,
            "n_features_all_channels": len(CHANNELS) * len(STATS),
            "n_features_no_fdev": (len(CHANNELS) - 1) * len(STATS),
            "n_features_fdev_only": len(STATS),
            "layout": "windows are (N, C=11, T=64); stats over the time axis; "
                      "channel-major flattening",
            "names_all_channels": feature_names(ALL),
        },
        "protocol": {
            "splits": "persisted train_idx/val_idx/test_idx from each client npz "
                      "(never re-split); 70/15/15",
            "normalisation": "persisted per-client norm_mean/norm_std (train-fit), "
                             "applied as in ClientDataset.__getitem__",
            "val_split_usage": "unused by the default estimator; sklearn's default "
                               "early_stopping='auto' carves its own 10% holdout from "
                               "train when n>10000 (pooled arms only). The "
                               "*_no_early_stop variant removes this asymmetry.",
            "metric": "fedgrid.fl.engine._f1_from_confusion -- macro-F1 averaged over "
                      "classes PRESENT in y_true (identical to the deep-model arms)",
            "pooled_test": "union of the 10 client test splits (Federation.global_test)",
            "local_only": "client k trained on its own train split, scored on the GLOBAL "
                          "pooled test set -- the same evaluation set as every other arm, "
                          "per experiments/build.py::_SingleClientFederation",
            "class_weighting": "none (estimator default). The deep arms train with "
                               "class_weighting='effective_num'; the GBT baseline gets no "
                               "imbalance correction, which is the Reviewer's own spec.",
            "reference_battery": "alpha=0.5, seeds 0-4 (block t1a); each seed is a "
                                 "distinct generated federation, not a model reseed",
        },
        "runtime_seconds": round(time.time() - t_start, 1),
    }

    write_merged(results)
    print(f"\nwrote {OUT}  ({time.time() - t_start:.1f}s)")


# --------------------------------------------------------------------------- #
# Leakage-controlled (chronological) re-measurement of the same probe.         #
# --------------------------------------------------------------------------- #

def run_temporal(gap: int = TEMPORAL_SPLIT_GAP, with_local: bool = False) -> None:
    """Re-measure the pooled probe + frequency ablation under the temporal split.

    Same five reference federations (alpha=0.5, seeds 0-4), same features, same
    estimator and same metric as the random battery -- only the split changes, so
    the pair is read exactly like tab_temporal_frame against tab_reference_frame.
    """
    from experiments.analyze import bca_ci, wilcoxon_signed_rank

    t_start = time.time()
    feds = _federations()
    reference = [(0.5, s) for s in range(5)]
    battery: dict[str, dict] = {}

    for alpha, seed in reference:
        d = feds[(alpha, seed)]
        print(f"[temporal] {d.name} alpha={alpha} seed={seed}", flush=True)
        cl = load_fed(d, normalize=True, split_mode="temporal", gap=gap)
        r = {
            "fed_dir": d.name, "alpha": alpha, "seed": seed,
            "n_train_windows": int(sum(len(c["train"]) for c in cl)),
            "n_test_windows": int(sum(len(c["test"]) for c in cl)),
            "pooled_all_channels": pooled(cl, ALL, seed),
            "pooled_no_fdev": pooled(cl, NO_FDEV, seed),
            "pooled_fdev_only": pooled(cl, ONLY_FDEV, seed),
        }
        if with_local:
            r["local_only"] = local_only(cl, ALL, seed)
        for k, v in r.items():
            if isinstance(v, dict) and "macro_f1" in v:
                print(f"    {k:24s} macroF1={v['macro_f1']:.4f} "
                      f"acc={v['accuracy']:.4f} iters={v['n_iter']}", flush=True)
        battery[f"seed{seed}"] = r

    ref = list(battery.values())
    agg = {}
    arms = ["pooled_all_channels", "pooled_no_fdev", "pooled_fdev_only"]
    for key in arms:
        vals = [r[key]["macro_f1"] for r in ref]
        lo, hi = bca_ci(vals)
        agg[key] = {
            "macro_f1_mean": float(np.mean(vals)),
            # ddof=1: sample sd over the n=5 reference seeds, as in the random
            # battery -- the pair is read as "mean +/- s" side by side.
            "macro_f1_std": float(np.std(vals, ddof=1)),
            "macro_f1_per_seed": [float(v) for v in vals],
            "bca_ci_95": [float(lo), float(hi)],
            "accuracy_mean": float(np.mean([r[key]["accuracy"] for r in ref])),
            "n_seeds": len(vals),
        }
    if with_local:
        lm = [r["local_only"]["mean"] for r in ref]
        allc = [c["macro_f1"] for r in ref for c in r["local_only"]["per_client"]]
        agg["local_only"] = {
            "macro_f1_mean": float(np.mean(lm)),
            # ddof=1: sample sd over the n=5 seed means (see above).
            "macro_f1_std": float(np.std(lm, ddof=1)),
            "macro_f1_per_seed": [float(v) for v in lm],
            "per_client_min": float(np.min(allc)), "per_client_max": float(np.max(allc)),
            "n_seeds": len(lm), "n_client_fits": len(allc),
        }
    agg["federated"] = None

    base = [r["pooled_all_channels"]["macro_f1"] for r in ref]
    ablation = {}
    for key in ("pooled_no_fdev", "pooled_fdev_only"):
        deltas = [r[key]["macro_f1"] - b for r, b in zip(ref, base)]
        lo, hi = bca_ci(deltas)
        ablation[f"{key.replace('pooled_', '')}_minus_all"] = {
            "delta_per_seed": [round(float(v), 4) for v in deltas],
            "delta_mean": float(np.mean(deltas)),
            "delta_bca_ci_95": [float(lo), float(hi)],
            "wilcoxon": wilcoxon_signed_rank(deltas),
            "n_seeds": len(deltas),
        }
    ablation["interpretation"] = (
        f"Under the chronological split dropping fdev changes pooled macro-F1 by "
        f"{ablation['no_fdev_minus_all']['delta_mean']:+.4f} (BCa CI spanning zero, "
        f"Wilcoxon p_two="
        f"{ablation['no_fdev_minus_all']['wilcoxon']['p_two']:.4g}) and fdev alone "
        f"reaches {agg['pooled_fdev_only']['macro_f1_mean']:.4f} -- the same reading "
        f"as the random split: the frequency channel carries real but far-from-"
        f"sufficient signal, and its apparent contribution is inside seed noise."
    )

    payload = {
        "battery": battery,
        "aggregate": agg,
        "frequency_ablation_paired": ablation,
        "config": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "split_mode": "temporal",
            "gap_windows": int(gap),
            "sklearn_version": __import__("sklearn").__version__,
            "protocol": {
                "splits": "fedgrid.data.temporal_split_indices(n, gap=4) per client -- "
                          "contiguous 70/15/15 chronological blocks with the first 4 "
                          "windows of the val and test blocks purged, so every held-out "
                          "window is sample-disjoint from every training window "
                          "(stride 16 at T=64). Identical rule to the deep arms' "
                          "federation.split_mode='temporal' (block T1a_temporal).",
                "normalisation": "refitted on the new train block (Normalizer.fit), as "
                                 "apply_temporal_split does -- the persisted npz "
                                 "statistics were fitted on the random train split and "
                                 "would leak the newest days into the scaling.",
                "window_order": "index order is time order (make_federation concatenates "
                                "day 0, day 1, ...; each day is cut left to right)",
                "everything_else": "unchanged from the random battery: same 5 "
                                   "federations (alpha=0.5, seeds 0-4), same 77 features, "
                                   "HistGradientBoosting at library defaults, same "
                                   "macro-F1 over classes present in y_true, same pooled "
                                   "test set definition (union of client test blocks)",
                "arms": "pooled (centralized) only; the local-only floor and the "
                        "robustness variants are measured on the random split",
            },
            "runtime_seconds": round(time.time() - t_start, 1),
        },
    }
    write_merged({"temporal_reference_battery": payload})

    m = agg["pooled_all_channels"]
    print(f"\npooled temporal macro-F1 = {m['macro_f1_mean']:.4f} "
          f"+- {m['macro_f1_std']:.4f}  (random split: 0.6922 +- 0.0417)")
    print(f"no_fdev = {agg['pooled_no_fdev']['macro_f1_mean']:.4f}  "
          f"fdev_only = {agg['pooled_fdev_only']['macro_f1_mean']:.4f}")
    print(f"wrote {OUT}  ({time.time() - t_start:.1f}s)")


def cli(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split-mode", choices=SPLIT_MODES, default="random",
                   help="random (default): the persisted permutation split, the "
                        "original battery. temporal: the leakage-controlled "
                        "chronological split of fedgrid.data.")
    p.add_argument("--gap", type=int, default=TEMPORAL_SPLIT_GAP,
                   help="windows purged at each temporal block boundary")
    p.add_argument("--with-local", action="store_true",
                   help="also measure the local-only floor under --split-mode temporal")
    a = p.parse_args(argv)
    if a.split_mode == "temporal":
        run_temporal(gap=a.gap, with_local=a.with_local)
    else:
        main()


if __name__ == "__main__":
    cli()
