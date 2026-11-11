"""Statistical battery over a ``runs/`` directory of RunResults.

Implements the non-parametric battery of ``docs/VALIDATION_STANDARD.md`` §3.2 and
the hypotheses of ``experiments/HYPOTHESES.md``: per-arm mean +/- std with **BCa
95% bootstrap CIs** (10k resamples), **Wilcoxon signed-rank + Holm** for the
pairwise claims, **Friedman + Nemenyi/Holm** across the defense arms, **Cliff's
delta** effect sizes, **McNemar** for the SNN-vs-CNN-LSTM matched-accuracy claim,
and ordered-trend tests (**Jonckheere-Terpstra**, Spearman) for the LDP-sigma and
non-IID-alpha curves.

**Stats path: numpy + stdlib only.** ``scipy`` is not a project dependency (checked
at import and reported in the JSON summary's ``meta``); every test here is a
self-contained implementation with its formula cited, so the battery is
reproducible with or without scipy present. The normal CDF/quantile come from
``statistics.NormalDist``; the chi-square survival function uses the regularised
incomplete gamma (Press et al., *Numerical Recipes* 3rd ed., §6.2).

Wave-level experiment analysis module. Exercised on synthetic RunResults in the tests; real
RunResults arrive in phase 4.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

__all__ = [
    "SCIPY_AVAILABLE",
    "analyze_dir",
    "bca_ci",
    "cliffs_delta",
    "friedman",
    "holm",
    "jonckheere",
    "mcnemar_from_correct",
    "mcnemar_test",
    "nemenyi_cd",
    "wilcoxon_signed_rank",
]

SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None
_NORMAL = NormalDist()
_ALPHA = 0.05

# Studentised-range / sqrt(2) critical values at alpha=0.05 for the Nemenyi test
# (Demsar, JMLR 7:1-30, 2006, Table 5(a)), indexed by the number of arms k.
_NEMENYI_Q05: dict[int, float] = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
    7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}


# --------------------------------------------------------------------------- #
# Special functions (no scipy): normal via NormalDist, chi-square via incomplete
# gamma (Numerical Recipes gser/gcf).
# --------------------------------------------------------------------------- #


def _gser(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x) by series (NR §6.2, x < a+1)."""
    if x <= 0.0:
        return 0.0
    gln = math.lgamma(a)
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(1000):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - gln)


def _gcf(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) by continued fraction (NR, x >= a+1)."""
    gln = math.lgamma(a)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def _chi2_sf(x: float, df: int) -> float:
    """Chi-square survival function = Q(df/2, x/2)."""
    if x <= 0.0:
        return 1.0
    a = df / 2.0
    xx = x / 2.0
    return 1.0 - _gser(a, xx) if xx < a + 1.0 else _gcf(a, xx)


def _rankdata(a: Sequence[float] | np.ndarray) -> np.ndarray:
    """Ranks with ties averaged (SciPy 'average' semantics), 1-based."""
    arr = np.asarray(a, dtype=float)
    n = arr.size
    order = arr.argsort(kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    sorted_a = arr[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


# --------------------------------------------------------------------------- #
# Core tests.
# --------------------------------------------------------------------------- #


def bca_ci(
    data: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10000,
    alpha: float = _ALPHA,
    seed: int = 0,
) -> tuple[float, float]:
    """BCa bootstrap CI (Efron, JASA 1987): bias- and acceleration-corrected.

    numpy implementation of the accelerated bias-corrected percentile interval
    (10k resamples per §3.2). Degenerate inputs (n<2 or a constant statistic)
    return the point estimate for both bounds rather than raising.
    """
    d = np.asarray(data, dtype=float)
    n = d.size
    theta = float(statistic(d))
    if n < 2:
        return theta, theta
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boots = np.array([float(statistic(d[row])) for row in idx])
    if np.allclose(boots, boots[0]):
        return theta, theta
    prop = float(np.mean(boots < theta))
    prop = min(max(prop, 1e-9), 1.0 - 1e-9)
    z0 = _NORMAL.inv_cdf(prop)
    jack = np.array([float(statistic(np.delete(d, i))) for i in range(n)])
    jbar = jack.mean()
    diff = jbar - jack
    denom = 6.0 * float(np.sum(diff ** 2)) ** 1.5
    accel = float(np.sum(diff ** 3)) / denom if denom != 0.0 else 0.0

    def _endpoint(z: float) -> float:
        adj = z0 + (z0 + z) / (1.0 - accel * (z0 + z))
        return min(max(_NORMAL.cdf(adj), 0.0), 1.0)

    lo = float(np.percentile(boots, 100.0 * _endpoint(_NORMAL.inv_cdf(alpha / 2.0))))
    hi = float(np.percentile(boots, 100.0 * _endpoint(_NORMAL.inv_cdf(1.0 - alpha / 2.0))))
    return lo, hi


def wilcoxon_signed_rank(x: Sequence[float], y: Sequence[float] | None = None) -> dict[str, float]:
    """Wilcoxon signed-rank test; exact for n<=18, normal approximation above.

    Returns ``T_plus`` (sum of positive-difference ranks), ``n`` (non-zero pairs)
    and two- and one-sided p-values. Exact enumeration is the null distribution of
    T_plus over all 2^n sign assignments of the ranks (valid with averaged tie
    ranks). ``p_greater`` = P(T_plus >= observed) tests x > y; ``p_less`` the
    reverse. Zero differences are dropped (Wilcoxon's convention).
    """
    xa = np.asarray(x, dtype=float)
    d = xa if y is None else xa - np.asarray(y, dtype=float)
    d = d[d != 0.0]
    n = int(d.size)
    if n == 0:
        return {"T_plus": 0.0, "n": 0.0, "p_two": 1.0, "p_greater": 1.0, "p_less": 1.0}
    r = _rankdata(np.abs(d))
    t_plus = float(r[d > 0].sum())
    if n <= 18:
        bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
        sums = bits @ r
        mean = float(sums.mean())
        obs_dev = abs(t_plus - mean)
        p_two = float(np.mean(np.abs(sums - mean) >= obs_dev - 1e-9))
        p_greater = float(np.mean(sums >= t_plus - 1e-9))
        p_less = float(np.mean(sums <= t_plus + 1e-9))
    else:
        mean = n * (n + 1) / 4.0
        _, counts = np.unique(np.abs(d), return_counts=True)
        tie = float(np.sum(counts ** 3 - counts))
        var = n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0
        if var <= 0:
            return {"T_plus": t_plus, "n": float(n), "p_two": 1.0,
                    "p_greater": 1.0, "p_less": 1.0}
        sd = math.sqrt(var)
        z = (t_plus - mean) / sd
        cc = 0.5 / sd  # continuity correction
        p_two = 2.0 * _NORMAL.cdf(-(abs(z) - cc)) if abs(z) > cc else 1.0
        p_greater = _NORMAL.cdf(-(z - cc))  # P(T+ >= obs)
        p_less = _NORMAL.cdf(z + cc)  # P(T+ <= obs)
    return {"T_plus": t_plus, "n": float(n),
            "p_two": min(1.0, p_two), "p_greater": min(1.0, p_greater),
            "p_less": min(1.0, p_less)}


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (Holm, Scand. J. Statist. 1979)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)  # enforce monotonicity down the sorted list
        adj[i] = min(1.0, running)
    return [float(v) for v in adj]


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Cliff's delta with magnitude band (Cliff 1993; bands: Romano et al. 2006)."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size == 0 or ya.size == 0:
        return {"delta": 0.0, "magnitude": "undefined"}
    gt = int(np.sum(xa[:, None] > ya[None, :]))
    lt = int(np.sum(xa[:, None] < ya[None, :]))
    delta = (gt - lt) / (xa.size * ya.size)
    a = abs(delta)
    mag = ("negligible" if a < 0.147 else "small" if a < 0.33
           else "medium" if a < 0.474 else "large")
    return {"delta": float(delta), "magnitude": mag}


def friedman(blocks: np.ndarray) -> dict[str, Any]:
    """Friedman test across k arms over n blocks (Demsar 2006); chi-square p, df=k-1."""
    m = np.asarray(blocks, dtype=float)
    n, k = m.shape
    ranks = np.vstack([_rankdata(row) for row in m])
    rj = ranks.sum(axis=0)
    q = 12.0 / (n * k * (k + 1)) * float(np.sum(rj ** 2)) - 3.0 * n * (k + 1)
    return {"Q": float(q), "p": float(_chi2_sf(q, k - 1)),
            "avg_ranks": [float(v) for v in rj / n], "n": int(n), "k": int(k)}


def nemenyi_cd(k: int, n: int, alpha: float = _ALPHA) -> float:
    """Nemenyi critical difference for k arms over n blocks (Demsar 2006, alpha=0.05)."""
    if alpha != _ALPHA or k not in _NEMENYI_Q05:
        raise ValueError(f"Nemenyi table has alpha=0.05, k in 2..10; got alpha={alpha}, k={k}")
    return _NEMENYI_Q05[k] * math.sqrt(k * (k + 1) / (6.0 * n))


def nemenyi_posthoc(avg_ranks: Sequence[float], n: int) -> dict[str, dict[str, float]]:
    """Pairwise rank-difference z-tests (Holm-adjusted by the caller)."""
    ar = list(avg_ranks)
    k = len(ar)
    se = math.sqrt(k * (k + 1) / (6.0 * n))
    out: dict[str, dict[str, float]] = {}
    for i in range(k):
        for j in range(i + 1, k):
            z = abs(ar[i] - ar[j]) / se if se > 0 else 0.0
            out[f"{i}-{j}"] = {"z": float(z), "p": float(2.0 * _NORMAL.cdf(-z))}
    return out


def mcnemar_test(b: int, c: int) -> dict[str, float]:
    """Exact two-sided McNemar on discordant counts (b, c) (binomial, p=0.5).

    The tail is summed in **log space** above ``n = 1000``: the pooled test over
    the whole global test set puts thousands of discordant pairs in play, and the
    literal ``comb(n, i) * 0.5**n`` product overflows there (the binomial
    coefficient exceeds the float range while ``0.5**n`` underflows to zero).
    Below that threshold the integer path is kept so small-n results stay
    bit-identical to the textbook computation.
    """
    b, c = int(b), int(c)
    n = b + c
    if n == 0:
        return {"b": 0.0, "c": 0.0, "stat": 0.0, "p": 1.0}
    lo = min(b, c)
    if n <= 1000:
        tail = sum(math.comb(n, i) for i in range(lo + 1)) * (0.5 ** n)
    else:
        log_half = n * math.log(0.5)
        log_fact_n = math.lgamma(n + 1)
        tail = math.fsum(
            math.exp(log_fact_n - math.lgamma(i + 1) - math.lgamma(n - i + 1) + log_half)
            for i in range(lo + 1)
        )
    p = min(1.0, 2.0 * tail)
    stat = (abs(b - c) - 1.0) ** 2 / n  # continuity-corrected chi-square (reported)
    return {"b": float(b), "c": float(c), "stat": float(stat), "p": float(p)}


def mcnemar_from_correct(correct_a: Sequence[int], correct_b: Sequence[int]) -> dict[str, float]:
    """McNemar from two aligned per-sample correctness vectors (True/1 = correct)."""
    ca = np.asarray(correct_a).astype(bool)
    cb = np.asarray(correct_b).astype(bool)
    if ca.shape != cb.shape:
        raise ValueError(f"correctness vectors misaligned: {ca.shape} vs {cb.shape}")
    b = int(np.sum(ca & ~cb))  # a right, b wrong
    c = int(np.sum(~ca & cb))  # a wrong, b right
    return mcnemar_test(b, c)


def jonckheere(groups: Sequence[Sequence[float]]) -> dict[str, float]:
    """Jonckheere-Terpstra ordered-trend test; normal approximation with tie handling.

    ``groups`` are given in the hypothesised increasing order. Large J is evidence
    the later groups are stochastically larger; ``p_one_increasing`` is the
    one-sided p for that ordered alternative.
    """
    gs = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    if len(gs) < 2:
        return {"J": 0.0, "z": 0.0, "p_two": 1.0, "p_one_increasing": 1.0}
    j_stat = 0.0
    for i in range(len(gs)):
        for jj in range(i + 1, len(gs)):
            gi, gj = gs[i], gs[jj]
            j_stat += float(np.sum(gi[:, None] < gj[None, :])) \
                + 0.5 * float(np.sum(gi[:, None] == gj[None, :]))
    n_tot = sum(len(g) for g in gs)
    mu = (n_tot ** 2 - sum(len(g) ** 2 for g in gs)) / 4.0
    var = (n_tot ** 2 * (2 * n_tot + 3)
           - sum(len(g) ** 2 * (2 * len(g) + 3) for g in gs)) / 72.0
    if var <= 0:
        return {"J": float(j_stat), "z": 0.0, "p_two": 1.0, "p_one_increasing": 1.0}
    z = (j_stat - mu) / math.sqrt(var)
    return {"J": float(j_stat), "z": float(z),
            "p_two": float(2.0 * _NORMAL.cdf(-abs(z))),
            "p_one_increasing": float(_NORMAL.cdf(-z))}


def spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    """Spearman rho with large-sample normal p (rho*sqrt(n-1) ~ N(0,1))."""
    rx = _rankdata(x)
    ry = _rankdata(y)
    n = rx.size
    if n < 3 or np.std(rx) == 0 or np.std(ry) == 0:
        return {"rho": 0.0, "p_two": 1.0}
    rho = float(np.corrcoef(rx, ry)[0, 1])
    z = rho * math.sqrt(n - 1)
    return {"rho": rho, "p_two": float(2.0 * _NORMAL.cdf(-abs(z)))}


# --------------------------------------------------------------------------- #
# RunResult loading + record extraction.
# --------------------------------------------------------------------------- #


@dataclass
class Record:
    run_id: str
    arm: str
    model: str
    seed: int
    block: str
    attack: str | None
    alpha: float | None
    sigma: float | None
    macro_f1: float | None
    acc: float | None
    energy_ratio: float | None
    correct: list[int] | None


def _record(run_id_fallback: str, config: dict[str, Any], final: dict[str, Any],
            energy: dict[str, Any] | None, seed: int, method: str) -> Record:
    manifest = config.get("manifest") if isinstance(config, dict) else None
    if isinstance(manifest, dict):
        arm = str(manifest.get("arm", method))
        model = str(manifest.get("model", "?"))
        seed = int(manifest.get("seed", seed))
        block = str(manifest.get("block", "?"))
        inner = manifest.get("config", {}) or {}
        run_id = str(manifest.get("run_id", run_id_fallback))
    else:
        arm = str(config.get("arm", method))
        model = str(config.get("model", "?"))
        block = str(config.get("block", "?"))
        inner = config
        run_id = run_id_fallback
    attack = None
    alpha = None
    sigma = None
    if isinstance(inner, dict):
        atk = inner.get("attack")
        if isinstance(atk, dict):
            attack = atk.get("type")
        fed = inner.get("federation", {})
        if isinstance(fed, dict):
            alpha = fed.get("alpha")
        ldp = inner.get("ldp")
        if isinstance(ldp, dict):
            sigma = ldp.get("sigma")
    return Record(
        run_id=run_id, arm=arm, model=model, seed=int(seed), block=block,
        attack=attack, alpha=alpha, sigma=sigma,
        macro_f1=final.get("macro_f1"), acc=final.get("acc"),
        energy_ratio=(energy or {}).get("ratio") if energy else None,
        correct=final.get("correct"),
    )


def load_runs(runs_dir: Path | str) -> list[Record]:
    """Load every ``*.json`` RunResult under ``runs_dir`` into records."""
    from fedgrid.metrics.runresult import load_run

    records: list[Record] = []
    for path in sorted(Path(runs_dir).glob("*.json")):
        try:
            rr = load_run(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError):
            continue
        records.append(_record(path.stem, rr.config, rr.final, rr.energy, rr.seed, rr.method))
    return records


# --------------------------------------------------------------------------- #
# Battery.
# --------------------------------------------------------------------------- #


def _cell_key(rec: Record) -> tuple[str, str | None]:
    return rec.model, rec.attack


def _by_seed(recs: list[Record]) -> dict[int, float]:
    return {r.seed: float(r.macro_f1) for r in recs if r.macro_f1 is not None}


def _summary(records: list[Record], n_resamples: int) -> dict[str, Any]:
    groups: dict[tuple[str, str, str | None], list[float]] = defaultdict(list)
    for r in records:
        if r.macro_f1 is not None:
            groups[(r.arm, r.model, r.attack)].append(float(r.macro_f1))
    out: dict[str, Any] = {}
    for (arm, model, attack), vals in sorted(groups.items(), key=lambda kv: str(kv[0])):
        lo, hi = bca_ci(vals, np.mean, n_resamples=n_resamples)
        key = f"{arm}|{model}|{attack or 'clean'}"
        out[key] = {
            "arm": arm, "model": model, "attack": attack, "n": len(vals),
            "mean_macro_f1": float(np.mean(vals)), "std_macro_f1": float(np.std(vals, ddof=1))
            if len(vals) > 1 else 0.0,
            "bca_ci_low": lo, "bca_ci_high": hi,
        }
    return out


def _pairwise(records: list[Record]) -> list[dict[str, Any]]:
    """Wilcoxon + Holm of each defense arm vs the no-defense baseline, per (model, attack)."""
    cells: dict[tuple[str, str | None], dict[str, list[Record]]] = defaultdict(
        lambda: defaultdict(list))
    for r in records:
        cells[_cell_key(r)][r.arm].append(r)

    results: list[dict[str, Any]] = []
    for (model, attack), arms in sorted(cells.items(), key=lambda kv: str(kv[0])):
        baseline = ("no_defense" if "no_defense" in arms
                    else "fedavg" if "fedavg" in arms else None)
        if baseline is None:
            continue
        base = _by_seed(arms[baseline])
        family: list[dict[str, Any]] = []
        for arm, recs in sorted(arms.items()):
            if arm == baseline:
                continue
            other = _by_seed(recs)
            seeds = sorted(set(base) & set(other))
            if len(seeds) < 2:
                continue
            a = np.array([other[s] for s in seeds])  # candidate arm
            b = np.array([base[s] for s in seeds])   # baseline
            w = wilcoxon_signed_rank(a, b)
            eff = cliffs_delta(a, b)
            diff = a - b
            dlo, dhi = bca_ci(diff, np.mean)
            family.append({
                "model": model, "attack": attack, "arm": arm, "baseline": baseline,
                "n_pairs": len(seeds), "T_plus": w["T_plus"],
                "p_two": w["p_two"], "p_one_greater": w["p_greater"],
                "cliffs_delta": eff["delta"], "magnitude": eff["magnitude"],
                "mean_diff": float(np.mean(diff)), "diff_ci_low": dlo, "diff_ci_high": dhi,
            })
        for entry, p_adj in zip(family, holm([f["p_two"] for f in family]), strict=True):
            entry["p_holm"] = p_adj
        results.extend(family)
    return results


def _friedman_across_arms(records: list[Record]) -> list[dict[str, Any]]:
    """Friedman + Nemenyi across defense arms per (model, attack) cell with >=3 arms."""
    cells: dict[tuple[str, str | None], dict[str, dict[int, float]]] = defaultdict(dict)
    for r in records:
        if r.macro_f1 is None:
            continue
        cells[_cell_key(r)].setdefault(r.arm, {})[r.seed] = float(r.macro_f1)

    out: list[dict[str, Any]] = []
    for (model, attack), arms in sorted(cells.items(), key=lambda kv: str(kv[0])):
        arm_names = sorted(arms)
        if len(arm_names) < 3:
            continue
        common = sorted(set.intersection(*(set(arms[a]) for a in arm_names)))
        if len(common) < 2:
            continue
        block = np.array([[arms[a][s] for a in arm_names] for s in common])
        fr = friedman(block)
        posthoc = nemenyi_posthoc(fr["avg_ranks"], fr["n"])
        # Holm over the pairwise post-hoc p-values.
        keys = list(posthoc)
        for k_, p_adj in zip(keys, holm([posthoc[k]["p"] for k in keys]), strict=True):
            posthoc[k_]["p_holm"] = p_adj
        out.append({
            "model": model, "attack": attack, "arms": arm_names,
            "Q": fr["Q"], "p": fr["p"], "avg_ranks": fr["avg_ranks"],
            "nemenyi_cd": nemenyi_cd(len(arm_names), fr["n"]),
            "posthoc": {f"{arm_names[int(k.split('-')[0])]}_vs_{arm_names[int(k.split('-')[1])]}":
                        v for k, v in posthoc.items()},
        })
    return out


def _mcnemar_parity(records: list[Record]) -> dict[str, Any]:
    """SNN vs CNN-LSTM matched-accuracy McNemar on the clean reference arm.

    Pairs the two models at the same (arm, seed) via their per-sample correctness
    vectors; reports per-seed McNemar and the pooled discordance.
    """
    parity_arms = ("fedavg", "centralized")
    by_key: dict[tuple[str, int, str], Record] = {}
    for r in records:
        if r.attack is None and r.correct is not None and r.arm in parity_arms:
            by_key[(r.arm, r.seed, r.model)] = r

    per_seed: list[dict[str, Any]] = []
    tot_b = tot_c = 0
    for (arm, seed, model), rec in by_key.items():
        if model != "cnn_lstm":
            continue
        snn = by_key.get((arm, seed, "snn"))
        if snn is None or rec.correct is None or snn.correct is None:
            continue
        if len(rec.correct) != len(snn.correct):
            continue
        res = mcnemar_from_correct(snn.correct, rec.correct)  # snn vs cnn_lstm
        res.update({"arm": arm, "seed": seed})
        per_seed.append(res)
        tot_b += int(res["b"])
        tot_c += int(res["c"])
    if not per_seed:
        return {"available": False,
                "reason": "no matched clean cnn_lstm/snn runs with per-sample correctness",
                "per_seed": [], "pooled": None}
    return {"available": True, "per_seed": per_seed, "pooled": mcnemar_test(tot_b, tot_c)}


def _trends(records: list[Record]) -> list[dict[str, Any]]:
    """Ordered-trend tests: non-IID alpha (H5a) and LDP sigma (H4) where present."""
    out: list[dict[str, Any]] = []

    # H5a: macro-F1 increasing with alpha, per (arm, model).
    alpha_cells: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for r in records:
        if r.attack is None and r.alpha is not None and r.macro_f1 is not None:
            alpha_cells[(r.arm, r.model)][float(r.alpha)].append(float(r.macro_f1))
    for (arm, model), by_alpha in sorted(alpha_cells.items()):
        alphas = sorted(by_alpha)
        if len(alphas) < 3:
            continue
        groups = [by_alpha[a] for a in alphas]
        jt = jonckheere(groups)
        means = [float(np.mean(by_alpha[a])) for a in alphas]
        sp = spearman([a for a in alphas for _ in by_alpha[a]],
                      [v for a in alphas for v in by_alpha[a]])
        out.append({"family": "H5a_alpha", "arm": arm, "model": model,
                    "order": alphas, "group_means": means,
                    "jonckheere": jt, "spearman": sp})

    # H4: honest false-reject proxy would need per-round scores; here we trend
    # macro-F1 vs sigma at fixed defense (a coarse LDP-degradation signal).
    sigma_cells: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.sigma is not None and r.macro_f1 is not None:
            sigma_cells[r.model][float(r.sigma)].append(float(r.macro_f1))
    for model, by_sigma in sorted(sigma_cells.items()):
        sigmas = sorted(by_sigma)
        if len(sigmas) < 3:
            continue
        # decreasing macro-F1 with sigma == increasing FRR; test the reversed order
        groups = [by_sigma[s] for s in reversed(sigmas)]
        jt = jonckheere(groups)
        out.append({"family": "H4_sigma", "model": model, "order_desc": list(reversed(sigmas)),
                    "group_means": [float(np.mean(by_sigma[s])) for s in sigmas],
                    "jonckheere": jt})
    return out


def analyze_dir(runs_dir: Path | str, n_resamples: int = 10000) -> dict[str, Any]:
    """Load ``runs_dir`` and compute the whole statistical battery into one dict."""
    records = load_runs(runs_dir)
    return analyze_records(records, n_resamples=n_resamples)


def analyze_records(records: list[Record], n_resamples: int = 10000) -> dict[str, Any]:
    """Battery over already-loaded records (the unit tests drive this directly)."""
    return {
        "meta": {
            "n_runs": len(records),
            "stats_path": "numpy+stdlib (scipy "
            + ("available but unused" if SCIPY_AVAILABLE else "not installed") + ")",
            "bootstrap_resamples": n_resamples,
            "alpha": _ALPHA,
        },
        "per_arm": _summary(records, n_resamples),
        "pairwise_wilcoxon_holm": _pairwise(records),
        "friedman_nemenyi": _friedman_across_arms(records),
        "mcnemar_parity": _mcnemar_parity(records),
        "trends": _trends(records),
    }


# --------------------------------------------------------------------------- #
# Human-readable report + CLI.
# --------------------------------------------------------------------------- #


def format_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    meta = summary["meta"]
    lines.append(f"FedGrid statistical battery  ({meta['n_runs']} runs, {meta['stats_path']})")
    lines.append("")
    lines.append("Per-arm macro-F1 (mean +/- std [BCa 95% CI], n):")
    lines.append(f"  {'arm|model|attack':<40} {'mean':>7} {'std':>6}  {'95% CI':>18}  n")
    for key, s in summary["per_arm"].items():
        ci = f"[{s['bca_ci_low']:.3f}, {s['bca_ci_high']:.3f}]"
        lines.append(f"  {key:<40} {s['mean_macro_f1']:>7.3f} {s['std_macro_f1']:>6.3f}"
                     f"  {ci:>18}  {s['n']}")
    lines.append("")
    lines.append("Pairwise vs baseline (Wilcoxon signed-rank, Holm-adjusted):")
    lines.append(f"  {'arm vs baseline (model/attack)':<44} {'d(F1)':>7} {'p_holm':>8}"
                 f" {'delta':>7}  magnitude")
    for e in summary["pairwise_wilcoxon_holm"]:
        tag = f"{e['arm']} vs {e['baseline']} ({e['model']}/{e['attack'] or 'clean'})"
        lines.append(f"  {tag:<44} {e['mean_diff']:>7.3f} {e.get('p_holm', float('nan')):>8.4f}"
                     f" {e['cliffs_delta']:>7.3f}  {e['magnitude']}")
    lines.append("")
    for fr in summary["friedman_nemenyi"]:
        ranks = [round(x, 2) for x in fr["avg_ranks"]]
        lines.append(f"Friedman ({fr['model']}/{fr['attack'] or 'clean'}): "
                     f"Q={fr['Q']:.3f} p={fr['p']:.4f}  arms={fr['arms']}  "
                     f"avg_ranks={ranks}  CD={fr['nemenyi_cd']:.3f}")
    mc = summary["mcnemar_parity"]
    lines.append("")
    if mc["available"]:
        pooled = mc["pooled"]
        lines.append(f"McNemar SNN vs CNN-LSTM (pooled): b={int(pooled['b'])} c={int(pooled['c'])} "
                     f"p={pooled['p']:.4f}  ({len(mc['per_seed'])} seed-pairs)")
    else:
        lines.append(f"McNemar SNN vs CNN-LSTM: unavailable ({mc['reason']})")
    for tr in summary["trends"]:
        jt = tr["jonckheere"]
        lines.append(f"Trend {tr['family']} {tr.get('arm', '')}/{tr.get('model', '')}: "
                     f"JT z={jt['z']:.3f} p_inc={jt['p_one_increasing']:.4f} "
                     f"means={[round(x, 3) for x in tr['group_means']]}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m experiments.analyze",
        description="Compute the FedGrid statistical battery over a runs/ directory.",
    )
    p.add_argument("--runs-dir", default="runs", help="directory of RunResult JSONs")
    p.add_argument("--out", type=Path, default=None, help="write the JSON summary here")
    p.add_argument("--resamples", type=int, default=10000, help="BCa bootstrap resamples")
    args = p.parse_args(argv)

    summary = analyze_dir(args.runs_dir, n_resamples=args.resamples)
    print(format_report(summary))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote JSON summary -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
