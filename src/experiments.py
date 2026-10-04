"""Backtests of the Gumbel-tail forecast (Jones et al., arXiv:2502.16797) on the scored prompt sets.

E1 reproduction: evaluation and deployment both drawn from set A (in distribution).
E2 shift:        evaluation from set A, deployment from set B (unseen roleplay wrappers).
E2b reference:   evaluation and deployment both from set B (does the method work on B itself?).
E3 recalibration: keep the slope fitted on m set-A prompts, re-estimate the intercept from k set-B
                  prompts; compare with set-A-only and set-B-only fits.

As in Jones et al. Section 4.2, every forecast uses a disjoint evaluation set and deployment set, and
the truth is the largest elicitation probability in the deployment set. Each experiment is repeated
over many random shuffles. Errors are in log10 units: forecast minus truth (negative = underestimate).

Usage: python src/experiments.py   -> results/e1_reproduction.csv, e2_shift.csv, e3_recalibration.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.forecast import (gumbel_fit, gumbel_forecast, intercept_given_slope, lognormal_forecast,
                          psi_from_log10p)

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261003


def load(metric: str = "log10p_any") -> tuple[np.ndarray, np.ndarray]:
    a = pd.read_parquet(ROOT / "results" / "scores_A.parquet")[metric].to_numpy()
    b = pd.read_parquet(ROOT / "results" / "scores_B.parquet")[metric].to_numpy()
    return a, b


def _row(method, m, n, forecast, truth, **extra):
    return dict(method=method, m=m, n=n, forecast=forecast, truth=truth, err=forecast - truth, **extra)


def backtest(eval_pool, dep_pool, ms, ns, reps, rng, label, same_pool):
    """Disjoint evaluation (size m) and deployment (size n) sets; truth = max log10 p in deployment."""
    rows = []
    for rep in range(reps):
        for m in ms:
            for n in ns:
                if same_pool:
                    if m + n > len(eval_pool):
                        continue
                    idx = rng.permutation(len(eval_pool))
                    groups = len(eval_pool) // (m + n)
                    for g in range(groups):
                        block = idx[g * (m + n):(g + 1) * (m + n)]
                        ev, dep = eval_pool[block[:m]], eval_pool[block[m:]]
                        rows += _forecasts(ev, dep, m, n, label, rep)
                else:
                    if n > len(dep_pool) or m > len(eval_pool):
                        continue
                    didx = rng.permutation(len(dep_pool))
                    for g in range(len(dep_pool) // n):
                        dep = dep_pool[didx[g * n:(g + 1) * n]]
                        ev = eval_pool[rng.choice(len(eval_pool), m, replace=False)]
                        rows += _forecasts(ev, dep, m, n, label, rep)
    return pd.DataFrame(rows)


def _forecasts(ev, dep, m, n, label, rep):
    psi = psi_from_log10p(ev)
    a, b = gumbel_fit(psi)
    truth = dep.max()
    return [_row("gumbel_tail", m, n, float(gumbel_forecast(a, b, n)), truth, setting=label, rep=rep, slope=a),
            _row("log_normal", m, n, float(lognormal_forecast(psi, n)), truth, setting=label, rep=rep, slope=np.nan)]


def recalibration(a_pool, b_pool, m=1000, ks=(25, 50, 100, 200, 500), ns=(2000, 5000, 10000), reps=200,
                  rng=None):
    """Hybrid: slope from m set-A prompts, intercept from k set-B prompts (top j = min(10, max(3, k // 10)))."""
    rows = []
    for rep in range(reps):
        for k in ks:
            j = min(10, max(3, k // 10))
            for n in ns:
                if k + n > len(b_pool):
                    continue
                bidx = rng.permutation(len(b_pool))
                early, dep = b_pool[bidx[:k]], b_pool[bidx[k:k + n]]
                ev_a = a_pool[rng.choice(len(a_pool), m, replace=False)]
                psi_a, psi_b = psi_from_log10p(ev_a), psi_from_log10p(early)
                a_A, b_A = gumbel_fit(psi_a)
                truth = dep.max()
                f_a = float(gumbel_forecast(a_A, b_A, n))
                b_h = intercept_given_slope(psi_b, a_A, j)
                f_h = float(gumbel_forecast(a_A, b_h, n))
                a_B, b_B = gumbel_fit(psi_b, j)
                f_b = float(gumbel_forecast(a_B, b_B, n)) if a_B < 0 else 0.0  # a >= 0: no decaying tail, forecast p = 1
                for meth, f in (("A_only", f_a), ("hybrid", f_h), ("B_only", f_b)):
                    rows.append(_row(meth, m, n, f, truth, k=k, rep=rep))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = df.groupby(keys)
    return pd.DataFrame({
        "forecasts": g.size(),
        "mean_abs_log10_err": g.err.apply(lambda e: e.abs().mean()),
        "median_err": g.err.median(),
        "within_1_order": g.err.apply(lambda e: (e.abs() <= 1).mean()),
        "underestimate": g.err.apply(lambda e: (e < 0).mean()),
        "median_truth": g.truth.median(),
    }).round(3)


def main(metric: str = "log10p_any", reps: int = 50) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    a, b = load(metric)
    ns = [1000, 2000, 5000, 10000]
    e1 = backtest(a, None, [100, 500, 1000], ns + [20000, 50000, 90000], reps, rng, "A->A", True)
    e2 = backtest(a, b, [100, 500, 1000], ns + [19000], reps, rng, "A->B", False)
    e2b = backtest(b, None, [100, 500, 1000], ns + [18000], reps, rng, "B->B", True)
    e3 = recalibration(a, b, rng=rng, reps=reps * 4)
    out = ROOT / "results"
    tag = "" if metric == "log10p_any" else f"_{metric}"
    pd.concat([e1, e2, e2b]).to_csv(out / f"backtests{tag}.csv", index=False)
    e3.to_csv(out / f"e3_recalibration{tag}.csv", index=False)
    return dict(e1=e1, e2=e2, e2b=e2b, e3=e3)


if __name__ == "__main__":
    main()
