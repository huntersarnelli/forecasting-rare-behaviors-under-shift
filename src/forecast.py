"""Forecasting methods from Jones et al., "Forecasting Rare Language Model Behaviors", arXiv:2502.16797.

Gumbel-tail method (Section 3.3): with elicitation score psi = -log(-log p), the log survival function
of psi is approximately linear in its upper tail, log S(psi) = a * psi + b. Fit a, b by ordinary least
squares on the ten highest evaluation scores, then the 1/n quantile is
    Q_psi(n) = -(log n + b) / a,
from log(1/n) = a * Q_psi(n) + b (Equations 3-4). Equation 5 in the paper prints -(1/a)(log n - b),
a sign slip; we implement Equations 3-4. The forecast worst-query risk is Q_p(n) = exp(-exp(-Q_psi(n))).

The paper does not say which empirical survival probability goes with the i-th largest score;
we use i / m (config.yaml, fit.survival_convention).

Log-normal baseline (Section 4.1): psi ~ Normal(mu, sigma) fitted to all m evaluation scores.
The paper uses the expected maximum of n draws; we use the 1 - 1/n quantile, the same target as
the Gumbel-tail forecast.

Recalibration (this project): keep the slope a fitted on the large evaluation set, re-estimate only
the intercept b from a small sample of shifted prompts.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

LN10 = np.log(10)


def psi_from_log10p(log10p: np.ndarray) -> np.ndarray:
    """Elicitation score psi = -ln(-ln p). ln p is clipped below 0 so p ~ 1 stays finite."""
    lnp = np.minimum(np.asarray(log10p, dtype=float) * LN10, -1e-12)
    return -np.log(-lnp)


def log10p_from_psi(psi: np.ndarray | float) -> np.ndarray:
    """Inverse of psi_from_log10p: log10 of p = exp(-exp(-psi))."""
    return -np.exp(-np.asarray(psi, dtype=float)) / LN10


def tail_points(psi: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """The k highest scores and their log empirical survival probabilities ln(i / m)."""
    m = len(psi)
    top = np.sort(psi)[::-1][:k]
    return top, np.log(np.arange(1, len(top) + 1) / m)


def gumbel_fit(psi: np.ndarray, k: int = 10) -> tuple[float, float]:
    """OLS fit of log S = a * psi + b on the k highest scores. Returns (a, b); a < 0 for a decaying tail."""
    x, y = tail_points(psi, k)
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def intercept_given_slope(psi: np.ndarray, a: float, k: int = 10) -> float:
    """Least-squares intercept b for a fixed slope a, on the k highest scores of `psi`."""
    x, y = tail_points(psi, k)
    return float(np.mean(y - a * x))


def gumbel_forecast(a: float, b: float, n: int | np.ndarray) -> np.ndarray:
    """Forecast log10 worst-query risk over n queries."""
    q_psi = -(np.log(n) + b) / a
    return log10p_from_psi(q_psi)


def lognormal_forecast(psi: np.ndarray, n: int | np.ndarray) -> np.ndarray:
    """Log-normal baseline: 1 - 1/n quantile of a normal fitted to all evaluation scores."""
    mu, sigma = float(np.mean(psi)), float(np.std(psi, ddof=1))
    return log10p_from_psi(mu + sigma * norm.ppf(1 - 1 / np.asarray(n, dtype=float)))
