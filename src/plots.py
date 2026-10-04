"""Figures for the write-up. Reads results/*.csv and scores; writes results/fig_*.png.

Usage: python src/plots.py   (run src/experiments.py first)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def _band(ax, x, lo, mid, hi, color, label, ls="-", marker="o"):
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, mid, color=color, lw=2, ls=ls, marker=marker, ms=5, label=label)


def fig_headline():
    bt = pd.read_csv(OUT / "backtests.csv")
    g = bt[(bt.method == "gumbel_tail") & (bt.m == 1000)]
    fig, ax = plt.subplots(figsize=(7, 4.2))

    fc = g[g.setting == "A->A"].groupby("n").forecast.quantile([0.1, 0.5, 0.9]).unstack()
    ta = g[g.setting == "A->A"].groupby("n").truth.quantile([0.1, 0.5, 0.9]).unstack()
    tb = g[g.setting == "A->B"].groupby("n").truth.quantile([0.1, 0.5, 0.9]).unstack()
    for df, color, label, ls, mk in ((tb, ORANGE, "Actual worst case, roleplay prompts (set B)", "-", "o"),
                                     (ta, BLUE, "Actual worst case, plain prompts (set A)", "-", "o"),
                                     (fc, MUTED, "Forecast from 1,000 set A prompts", "--", "s")):
        _band(ax, df.index, 10 ** df[0.1], 10 ** df[0.5], 10 ** df[0.9], color, label, ls, mk)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(0.02, 1.3)
    ax.set_xlabel("Deployment size n (prompts)")
    ax.set_ylabel("Worst-case leak probability\n(riskiest prompt among n)")
    ax.set_title("A forecast built on plain prompts misses the roleplay shift", loc="left", color=INK, fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.text(1.0, -0.2, "Lines: medians over repeated disjoint splits. Bands: 10th–90th percentile.",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=MUTED)
    fig.savefig(OUT / "fig1_headline.png")
    plt.close(fig)


def fig_tails():
    a = pd.read_parquet(OUT / "scores_A.parquet").log10p_any.to_numpy()
    b = pd.read_parquet(OUT / "scores_B.parquet").log10p_any.to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for x, color, label in ((b, ORANGE, "Roleplay prompts (set B, 20,000)"),
                            (a, BLUE, "Plain prompts (set A, 100,000)")):
        xs = np.sort(x)[::-1]
        frac = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(10 ** xs, frac, color=color, lw=2, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-8, 1.3); ax.set_ylim(3e-6, 1.2)
    ax.set_xlabel("Leak probability p")
    ax.set_ylabel("Fraction of prompts that leak\nwith probability above p")
    ax.set_title("Roleplay fattens the tail, not just shifts it", loc="left", color=INK, fontsize=11)
    ax.legend(loc="lower left", fontsize=8.5)
    fig.savefig(OUT / "fig2_tails.png")
    plt.close(fig)


def fig_recalibration(n=2000):
    e3 = pd.read_csv(OUT / "e3_recalibration.csv")
    d = e3[e3.n == n]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    truth = 10 ** d.truth.median()
    ax.axhline(truth, color=INK, lw=1.2, ls=":")
    ax.text(d.k.max() * 1.05, truth, f" actual\n {truth:.2f}", va="center", fontsize=8, color=INK)
    labels = {"B_only": ("Fit only on the k roleplay prompts", ORANGE, "o"),
              "hybrid": ("Slope from set A, level from k roleplay prompts", AQUA, "D"),
              "A_only": ("Set A only (no roleplay data)", BLUE, "s")}
    for meth, (label, color, mk) in labels.items():
        q = d[d.method == meth].groupby("k").forecast.quantile([0.1, 0.5, 0.9]).unstack()
        _band(ax, q.index, 10 ** q[0.1], 10 ** q[0.5], 10 ** q[0.9], color, label, "-", mk)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks([25, 50, 100, 200, 500]); ax.set_xticklabels(["25", "50", "100", "200", "500"])
    ax.set_ylim(0.03, 1.3)
    ax.set_xlabel("Roleplay prompts available for recalibration (k)")
    ax.set_ylabel(f"Forecast worst-case leak probability\nover n = {n:,} roleplay prompts")
    ax.set_title("A few dozen shifted prompts fix the forecast; keeping set A's slope does not",
                 loc="left", color=INK, fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.savefig(OUT / "fig3_recalibration.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_headline(); fig_tails(); fig_recalibration()
