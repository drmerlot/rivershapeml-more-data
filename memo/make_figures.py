"""Figures for the memo, from the pipeline outputs in data/derived/.

  fig_errors.png  : held-out error and bias by drainage area, existing RiverShapeML data vs the expanded set (A vs B2)
  fig_product.png : median bankfull width vs drainage area for the four products, with the 90% prediction band

Usage: python memo/make_figures.py   (after ./run_all.sh)
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DER, MEMO = ROOT / "data/derived", ROOT / "memo"
sys.path.insert(0, str(ROOT / "src"))
from metrics import summarize  # noqa: E402

BLUE, ORANGE, GREEN, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e0"
LABELS = ["<10", "10-100", "100-1k", "1k-10k", ">10k"]
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.dpi": 220})


def fig_errors():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
    x = np.arange(len(LABELS))
    for row, (target, name) in enumerate([("TW_bf", "Bankfull width"), ("Y_bf", "Bankfull depth")]):
        p = pd.read_parquet(DER / f"runs/riverml_{target}/predictions.parquet")
        s = summarize(p[p.variant.isin(["A_status_quo", "B2_plus_new_sites"])], ["da_bin"])
        s["da_bin"] = s.da_bin.astype(str)
        for col, (metric, title) in enumerate([("median_abs_err_%", "Median absolute error (%)"),
                                               ("bias_%", "Median bias (%)")]):
            ax = axes[row, col]
            for k, (v, color, lab) in enumerate([("A_status_quo", BLUE, "Existing RiverShapeML data (A)"),
                                                 ("B2_plus_new_sites", GREEN, "Expanded training set (B2)")]):
                vals = s[s.variant == v].set_index("da_bin").reindex(LABELS)[metric].to_numpy()
                bars = ax.bar(x + (k - 0.5) * 0.36, vals, 0.34, color=color, label=lab, edgecolor="white", linewidth=1)
                ax.bar_label(bars, labels=[f"{abs(v):.0f}" if abs(v) < 0.5 else f"{v:.0f}" for v in vals],
                             fontsize=7, color=MUTED, padding=1.5)
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.set_title(f"{name}: {title.lower()}", fontsize=9, color=INK, loc="left")
            ax.set_xticks(x, [f"{l}" for l in LABELS])
            if row == 1:
                ax.set_xlabel("Drainage area (km²)")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(MEMO / "fig_errors.png")
    plt.close(fig)


def fig_product():
    prod = pd.read_parquet(DER / "product_compare.parquet")
    edges = np.logspace(-1, 6.5, 31)
    prod["bin"] = pd.cut(prod.totdasqkm, edges)
    g = prod.groupby("bin", observed=True)
    mid = np.array([np.sqrt(iv.left * iv.right) for iv in g.size().index])
    n = g.size().to_numpy()
    keep = n >= 50
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axvspan(0.1, 5, color="#f1efe8", zorder=0)
    ax.text(0.13, 950, "44% of reaches;\n~50 training sites", fontsize=7.5, color=MUTED, va="top")
    lo = g.ours_tw_bf_m_p05.median().to_numpy()[keep]
    hi = g.ours_tw_bf_m_p95.median().to_numpy()[keep]
    ax.fill_between(mid[keep], lo, hi, color=GREEN, alpha=0.15, linewidth=0, label="90% prediction interval (expanded-set retrain)")
    for col, color, lab in [("deployed_tw_bf_m", BLUE, "Current model"), ("hydroshare_tw_bf_m", ORANGE, "2024 release"),
                            ("ours_tw_bf_m", GREEN, "Expanded-set retrain")]:
        ax.plot(mid[keep], g[col].median().to_numpy()[keep], color=color, linewidth=2, label=lab)
    ax.plot(mid[keep], g.bieger_tw_bf_m.median().to_numpy()[keep], color=MUTED, linewidth=1.6, linestyle=(0, (4, 3)),
            label="Bieger et al. (2015) regional curve")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 3e6)
    ax.set_ylim(1, 1500)
    ax.set_xlabel("Drainage area (km²)")
    ax.set_ylabel("Median bankfull width (m)")
    ax.grid(True, which="major", axis="both", color=GRID, linewidth=0.6)
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(MEMO / "fig_product.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_errors()
    fig_product()
    print("wrote", MEMO / "fig_errors.png", MEMO / "fig_product.png")
