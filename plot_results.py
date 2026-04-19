"""
plot_results.py — Reproduce the main figure from the Sim-OPRL paper.

Usage:
    python plot_results.py
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

RESULTS_PATH = "results/experiment_results.pkl"
OUTPUT_PATH = "results/main_figure.png"

COLORS = {
    "uniform": "#d62728",       # red
    "uncertainty": "#ff7f0e",   # orange
    "simoprl": "#1f77b4",       # blue
}
LABELS = {
    "uniform": "Uniform OPRL (baseline)",
    "uncertainty": "Uncertainty OPRL (baseline)",
    "simoprl": "Sim-OPRL (ours)",
}


def load_results():
    with open(RESULTS_PATH, "rb") as f:
        return pickle.load(f)


def plot(data: dict, checkpoints: list):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")

    for method in ["uniform", "uncertainty", "simoprl"]:
        seed_results = data[method]           # list of {q: return} dicts
        qs = sorted(checkpoints)
        means, stds = [], []

        for q in qs:
            vals = [r[q] for r in seed_results if q in r]
            means.append(np.mean(vals) if vals else float("nan"))
            stds.append(np.std(vals) if vals else 0.0)

        means = np.array(means)
        stds = np.array(stds)
        color = COLORS[method]
        lw = 2.5 if method == "simoprl" else 1.8
        zorder = 3 if method == "simoprl" else 2

        ax.plot(qs, means, "-o", color=color, linewidth=lw,
                markersize=5, label=LABELS[method], zorder=zorder)
        ax.fill_between(qs, means - stds, means + stds,
                        alpha=0.15, color=color, zorder=zorder - 1)

    # Reference lines
    ax.axhline(y=500, color="green", linestyle="--", linewidth=1, alpha=0.5, label="Max return (500)")
    ax.axhline(y=21, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="Random policy (~21)")

    ax.set_xlabel("Number of Preference Queries", fontsize=13)
    ax.set_ylabel("Policy Return (True Reward)", fontsize=13)
    ax.set_title("Sim-OPRL: Sample-Efficient Preference Elicitation\n"
                 "CartPole-v1 · Mean ± Std across 5 seeds",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 540)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    data = load_results()
    plot(data["results"], data["checkpoints"])
