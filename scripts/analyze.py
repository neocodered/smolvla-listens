"""eval_info.json of every condition -> results/summary.csv, per_task.csv and two plots.

usage (from the repo root): python scripts/analyze.py [outputs_dir]
Reads outputs/E2_<condition>/eval_info.json and instructions/<condition>.json.
"""
import json
import sys
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONDS = ["original", "paraphrase_a", "paraphrase_b", "empty", "swapped"]
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs")
RES = Path("results")

# Colours from the dataviz reference palette (light surface).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ_BLUES = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def wilson(k, n, z=1.96):
    """95% interval for a success rate k/n that stays sensible at 0/n and n/n."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def load():
    tasks = json.load(open("instructions/tasks.json"))
    rows = []
    for cond in CONDS:
        info = json.load(open(OUT_DIR / f"E2_{cond}" / "eval_info.json"))
        given = tasks if cond == "original" else json.load(open(f"instructions/{cond}.json"))
        for t in info["per_task"]:
            tid = str(t["task_id"])
            succ = t["metrics"]["successes"]
            k, n = int(sum(succ)), len(succ)
            lo, hi = wilson(k, n)
            rows.append({
                "condition": cond, "task_id": int(tid), "original_instruction": tasks[tid],
                "instruction_given": given[tid], "k": k, "n": n,
                "success_pct": 100 * k / n, "ci_low": 100 * lo, "ci_high": 100 * hi,
            })
    return pd.DataFrame(rows)


def summarize(df):
    rows = []
    for cond in CONDS:
        d = df[df.condition == cond]
        k, n = int(d.k.sum()), int(d.n.sum())
        lo, hi = wilson(k, n)
        rows.append({"condition": cond, "k": k, "n": n, "success_pct": 100 * k / n,
                     "ci_low": 100 * lo, "ci_high": 100 * hi})
    return pd.DataFrame(rows)


def paired(out_dir):
    """Same init states in every condition, so compare episode by episode against original.

    McNemar exact test: under "no effect", the discordant pairs split 50/50.
    """
    from math import comb

    def succ(cond):
        info = json.load(open(out_dir / f"E2_{cond}" / "eval_info.json"))
        return {t["task_id"]: t["metrics"]["successes"] for t in info["per_task"]}

    base = succ("original")
    rows = []
    for cond in CONDS[1:]:
        other = succ(cond)
        pairs = [(a, b) for t in base for a, b in zip(base[t], other[t])]
        only_orig = sum(a and not b for a, b in pairs)
        only_cond = sum(b and not a for a, b in pairs)
        m = only_orig + only_cond
        p = min(1.0, 2 * sum(comb(m, i) for i in range(min(only_orig, only_cond) + 1)) / 2**m) if m else 1.0
        rows.append({"condition": cond, "both_succeed": sum(a and b for a, b in pairs),
                     "only_original": only_orig, "only_condition": only_cond, "mcnemar_p": p})
    return pd.DataFrame(rows)


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, length=0)


def plot_bars(summary, path):
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    style(ax)
    x = np.arange(len(summary))
    pct = summary.success_pct.to_numpy()
    err = np.vstack([pct - summary.ci_low, summary.ci_high - pct])
    colors = [ORANGE if c == "swapped" else BLUE for c in summary.condition]
    # The swapped bar is the headline but can be 0 tall, so highlight its column instead.
    sw = list(summary.condition).index("swapped")
    ax.axvspan(sw - 0.42, sw + 0.42, color=ORANGE, alpha=0.12, zorder=1, linewidth=0)
    ax.text(sw, 30, "headline:\ninstruction-ignoring\nrate", ha="center", va="center", color=INK_2, fontsize=8.5)
    ax.bar(x, pct, width=0.6, color=colors, zorder=2)
    ax.errorbar(x, pct, yerr=err, fmt="none", ecolor=INK, elinewidth=1.2, capsize=5, zorder=3)
    for xi, (p, hi, k, n) in enumerate(zip(pct, summary.ci_high, summary.k, summary.n)):
        ax.text(xi, hi + 2.5, f"{p:.0f}%\n({k}/{n})", ha="center", va="bottom", color=INK, fontsize=10)
    ax.set_xticks(x, [c.replace("_", " ") for c in summary.condition], color=INK, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_ylabel("task success", color=INK_2)
    ax.set_title("SmolVLA on LIBERO-Goal: success on the scene's own task, by instruction given",
                 color=INK, fontsize=11, loc="left", pad=12)
    fig.text(0.01, 0.01, "10 tasks × 10 init states per condition. Error bars: 95% Wilson interval. "
             "Swapped = the instruction of task (i+5) mod 10.", color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_heatmap(df, path):
    grid = df.pivot(index="task_id", columns="condition", values="k")[CONDS]
    labels = df[df.condition == "original"].set_index("task_id").original_instruction
    cmap = matplotlib.colors.ListedColormap(SEQ_BLUES)
    norm = matplotlib.colors.BoundaryNorm([0, 0.5, 2, 4, 5.5, 7, 8.5, 9.5, 10.5], cmap.N)
    fig, ax = plt.subplots(figsize=(9.5, 6.2), facecolor=SURFACE)
    ax.imshow(grid.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
    # 2px surface gaps between cells
    ax.set_xticks(np.arange(-0.5, len(CONDS)), minor=True)
    ax.set_yticks(np.arange(-0.5, 10), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iat[i, j]
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=10,
                    color="#ffffff" if v >= 7 else INK)
    ax.set_xticks(range(len(CONDS)), [c.replace("_", " ") for c in CONDS], color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks(range(10), [f"{t}  {labels[t]}" for t in grid.index], color=INK, fontsize=9)
    ax.set_title("Successes out of 10 per task and condition", color=INK, fontsize=11, loc="left", pad=28)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    RES.mkdir(exist_ok=True)
    df = load()
    summary = summarize(df)
    df.to_csv(RES / "per_task.csv", index=False, float_format="%.1f")
    summary.to_csv(RES / "summary.csv", index=False, float_format="%.1f")
    plot_bars(summary, RES / "success_by_condition.png")
    plot_heatmap(df, RES / "per_task_heatmap.png")
    pairs = paired(OUT_DIR)
    pairs.to_csv(RES / "paired_vs_original.csv", index=False, float_format="%.3g")
    print(summary.to_string(index=False, float_format="%.1f"))
    print()
    print(pairs.to_string(index=False, float_format="%.2g"))


if __name__ == "__main__":
    main()
