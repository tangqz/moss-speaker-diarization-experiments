"""Render the current MOSS RSWA sentinel S and RTF trajectories."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {"128": "#D97706", "full": "#2563EB", "256": "#059669"}
LABELS = {"128": "R128", "full": "Full", "256": "R256"}


def load_rows(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    return sorted(rows, key=lambda row: (row["window"], row["step"]))


def plot_metric(rows, metric, ylabel, title, output):
    fig, ax = plt.subplots(figsize=(11.5, 6.5), dpi=160)
    fig.patch.set_facecolor("#FBFCFE")
    ax.set_facecolor("#FBFCFE")

    for window in ["128", "full", "256"]:
        group = [row for row in rows if row["window"] == window]
        sentinel = [row for row in group if row["sentinel"]]
        full_dev = [row for row in group if not row["sentinel"]]
        color = COLORS[window]
        ax.plot(
            [row["step"] for row in sentinel],
            [row[metric] for row in sentinel],
            color=color,
            linewidth=2.4,
            marker="o",
            markersize=6,
            label=LABELS[window],
        )
        # Full-Dev points are kept visible but are not connected to the
        # sentinel line, since they use the complete 26-meeting evaluation.
        if full_dev:
            ax.scatter(
                [row["step"] for row in full_dev],
                [row[metric] for row in full_dev],
                s=74,
                marker="s",
                facecolors="white",
                edgecolors=color,
                linewidths=1.8,
                zorder=4,
            )
        failed = [row for row in sentinel if not row["functional"]]
        if failed:
            ax.scatter(
                [row["step"] for row in failed],
                [row[metric] for row in failed],
                s=68,
                marker="x",
                color="#991B1B",
                linewidths=1.8,
                zorder=5,
            )

    ax.set_xlabel("Optimizer update")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=14)
    ax.grid(axis="y", color="#D9E0EA", linewidth=0.8)
    ax.grid(axis="x", color="#E9EDF3", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8A97A8")
    ax.spines["bottom"].set_color("#8A97A8")
    ax.set_xticks(sorted({row["step"] for row in rows}))
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ax.text(
        0,
        -0.18,
        "Circles: sentinel (6 meetings)  |  squares: complete Dev (26 meetings)  |  red ×: functional check failed",
        transform=ax.transAxes,
        fontsize=9,
        color="#52606D",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input)
    plot_metric(
        rows,
        "S",
        "S (lower is better)",
        "MOSS RSWA sentinel S by update",
        args.output_dir / "moss_rswa_sentinel_S_20260921.png",
    )
    plot_metric(
        rows,
        "RTF",
        "RTF (lower is faster)",
        "MOSS RSWA sentinel RTF by update",
        args.output_dir / "moss_rswa_sentinel_RTF_20260921.png",
    )


if __name__ == "__main__":
    main()
