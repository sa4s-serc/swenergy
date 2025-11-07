from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

from scripts.plot_utils import ensure_matplotlib
from scripts.plots.correlation_heatmaps import (
    CORRELATION_FEATURES,
    compute_correlation_matrix,
)


def plot_correlation_summary(
    issues: Sequence["IssueMetrics"],
    plots_dir: Path,
    *,
    top_n: int = 10,
) -> None:
    if not issues:
        return
    labels, matrix = compute_correlation_matrix(issues, CORRELATION_FEATURES)
    if matrix.size == 0 or not labels:
        return

    pairs: List[Tuple[str, str, float]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            value = float(matrix[i, j])
            pairs.append((labels[i], labels[j], value))
    if not pairs:
        return

    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    top_pairs = pairs[: top_n or len(pairs)]
    y_labels = [f"{a} ↔ {b}" for a, b, _ in top_pairs]
    values = [value for _, _, value in top_pairs]
    colors = ["#2e7d32" if value >= 0 else "#a5d6a7" for value in values]

    plt = ensure_matplotlib()
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(top_pairs) + 1.5))
    ax.barh(y_labels, values, color=colors)
    ax.set_xlabel("Pearson Correlation")
    ax.set_title("Top Feature Correlations (All Scaffolds)")
    ax.set_xlim(-1, 1)
    for idx, value in enumerate(values):
        ax.text(
            value + (0.02 if value >= 0 else -0.02),
            idx,
            f"{value:.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
        )
    ax.invert_yaxis()
    ax.axvline(0, color="#1b5e20", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(plots_dir / "correlation_top_pairs.png", dpi=200)
    plt.close(fig)
