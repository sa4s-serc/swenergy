from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from scripts.plot_utils import ensure_matplotlib


def plot_failure_stacked(
    grouped_runs: Mapping[str, Sequence["RunSummary"]],
    plots_dir: Path,
    failure_categories: Sequence[str],
) -> None:
    if not grouped_runs or not failure_categories:
        return
    plt = ensure_matplotlib()
    scaffolds = sorted(grouped_runs.keys())
    if not scaffolds:
        return
    category_counts = {category: [] for category in failure_categories}
    for scaffold in scaffolds:
        aggregate = Counter()
        for run in grouped_runs[scaffold]:
            aggregate.update(run.failure_counts)
        for category in failure_categories:
            category_counts[category].append(aggregate.get(category, 0))
    total_failures = sum(sum(values) for values in category_counts.values())
    if total_failures == 0:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(scaffolds))
    colors = ["#c0504d", "#4f81bd", "#9bbb59", "#808080"]
    for color, category in zip(colors, failure_categories):
        values = category_counts[category]
        ax.bar(scaffolds, values, bottom=bottom, label=category, color=color)
        bottom += np.array(values)
    ax.set_ylabel("Issue Count")
    ax.set_title("Failure Reasons per Scaffold")
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "failure_reasons_stacked.png", dpi=200)
    plt.close(fig)
