from __future__ import annotations

import statistics
from pathlib import Path
from typing import List, Mapping, Sequence

from scripts.plot_utils import ensure_matplotlib, format_scaffold_label


def plot_cost_bar(
    grouped_runs: Mapping[str, Sequence["RunSummary"]],
    plots_dir: Path,
) -> None:
    if not grouped_runs:
        return
    plt = ensure_matplotlib()
    labels: List[str] = []
    costs: List[float] = []
    for scaffold, runs in sorted(grouped_runs.items()):
        cost_values = [run.mean_cost_usd for run in runs if run.mean_cost_usd is not None]
        if not cost_values:
            continue
        labels.append(scaffold)
        costs.append(float(statistics.fmean(cost_values)))
    if not labels:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    display_labels = [format_scaffold_label(name) for name in labels]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(display_labels, costs, color="#66bb6a")
    ax.set_ylabel("Average Cost per Task (USD)")
    ax.set_title("Cost per Task by Scaffold")
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(plots_dir / "cost_per_task_by_scaffold.png", dpi=200)
    plt.close(fig)
