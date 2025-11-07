from __future__ import annotations

import statistics
from pathlib import Path
from typing import List, Mapping, Sequence

from scripts.plot_utils import annotate_without_overlap, ensure_matplotlib, format_scaffold_label


def plot_memory_token_cost_bubble(
    grouped_runs: Mapping[str, Sequence["RunSummary"]],
    plots_dir: Path,
) -> None:
    if not grouped_runs:
        return
    plt = ensure_matplotlib()
    labels: List[str] = []
    tokens: List[float] = []
    memory_totals: List[float] = []
    costs: List[float] = []
    for scaffold, runs in sorted(grouped_runs.items()):
        token_values = [
            run.total_tokens / run.issues
            for run in runs
            if run.total_tokens is not None and run.issues
        ]
        memory_values = [run.mean_memory_mb for run in runs if run.mean_memory_mb is not None]
        cost_values = [run.mean_cost_usd for run in runs if run.mean_cost_usd is not None]
        if not token_values or not memory_values or not cost_values:
            continue
        labels.append(scaffold)
        tokens.append(float(statistics.fmean(token_values)))
        memory_totals.append(float(statistics.fmean(memory_values)))
        costs.append(float(statistics.fmean(cost_values)))
    if not labels:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    cost_min = min(costs)
    cost_max = max(costs)
    if cost_min == cost_max:
        sizes = [400.0 for _ in costs]
    else:
        sizes = [200.0 + 600.0 * (c - cost_min) / (cost_max - cost_min) for c in costs]
    scatter = ax.scatter(tokens, memory_totals, s=sizes, alpha=0.7, color="#66bb6a", edgecolor="#1b5e20")
    display_labels = [format_scaffold_label(name) for name in labels]
    annotate_without_overlap(ax, display_labels, tokens, memory_totals, fontsize=8, padding=8.0)
    ax.set_xlabel("Average Tokens per Task")
    ax.set_ylabel("Average Memory Usage (MB)")
    ax.set_title("Tokens vs Memory vs Cost by Scaffold")
    if cost_min != cost_max:
        legend_sizes = [200.0, 500.0, 800.0]
        legend_labels = [
            f"${cost_min + (cost_max - cost_min) * (size - 200.0) / 600.0:.3f}"
            for size in legend_sizes
        ]
        handles = [
            ax.scatter([], [], s=size, color="#66bb6a", alpha=0.7, edgecolor="#1b5e20")
            for size in legend_sizes
        ]
        ax.legend(handles, legend_labels, title="Mean Cost (USD)", loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "memory_token_cost_bubble.png", dpi=200)
    plt.close(fig)
