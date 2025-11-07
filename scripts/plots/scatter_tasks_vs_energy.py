from __future__ import annotations

import statistics
from pathlib import Path
from typing import List, Mapping, Sequence

from scripts.plot_utils import annotate_without_overlap, ensure_matplotlib, format_scaffold_label


def plot_scatter_tasks_vs_energy(
    grouped_runs: Mapping[str, Sequence["RunSummary"]],
    plots_dir: Path,
) -> None:
    if not grouped_runs:
        return
    plt = ensure_matplotlib()
    labels: List[str] = []
    energies: List[float] = []
    solved: List[float] = []
    used_adjusted = False
    for scaffold, runs in sorted(grouped_runs.items()):
        energy_values = [
            (run.total_energy_adj_j if run.total_energy_adj_j is not None else run.total_energy_j)
            for run in runs
            if (run.total_energy_adj_j is not None) or (run.total_energy_j is not None)
        ]
        if any(run.total_energy_adj_j is not None for run in runs):
            used_adjusted = True
        resolved_values = [float(run.resolved) for run in runs]
        if not energy_values or not resolved_values:
            continue
        labels.append(format_scaffold_label(scaffold))
        energies.append(float(statistics.fmean(energy_values)))
        solved.append(float(statistics.fmean(resolved_values)))
    if not labels:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(energies, solved, color="#2e7d32")
    offset_overrides = {
        "Mini SWE Agent (Gemma)": (-0.08, 0.12),
        "Mini SWE Agent (Qwen)": (0.08, -0.12),
        "ACR (Gemma)": (-0.12, 0.05),
        "ACR (Qwen)": (-0.12, -0.08),
    }
    annotate_without_overlap(
        ax,
        labels,
        energies,
        solved,
        fontsize=8,
        padding=11.0,
        initial_radius=0.12,
        radial_growth=0.05,
        offset_overrides=offset_overrides,
    )
    ax.margins(x=0.18, y=0.25)
    xlabel = "Average Total Energy (J, idle-subtracted)" if used_adjusted else "Average Total Energy (J)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average Issues Resolved")
    ax.set_title("Number of Tasks Solved vs. Energy Usage")
    fig.tight_layout()
    fig.savefig(plots_dir / "scatter_tasks_vs_energy.png", dpi=200)
    plt.close(fig)
