from __future__ import annotations

import statistics
from pathlib import Path
from typing import List, Mapping, Sequence

from scripts.plot_utils import annotate_without_overlap, ensure_matplotlib, format_scaffold_label


def plot_scatter_time_vs_energy(
    grouped_runs: Mapping[str, Sequence["RunSummary"]],
    plots_dir: Path,
) -> None:
    if not grouped_runs:
        return
    plt = ensure_matplotlib()
    labels: List[str] = []
    energies: List[float] = []
    times: List[float] = []
    used_adjusted = False
    for scaffold, runs in sorted(grouped_runs.items()):
        energy_values = [
            (run.total_energy_adj_j if run.total_energy_adj_j is not None else run.total_energy_j)
            for run in runs
            if (run.total_energy_adj_j is not None) or (run.total_energy_j is not None)
        ]
        if any(run.total_energy_adj_j is not None for run in runs):
            used_adjusted = True
        time_values = [run.mean_time_s for run in runs if run.mean_time_s is not None]
        if not energy_values or not time_values:
            continue
        labels.append(format_scaffold_label(scaffold))
        energies.append(float(statistics.fmean(energy_values)))
        times.append(float(statistics.fmean(time_values)))
    if not labels:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(energies, times, color="#1b5e20")
    annotate_without_overlap(
        ax,
        labels,
        energies,
        times,
        fontsize=8,
        padding=10.0,
        initial_radius=0.08,
        radial_growth=0.035,
    )
    ax.margins(x=0.18, y=0.18)
    xlabel = "Average Total Energy (J, idle-subtracted)" if used_adjusted else "Average Total Energy (J)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average Time per Issue (s)")
    ax.set_title("Time per Issue vs. Energy Usage")
    fig.tight_layout()
    fig.savefig(plots_dir / "scatter_time_vs_energy.png", dpi=200)
    plt.close(fig)
