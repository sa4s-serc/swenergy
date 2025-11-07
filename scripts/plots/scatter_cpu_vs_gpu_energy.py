from __future__ import annotations

import statistics
from pathlib import Path
from typing import List, Mapping, Sequence

from scripts.plot_utils import annotate_without_overlap, ensure_matplotlib, format_scaffold_label


def plot_scatter_cpu_vs_gpu_energy(
    grouped_runs: Mapping[str, Sequence["RunSummary"]],
    plots_dir: Path,
) -> None:
    if not grouped_runs:
        return
    plt = ensure_matplotlib()
    labels: List[str] = []
    cpu_means: List[float] = []
    gpu_means: List[float] = []
    used_adjusted = False
    for scaffold, runs in sorted(grouped_runs.items()):
        cpu_values = [
            (run.total_cpu_energy_adj_j if run.total_cpu_energy_adj_j is not None else run.total_cpu_energy_j)
            for run in runs
            if (run.total_cpu_energy_adj_j is not None) or (run.total_cpu_energy_j is not None)
        ]
        gpu_values = [
            (run.total_gpu_energy_adj_j if run.total_gpu_energy_adj_j is not None else run.total_gpu_energy_j)
            for run in runs
            if (run.total_gpu_energy_adj_j is not None) or (run.total_gpu_energy_j is not None)
        ]
        if any(run.total_cpu_energy_adj_j is not None or run.total_gpu_energy_adj_j is not None for run in runs):
            used_adjusted = True
        if not cpu_values or not gpu_values:
            continue
        labels.append(format_scaffold_label(scaffold))
        cpu_means.append(float(statistics.fmean(cpu_values)))
        gpu_means.append(float(statistics.fmean(gpu_values)))
    if not labels:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cpu_means, gpu_means, color="#388e3c")
    annotate_without_overlap(
        ax,
        labels,
        cpu_means,
        gpu_means,
        fontsize=8,
        padding=10.0,
        initial_radius=0.07,
        radial_growth=0.03,
    )
    ax.margins(x=0.18, y=0.2)
    xlabel = "Average CPU Energy (J, idle-subtracted)" if used_adjusted else "Average CPU Energy (J)"
    ylabel = "Average GPU Energy (J, idle-subtracted)" if used_adjusted else "Average GPU Energy (J)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("CPU vs. GPU Energy by Scaffold")
    fig.tight_layout()
    fig.savefig(plots_dir / "scatter_cpu_vs_gpu_energy.png", dpi=200)
    plt.close(fig)
