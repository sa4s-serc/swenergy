from __future__ import annotations

import statistics
from pathlib import Path
from typing import List, Mapping, Sequence

from scripts.plot_utils import annotate_without_overlap, ensure_matplotlib, format_scaffold_label


def plot_scatter_patches_vs_energy(
    grouped_runs: Mapping[str, Sequence["RunSummary"]],
    plots_dir: Path,
) -> None:
    if not grouped_runs:
        return
    plt = ensure_matplotlib()
    labels: List[str] = []
    energies: List[float] = []
    patches: List[float] = []
    used_adjusted = False
    for scaffold, runs in sorted(grouped_runs.items()):
        energy_values = [
            (run.total_energy_adj_j if run.total_energy_adj_j is not None else run.total_energy_j)
            for run in runs
            if (run.total_energy_adj_j is not None) or (run.total_energy_j is not None)
        ]
        if any(run.total_energy_adj_j is not None for run in runs):
            used_adjusted = True
        patch_values = [float(run.patch_count) for run in runs]
        if not energy_values:
            continue
        labels.append(format_scaffold_label(scaffold))
        energies.append(float(statistics.fmean(energy_values)))
        patches.append(float(statistics.fmean(patch_values)))
    if not labels:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(energies, patches, color="#66bb6a")
    annotate_without_overlap(
        ax,
        labels,
        energies,
        patches,
        fontsize=8,
        padding=8.0,
        initial_radius=0.05,
        radial_growth=0.025,
    )
    ax.margins(x=0.15, y=0.2)
    xlabel = "Average Total Energy (J, idle-subtracted)" if used_adjusted else "Average Total Energy (J)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average Patches Generated")
    ax.set_title("Number of Patches Generated vs. Energy Usage")
    fig.tight_layout()
    fig.savefig(plots_dir / "scatter_patches_vs_energy.png", dpi=200)
    plt.close(fig)
