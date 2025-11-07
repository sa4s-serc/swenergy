from __future__ import annotations

import statistics
from pathlib import Path
from typing import List, Mapping, Sequence

from scripts.plot_utils import ensure_matplotlib, format_scaffold_label


def plot_energy_bar(
    grouped_runs: Mapping[str, Sequence["RunSummary"]],
    plots_dir: Path,
) -> None:
    if not grouped_runs:
        return
    plt = ensure_matplotlib()
    labels: List[str] = []
    energies: List[float] = []
    used_adjusted = False
    for scaffold, runs in sorted(grouped_runs.items()):
        # Prefer adjusted totals when present
        energy_values = [
            (run.total_energy_adj_j if run.total_energy_adj_j is not None else run.total_energy_j)
            for run in runs
            if (run.total_energy_adj_j is not None) or (run.total_energy_j is not None)
        ]
        if any(run.total_energy_adj_j is not None for run in runs):
            used_adjusted = True
        if not energy_values:
            continue
        labels.append(scaffold)
        energies.append(float(statistics.fmean(energy_values)))
    if not labels:
        return
    display_labels = [format_scaffold_label(name) for name in labels]
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(display_labels, energies, color="#2e7d32")
    ylabel = "Average Total Energy (J, idle-subtracted)" if used_adjusted else "Average Total Energy (J)"
    ax.set_ylabel(ylabel)
    ax.set_title("Energy Usage by Scaffold")
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(plots_dir / "energy_usage_by_scaffold.png", dpi=200)
    plt.close(fig)
