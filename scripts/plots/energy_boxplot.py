from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from scripts.plot_utils import ensure_matplotlib, format_scaffold_label


def plot_energy_boxplot(
    issues: Sequence["IssueMetrics"],
    plots_dir: Path,
) -> None:
    if not issues:
        return
    plt = ensure_matplotlib()
    energy_by_scaffold: Dict[str, List[float]] = defaultdict(list)
    used_adjusted = False
    for issue in issues:
        value = None
        # Prefer idle-adjusted energy when available
        if getattr(issue, "total_energy_adj_j", None) is not None:
            value = issue.total_energy_adj_j
            used_adjusted = True
        elif issue.total_energy_j is not None:
            value = issue.total_energy_j
        if value is not None:
            energy_by_scaffold[issue.scaffold_dir].append(value)
    if not energy_by_scaffold:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(energy_by_scaffold.keys())
    display_labels = [format_scaffold_label(label) for label in labels]
    data = [energy_by_scaffold[label] for label in labels]
    fig, ax = plt.subplots(figsize=(10, 6))
    boxprops = dict(color="#1b5e20", linewidth=1.2)
    whiskerprops = dict(color="#1b5e20", linewidth=1.0)
    capprops = dict(color="#1b5e20", linewidth=1.0)
    medianprops = dict(color="#2e7d32", linewidth=1.2)
    bp = ax.boxplot(
        data,
        labels=display_labels,
        showfliers=False,
        patch_artist=True,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        medianprops=medianprops,
    )
    for box in bp["boxes"]:
        box.set(facecolor="#a5d6a7", alpha=0.8)
    for idx, values in enumerate(data, start=1):
        if not values:
            continue
        if len(values) == 1:
            offsets = [0.0]
        else:
            offsets = np.linspace(-0.1, 0.1, len(values))
        xs = np.full(len(values), idx) + offsets
        ax.scatter(xs, values, alpha=0.7, color="#2e7d32", s=18)
    ylabel = "Per-Task Energy Usage (J, idle-subtracted)" if used_adjusted else "Per-Task Energy Usage (J)"
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(plots_dir / "energy_usage_boxplot.png", dpi=200)
    plt.close(fig)
