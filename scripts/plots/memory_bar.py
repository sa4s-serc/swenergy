from __future__ import annotations

import statistics
from pathlib import Path
from typing import List, Mapping, Sequence

import numpy as np

from scripts.plot_utils import ensure_matplotlib, format_scaffold_label


def plot_memory_bar(
    grouped_runs: Mapping[str, Sequence["RunSummary"]],
    plots_dir: Path,
) -> None:
    if not grouped_runs:
        return
    plt = ensure_matplotlib()
    labels: List[str] = []
    ram_means: List[float] = []
    vram_means: List[float] = []
    total_means: List[float] = []
    vram_shares: List[Optional[float]] = []
    for scaffold, runs in sorted(grouped_runs.items()):
        ram_values = [run.mean_ram_mb for run in runs if run.mean_ram_mb is not None]
        vram_values = [run.mean_vram_mb for run in runs if run.mean_vram_mb is not None]
        if not ram_values and not vram_values:
            continue
        ram_mean = float(statistics.fmean(ram_values)) if ram_values else 0.0
        vram_mean = float(statistics.fmean(vram_values)) if vram_values else 0.0
        total_mean = ram_mean + vram_mean
        vram_share = (vram_mean / total_mean * 100.0) if total_mean > 0 else None
        labels.append(scaffold)
        ram_means.append(ram_mean)
        vram_means.append(vram_mean)
        total_means.append(total_mean)
        vram_shares.append(vram_share)
    if not labels:
        return
    display_labels = [format_scaffold_label(name) for name in labels]
    plots_dir.mkdir(parents=True, exist_ok=True)
    x_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.bar(x_pos, ram_means, width=0.6, label="RAM", color="#a5d6a7")
    ax.bar(
        x_pos,
        vram_means,
        width=0.6,
        bottom=ram_means,
        label="VRAM",
        color="#2e7d32",
    )
    ax.set_ylabel("Average Total Memory (MB)")
    ax.set_title("Memory Usage by Scaffold (RAM + VRAM)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_labels, rotation=45, ha="right")

    max_total = max(total_means) if total_means else 0.0
    offset = max_total * 0.02 if max_total else 5.0
    for idx, total in enumerate(total_means):
        ax.text(
            x_pos[idx],
            total + offset,
            f"{total:.0f} MB",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    share_indices = [idx for idx, share in enumerate(vram_shares) if share is not None]
    if share_indices:
        ax2 = ax.twinx()
        share_x = [x_pos[idx] for idx in share_indices]
        share_values = [vram_shares[idx] for idx in share_indices]  # type: ignore[misc]
        ax2.plot(
            share_x,
            share_values,
            color="#1b5e20",
            marker="o",
            linewidth=1.5,
            label="VRAM Share (%)",
        )
        ax2.set_ylabel("VRAM Share (%)")
        ax2.set_ylim(0, 100)
        ax2.set_yticks(range(0, 101, 20))
        ax2.patch.set_visible(False)
        handles, labels_legend = ax.get_legend_handles_labels()
        handles.append(ax2.lines[0])
        labels_legend.append("VRAM Share (%)")
        ax.legend(handles, labels_legend, loc="upper left")
    else:
        ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(plots_dir / "memory_usage_by_scaffold.png", dpi=200)
    plt.close(fig)
