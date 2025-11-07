from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from scripts.plot_utils import ensure_matplotlib


def plot_per_run(per_source: List[Dict[str, Optional[float]]], plots_dir: Path) -> None:
    if not per_source:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt = ensure_matplotlib()

    labels = [Path(item["source_file"]).name for item in per_source if item.get("source_file")]
    resolve_rates = [item.get("resolve_rate") or 0.0 for item in per_source]
    total_energy_kj = [
        (item.get("total_energy_j") or 0.0) / 1000.0 for item in per_source
    ]
    mean_time = [item.get("mean_time_s") or 0.0 for item in per_source]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, resolve_rates, color="#4f81bd")
    ax.set_ylabel("Resolve rate")
    ax.set_ylim(0, 1)
    ax.set_title("Per-run resolve rate")
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(plots_dir / "per_run_resolve_rate.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, total_energy_kj, color="#c0504d")
    ax.set_ylabel("Total energy (kJ)")
    ax.set_title("Per-run total energy")
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(plots_dir / "per_run_energy.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(total_energy_kj, mean_time, c="#9bbb59")
    for label, energy, time in zip(labels, total_energy_kj, mean_time):
        ax.annotate(label, (energy, time), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Total energy (kJ)")
    ax.set_ylabel("Mean wall time (s)")
    ax.set_title("Energy vs. time per run")
    fig.tight_layout()
    fig.savefig(plots_dir / "per_run_energy_time.png", dpi=200)
    plt.close(fig)
