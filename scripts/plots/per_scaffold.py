from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from scripts.plot_utils import ensure_matplotlib


def plot_per_scaffold(
    per_scaffold: List[Dict[str, Optional[float]]],
    plots_dir: Path,
) -> None:
    if not per_scaffold:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt = ensure_matplotlib()

    labels = [
        f"{item.get('scaffold')} / {item.get('model_name')}" for item in per_scaffold
    ]
    resolve_rates = [item.get("resolve_rate") or 0.0 for item in per_scaffold]
    energy_per_success = [
        (item.get("energy_per_success_j") or 0.0) / 1000.0 for item in per_scaffold
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, resolve_rates, color="#8064a2")
    ax.set_ylabel("Resolve rate")
    ax.set_ylim(0, 1)
    ax.set_title("Resolve rate by scaffold/model")
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(plots_dir / "scaffold_resolve_rate.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, energy_per_success, color="#4bacc6")
    ax.set_ylabel("Energy per success (kJ)")
    ax.set_title("Energy per success by scaffold/model")
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(plots_dir / "scaffold_energy_per_success.png", dpi=200)
    plt.close(fig)
