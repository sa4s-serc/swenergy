from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts.plot_utils import ensure_matplotlib

CORRELATION_FEATURES: Sequence[Tuple[str, str]] = (
    ("total_energy_j", "Total Energy (J)"),
    ("time_s", "Time (s)"),
    ("ram_mb", "RAM (MB)"),
    ("vram_mb", "VRAM (MB)"),
    ("input_tokens", "Input Tokens"),
    ("output_tokens", "Output Tokens"),
    ("llm_calls", "LLM Calls"),   
)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return None
    corr = np.corrcoef(x_arr, y_arr)[0, 1]
    if np.isnan(corr):
        return None
    return float(corr)


def compute_correlation_matrix(
    issues: Sequence["IssueMetrics"],
    features: Sequence[Tuple[str, str]],
) -> Tuple[List[str], np.ndarray]:
    available_features: List[Tuple[str, str]] = []
    for key, label in features:
        values = [getattr(issue, key) for issue in issues]
        numeric = [value for value in values if value is not None]
        if len(numeric) >= 2:
            available_features.append((key, label))
    if not available_features:
        return [], np.zeros((0, 0))

    size = len(available_features)
    matrix = np.zeros((size, size), dtype=float)
    labels = [label for _, label in available_features]

    for i, (key_i, _) in enumerate(available_features):
        matrix[i, i] = 1.0
        for j in range(i + 1, size):
            key_j, _ = available_features[j]
            pairs_x: List[float] = []
            pairs_y: List[float] = []
            for issue in issues:
                val_i = getattr(issue, key_i)
                val_j = getattr(issue, key_j)
                if val_i is None or val_j is None:
                    continue
                pairs_x.append(val_i)
                pairs_y.append(val_j)
            corr = pearson(pairs_x, pairs_y) if len(pairs_x) >= 2 else None
            value = corr if corr is not None else 0.0
            matrix[i, j] = value
            matrix[j, i] = value
    return labels, matrix


def plot_correlation_heatmaps(
    issues: Sequence["IssueMetrics"],
    plots_dir: Path,
    sanitize_scaffold_key: Callable[[str], str],
    features: Sequence[Tuple[str, str]] = CORRELATION_FEATURES,
) -> None:
    if not issues:
        return
    plt = ensure_matplotlib()
    plots_dir.mkdir(parents=True, exist_ok=True)
    issues_by_scaffold: Dict[str, List["IssueMetrics"]] = defaultdict(list)
    for issue in issues:
        issues_by_scaffold[issue.scaffold_dir].append(issue)

    def render_heatmap(label: str, matrix: np.ndarray, feature_labels: List[str], filename: str) -> None:
        if matrix.size == 0 or not feature_labels:
            return
        fig, ax = plt.subplots(figsize=(1.2 * len(feature_labels), 1.2 * len(feature_labels)))
        im = ax.imshow(matrix, cmap="PiYG", vmin=-1, vmax=1)
        ax.set_xticks(range(len(feature_labels)))
        ax.set_xticklabels(feature_labels, rotation=45, ha="right", fontsize=12)
        ax.set_yticks(range(len(feature_labels)))
        ax.set_yticklabels(feature_labels, fontsize=12)
        ax.tick_params(axis="both", which="major", labelsize=12)
        for i in range(len(feature_labels)):
            for j in range(len(feature_labels)):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=11,
                )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=12)
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=200)
        plt.close(fig)

    # Prefer adjusted energy features when available
    if any(getattr(issue, "total_energy_adj_j", None) is not None for issue in issues):
        remap = {
            "total_energy_j": ("total_energy_adj_j", "Total Energy (J, adj)"),
            "cpu_energy_j": ("cpu_energy_adj_j", "CPU Energy (J, adj)"),
            "gpu_energy_j": ("gpu_energy_adj_j", "GPU Energy (J, adj)"),
        }
        features = tuple(remap.get(k, (k, lbl)) for k, lbl in features)

    overall_labels, overall_matrix = compute_correlation_matrix(issues, features)
    render_heatmap("All Scaffolds", overall_matrix, overall_labels, "correlation_all_scaffolds.png")

    for scaffold, scaffold_issues in sorted(issues_by_scaffold.items()):
        labels, matrix = compute_correlation_matrix(scaffold_issues, features)
        filename = f"correlation_{sanitize_scaffold_key(scaffold)}.png"
        render_heatmap(scaffold, matrix, labels, filename)
