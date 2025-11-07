from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from scripts.plot_utils import ensure_matplotlib, format_scaffold_label


def plot_token_histograms(
    issues: Sequence["IssueMetrics"],
    plots_dir: Path,
) -> None:
    if not issues:
        return
    plt = ensure_matplotlib()
    tokens_by_scaffold: Dict[str, List[float]] = defaultdict(list)
    for issue in issues:
        if issue.total_tokens is not None:
            tokens_by_scaffold[issue.scaffold_dir].append(issue.total_tokens)
    if not tokens_by_scaffold:
        return
    scaffolds = sorted(tokens_by_scaffold.keys())
    positive_values = [
        value for values in tokens_by_scaffold.values() for value in values if value > 0
    ]
    if positive_values:
        x_min = min(positive_values)
        x_max = max(positive_values)
        bins = np.logspace(np.log10(x_min), np.log10(x_max), num=20)
    else:
        x_min = 1.0
        x_max = 10.0
        bins = 20
    cols = min(3, len(scaffolds))
    rows = math.ceil(len(scaffolds) / cols)
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
    for index, scaffold in enumerate(scaffolds):
        ax = axes[index // cols][index % cols]
        series = tokens_by_scaffold[scaffold]
        if positive_values:
            adjusted = [max(value, x_min) for value in series]
            ax.hist(adjusted, bins=bins, color="#66bb6a", alpha=0.85)
        else:
            ax.hist(series, bins=bins, color="#66bb6a", alpha=0.85)
        ax.set_title(format_scaffold_label(scaffold))
        ax.set_xlabel("Tokens per Task (log scale)" if positive_values else "Tokens per Task")
        ax.set_ylabel("Frequency")
        if positive_values:
            ax.set_xscale("log")
            ax.set_xlim(x_min, x_max)
    total_plots = rows * cols
    for index in range(len(scaffolds), total_plots):
        axes[index // cols][index % cols].axis("off")
    fig.tight_layout()
    fig.savefig(plots_dir / "token_usage_histograms.png", dpi=200)
    plt.close(fig)
