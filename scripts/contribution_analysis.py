#!/usr/bin/env python3
"""
Investigate which scaffold-level factors correlate with energy usage and latency.

Example:
    python scripts/contribution_analysis.py \\
        --inputs acr_gemma big_swe_agent_gemma openhands_gemma mini_swe_agent_gemma \\
        --energy-scale 1e-6 \\
        --plots-dir plots
"""
from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from analyze_results import (
    Record,
    as_float,
    collect_paths,
    ensure_matplotlib,
    load_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute correlations between scaffold activity and energy/latency."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="JSONL files or directories containing run logs.",
    )
    parser.add_argument(
        "--glob",
        default="*.jsonl",
        help="Glob pattern used when scanning directories (default: %(default)s).",
    )
    parser.add_argument(
        "--energy-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to cpu/gpu energy to obtain Joules (use 1e-6 for microjoules).",
    )
    parser.add_argument(
        "--min-sample",
        type=int,
        default=5,
        help="Minimum number of datapoints required to report a correlation (default: %(default)s).",
    )
    parser.add_argument(
        "--per-scaffold",
        action="store_true",
        help="Report correlations per (scaffold, model) group instead of a single global summary.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        help="Optional directory to save correlation plots (requires matplotlib).",
    )
    return parser.parse_args()


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    denom = denom_x * denom_y
    if denom == 0:
        return None
    return num / denom


def collect_metrics(records: Iterable[Record], energy_scale: float) -> List[Dict]:
    rows: List[Dict] = []
    for rec in records:
        payload = rec.payload
        total_energy = 0.0
        cpu = as_float(payload.get("cpu_energy"))
        gpu = as_float(payload.get("gpu_energy"))
        if cpu is not None:
            total_energy += cpu
        if gpu is not None:
            total_energy += gpu
        if total_energy == 0.0:
            continue
        row = {
            "total_energy_j": total_energy * energy_scale,
            "cpu_energy_j": (cpu or 0.0) * energy_scale,
            "gpu_energy_j": (gpu or 0.0) * energy_scale,
            "time_taken": as_float(payload.get("time_taken")),
            "llm_calls": as_float(payload.get("llm_calls")),
            "tool_calls": as_float(payload.get("tool_calls")),
            "input_tokens": as_float(payload.get("input_tokens")),
            "output_tokens": as_float(payload.get("output_tokens")),
            "scaffold": payload.get("scaffold"),
            "model_name": payload.get("model_name"),
        }
        rows.append(row)
    return rows


def prepare_groups(rows: List[Dict], per_scaffold: bool) -> Dict[Tuple, List[Dict]]:
    grouped = defaultdict(list)
    if per_scaffold:
        for row in rows:
            grouped[(row["scaffold"], row["model_name"])].append(row)
    else:
        grouped[("ALL", "ALL")].extend(rows)
    return grouped


def describe_correlations(
    grouped: Dict[Tuple, List[Dict]],
    min_sample: int,
) -> List[Dict[str, Optional[float]]]:
    correlations: List[Dict[str, Optional[float]]] = []
    for (scaffold, model_name), rows in grouped.items():
        if len(rows) < min_sample:
            continue
        energy = [row["total_energy_j"] for row in rows]
        latency = [row["time_taken"] for row in rows if row["time_taken"] is not None]
        corr_entry: Dict[str, Optional[float]] = {
            "scaffold": scaffold,
            "model_name": model_name,
            "sample_size": len(rows),
        }
        for feature in ("llm_calls", "tool_calls", "input_tokens", "output_tokens"):
            xs = [row[feature] for row in rows if row[feature] is not None]
            ys = [row["total_energy_j"] for row in rows if row[feature] is not None]
            corr_entry[f"energy_vs_{feature}"] = pearson(xs, ys)
            if latency:
                lt = [row["time_taken"] for row in rows if row[feature] is not None and row["time_taken"] is not None]
                xs_for_latency = [row[feature] for row in rows if row[feature] is not None and row["time_taken"] is not None]
                corr_entry[f"time_vs_{feature}"] = pearson(xs_for_latency, lt)
        corr_entry["energy_vs_time"] = pearson(
            [row["total_energy_j"] for row in rows if row["time_taken"] is not None],
            [row["time_taken"] for row in rows if row["time_taken"] is not None],
        )
        correlations.append(corr_entry)
    return correlations


def format_value(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def plot_correlations(correlations: List[Dict], plots_dir: Path) -> None:
    if not correlations:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt = ensure_matplotlib()
    features = ["llm_calls", "tool_calls", "input_tokens", "output_tokens"]

    for corr in correlations:
        label = f"{corr.get('scaffold')}__{corr.get('model_name')}"
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        datasets = [
            (
                "Energy correlations",
                [corr.get(f"energy_vs_{feature}") for feature in features],
                axes[0],
            ),
            (
                "Time correlations",
                [corr.get(f"time_vs_{feature}") for feature in features],
                axes[1],
            ),
        ]
        for title, values, axis in datasets:
            filtered = [
                (feature.replace("_", " "), value)
                for feature, value in zip(features, values)
                if value is not None
            ]
            if not filtered:
                axis.axis("off")
                axis.set_title(f"{title}\n(no data)")
                continue
            labels, collected = zip(*filtered)
            axis.bar(labels, collected, color="#4bacc6")
            axis.set_ylim(-1, 1)
            axis.set_title(title)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.tick_params(axis="x", labelrotation=45)
            for tick in axis.get_xticklabels():
                tick.set_horizontalalignment("right")

        fig.suptitle(f"Pearson correlations for {label}")
        fig.tight_layout()
        fig.savefig(plots_dir / f"correlations_{label}.png", dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = collect_paths(args.inputs, args.glob)
    records = load_records(paths)
    rows = collect_metrics(records, args.energy_scale)
    grouped = prepare_groups(rows, args.per_scaffold)
    corrs = describe_correlations(grouped, args.min_sample)
    if not corrs:
        raise SystemExit("Insufficient data to compute correlations.")

    columns = [
        "scaffold",
        "model_name",
        "sample_size",
        "energy_vs_llm_calls",
        "energy_vs_tool_calls",
        "energy_vs_input_tokens",
        "energy_vs_output_tokens",
        "energy_vs_time",
        "time_vs_llm_calls",
        "time_vs_tool_calls",
        "time_vs_input_tokens",
        "time_vs_output_tokens",
    ]
    header = " | ".join(columns)
    print(header)
    print("-" * len(header))
    for corr in corrs:
        row = []
        for column in columns:
            value = corr.get(column)
            if column in {"scaffold", "model_name", "sample_size"}:
                row.append(str(value))
            else:
                row.append(format_value(value))
        print(" | ".join(row))

    if args.plots_dir:
        plot_correlations(corrs, args.plots_dir)


if __name__ == "__main__":
    main()
