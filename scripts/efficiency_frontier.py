#!/usr/bin/env python3
"""
Construct efficiency frontiers that balance task resolution against energy usage.

Example:
    python scripts/efficiency_frontier.py \\
        --inputs acr_gemma big_swe_agent_gemma openhands_gemma mini_swe_agent_gemma \\
        --energy-scale 1e-6 \\
        --max-time 1200 \\
        --max-vram-gb 14 \\
        --plots-dir plots
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from analyze_results import (
    Record,
    collect_paths,
    is_success,
    load_records,
    as_float,
    ensure_matplotlib,
)

GIB = 1024 ** 3


@dataclass
class ConstraintConfig:
    max_time: Optional[float]
    max_vram_bytes: Optional[float]
    max_ram_bytes: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive energy-efficiency frontiers for agent scaffolds."
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
        help="Glob pattern when scanning directories (default: %(default)s).",
    )
    parser.add_argument(
        "--energy-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to cpu/gpu energy to obtain Joules (use 1e-6 for microjoules).",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        help="Optional per-issue wall-clock limit in seconds; records exceeding this are dropped.",
    )
    parser.add_argument(
        "--max-vram-gb",
        type=float,
        help="Optional per-issue VRAM ceiling in GiB; records exceeding this are dropped.",
    )
    parser.add_argument(
        "--max-ram-gb",
        type=float,
        help="Optional per-issue RAM ceiling in GiB; records exceeding this are dropped.",
    )
    parser.add_argument(
        "--per-scaffold",
        action="store_true",
        help="Aggregate by (scaffold, model) instead of per run file.",
    )
    parser.add_argument(
        "--show-dropped",
        action="store_true",
        help="Include counts of records removed by constraints in the output.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        help="Optional directory to save frontier plots (requires matplotlib).",
    )
    return parser.parse_args()


def build_constraints(args: argparse.Namespace) -> ConstraintConfig:
    return ConstraintConfig(
        max_time=args.max_time,
        max_vram_bytes=args.max_vram_gb * GIB if args.max_vram_gb else None,
        max_ram_bytes=args.max_ram_gb * GIB if args.max_ram_gb else None,
    )


def record_satisfies_constraints(record: Record, cfg: ConstraintConfig) -> bool:
    payload = record.payload
    if cfg.max_time is not None:
        t = as_float(payload.get("time_taken"))
        if t is not None and t > cfg.max_time:
            return False
    if cfg.max_vram_bytes is not None:
        vram = as_float(payload.get("vram_used"))
        if vram is not None and vram > cfg.max_vram_bytes:
            return False
    if cfg.max_ram_bytes is not None:
        ram = as_float(payload.get("ram_used"))
        if ram is not None and ram > cfg.max_ram_bytes:
            return False
    return True


def group_records(
    records: Iterable[Record], per_scaffold: bool
) -> Dict[Tuple, List[Record]]:
    grouped = defaultdict(list)
    for record in records:
        if per_scaffold:
            key = (record.payload.get("scaffold"), record.payload.get("model_name"))
        else:
            key = record.source.as_posix()
        grouped[key].append(record)
    return grouped


def summarize_group(
    key,
    records: List[Record],
    cfg: ConstraintConfig,
    energy_scale: float,
) -> Optional[Dict]:
    kept: List[Record] = []
    dropped = 0
    for rec in records:
        if record_satisfies_constraints(rec, cfg):
            kept.append(rec)
        else:
            dropped += 1

    if not kept:
        return None

    success_count = sum(1 for rec in kept if is_success(rec.payload))
    total_energy = 0.0
    for rec in kept:
        cpu = as_float(rec.payload.get("cpu_energy")) or 0.0
        gpu = as_float(rec.payload.get("gpu_energy")) or 0.0
        total_energy += (cpu + gpu) * energy_scale

    mean_time = None
    times = [as_float(rec.payload.get("time_taken")) for rec in kept]
    times = [t for t in times if t is not None]
    if times:
        mean_time = sum(times) / len(times)

    denom = len(kept) if kept else 0
    resolve_rate = success_count / denom if denom else 0.0

    scaffold = kept[0].payload.get("scaffold")
    model = kept[0].payload.get("model_name")

    summary = {
        "key": key,
        "scaffold": scaffold,
        "model_name": model,
        "issues_considered": denom,
        "successes": success_count,
        "resolve_rate": resolve_rate,
        "total_energy_j": total_energy,
        "mean_energy_per_issue_j": total_energy / denom if denom else None,
        "energy_per_success_j": (total_energy / success_count) if success_count else None,
        "mean_time_s": mean_time,
        "dropped": dropped,
    }
    return summary


def build_frontier(points: Sequence[Dict]) -> List[Dict]:
    sorted_points = sorted(
        [p for p in points if p.get("total_energy_j") is not None],
        key=lambda item: item["total_energy_j"],
    )
    frontier: List[Dict] = []
    best_success = -1
    for point in sorted_points:
        successes = point.get("successes", 0)
        if successes >= best_success:
            frontier.append(point)
            best_success = successes
    return frontier


def format_summary(summary: Dict, show_dropped: bool) -> str:
    parts = [
        summary.get("scaffold", "-") or "-",
        summary.get("model_name", "-") or "-",
        f"{summary.get('issues_considered', 0)}",
        f"{summary.get('successes', 0)}",
        f"{summary.get('resolve_rate', 0.0):.3f}",
        f"{summary.get('total_energy_j', 0.0):.2f}",
        (
            "-"
            if summary.get("energy_per_success_j") is None
            else f"{summary['energy_per_success_j']:.2f}"
        ),
        (
            "-"
            if summary.get("mean_time_s") is None
            else f"{summary['mean_time_s']:.1f}"
        ),
    ]
    if show_dropped:
        parts.append(str(summary.get("dropped", 0)))
    return " | ".join(parts)


def plot_summaries(
    summaries: List[Dict],
    frontier: List[Dict],
    plots_dir: Path,
) -> None:
    if not summaries:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt = ensure_matplotlib()

    colors = [
        "#4f81bd",
        "#c0504d",
        "#9bbb59",
        "#8064a2",
        "#4bacc6",
        "#f79646",
    ]
    scaffold_palette = {}
    for summary in summaries:
        scaffold = summary.get("scaffold") or "unknown"
        if scaffold not in scaffold_palette:
            scaffold_palette[scaffold] = colors[len(scaffold_palette) % len(colors)]

    fig, ax = plt.subplots(figsize=(8, 6))
    for summary in summaries:
        energy = (summary.get("total_energy_j") or 0.0) / 1000.0
        successes = summary.get("successes") or 0
        scaffold = summary.get("scaffold") or "unknown"
        color = scaffold_palette[scaffold]
        ax.scatter(energy, successes, color=color, label=scaffold, s=60, alpha=0.7)
        ax.annotate(
            summary.get("label") or "",
            (energy, successes),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    if frontier:
        frontier_sorted = sorted(
            frontier, key=lambda item: item.get("total_energy_j") or float("inf")
        )
        energies = [(item.get("total_energy_j") or 0.0) / 1000.0 for item in frontier_sorted]
        successes = [item.get("successes") or 0 for item in frontier_sorted]
        ax.plot(energies, successes, color="black", linestyle="--", linewidth=1.5, label="Frontier")

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="lower right")
    ax.set_xlabel("Total energy (kJ)")
    ax.set_ylabel("Success count")
    ax.set_title("Energy vs. successes")
    fig.tight_layout()
    fig.savefig(plots_dir / "frontier_scatter.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    energy_per_success = [
        ((summary.get("energy_per_success_j") or 0.0) / 1000.0, summary.get("resolve_rate") or 0.0, summary)
        for summary in summaries
        if summary.get("energy_per_success_j") is not None
    ]
    if energy_per_success:
        xs = [item[0] for item in energy_per_success]
        ys = [item[1] for item in energy_per_success]
        labels = [item[2].get("label") or "" for item in energy_per_success]
        colors_points = [
            scaffold_palette.get(item[2].get("scaffold") or "unknown", "#4f81bd")
            for item in energy_per_success
        ]
        ax.scatter(xs, ys, color=colors_points, s=60, alpha=0.7)
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xlabel("Energy per success (kJ)")
        ax.set_ylabel("Resolve rate")
        ax.set_title("Resolve rate vs. energy per success")
    else:
        ax.axis("off")
        ax.set_title("Resolve rate vs. energy per success\n(no data)")

    fig.tight_layout()
    fig.savefig(plots_dir / "resolve_vs_energy.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    constraints = build_constraints(args)
    paths = collect_paths(args.inputs, args.glob)
    records = load_records(paths)
    grouped = group_records(records, args.per_scaffold)

    summaries: List[Dict] = []
    for key, recs in grouped.items():
        summary = summarize_group(key, recs, constraints, args.energy_scale)
        if summary is not None:
            summary["label"] = (
                f"{summary.get('scaffold', '-') or '-'}::{summary.get('model_name', '-') or '-'}"
                if args.per_scaffold
                else str(key)
            )
            summaries.append(summary)

    if not summaries:
        raise SystemExit("No summaries available after applying constraints.")

    header_parts = [
        "label",
        "scaffold",
        "model",
        "issues",
        "successes",
        "resolve_rate",
        "total_energy_j",
        "energy_per_success_j",
        "mean_time_s",
    ]
    if args.show_dropped:
        header_parts.append("dropped")
    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))
    for summary in sorted(
        summaries,
        key=lambda s: (s.get("scaffold") or "", s.get("total_energy_j") or float("inf")),
    ):

        print(f"{summary.get('label', '-') or '-'} | {format_summary(summary, args.show_dropped)}")

    frontier = build_frontier(summaries)
    print("\nPareto frontier (successes maximised, energy minimised)")
    print(header)
    print("-" * len(header))
    for summary in frontier:
        print(f"{summary.get('label', '-') or '-'} | {format_summary(summary, args.show_dropped)}")

    if args.plots_dir:
        plot_summaries(summaries, frontier, args.plots_dir)


if __name__ == "__main__":
    main()
