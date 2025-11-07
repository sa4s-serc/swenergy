#!/usr/bin/env python3
"""
Utilities for aggregating agent run metrics captured in JSONL logs.

Example:
    python scripts/analyze_results.py \\
        --inputs acr_gemma big_swe_agent_gemma openhands_gemma mini_swe_agent_gemma \\
        --energy-scale 1e-6 \\
        --csv run_summary.csv \\
        --per-issue per_issue_metrics.csv \\
        --plots-dir plots
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.plots import (
    plot_correlation_heatmaps,
    plot_cost_bar,
    plot_memory_bar,
    plot_memory_token_cost_bubble,
    plot_correlation_summary,
    plot_scatter_cpu_vs_gpu_energy,
    plot_scatter_time_vs_energy,
    plot_energy_bar,
    plot_energy_boxplot,
    plot_failure_modes_from_analyses,
    plot_scatter_patches_vs_energy,
    plot_scatter_tasks_vs_energy,
    plot_token_histograms,
)


ENERGY_SUCCESS_DENOM_SENTINEL = 1e-12  # Prevent divide-by-zero for empty groups.


@dataclass
class Record:
    payload: Dict
    source: Path

    @property
    def scaffold(self) -> Optional[str]:
        return self.payload.get("scaffold")

    @property
    def model_name(self) -> Optional[str]:
        model = self.payload.get("model_name")
        if model:
            return model
        return self.payload.get("model_name_or_path")


BYTES_PER_MB = 1024 * 1024
FAILURE_TIMEOUT = "Timeout"
FAILURE_CONTEXT = "Context Overflow"
FAILURE_AGENT_DONE = "Agent Marked Done"
FAILURE_OTHER = "Other"
FAILURE_CATEGORIES = (
    FAILURE_TIMEOUT,
    FAILURE_CONTEXT,
    FAILURE_AGENT_DONE,
    FAILURE_OTHER,
)

@dataclass(frozen=True)
class SBReport:
    path: Path
    base_name: str
    full_key: str
    full_key_no_agent: str
    scaffold_key: str
    scaffold_key_no_agent: str
    run_digits: Set[str]
    resolved_ids: Set[str]
    unresolved_ids: Set[str]

    @property
    def all_ids(self) -> Set[str]:
        return self.resolved_ids | self.unresolved_ids


@dataclass
class IssueMetrics:
    instance_id: str
    source: Path
    scaffold_dir: str
    agent_scaffold: Optional[str]
    model_name: Optional[str]
    resolved: bool
    stop_reason: Optional[str]
    failure_category: Optional[str]
    cpu_energy_j: Optional[float]
    gpu_energy_j: Optional[float]
    total_energy_j: Optional[float]
    time_s: Optional[float]
    llm_calls: Optional[float]
    input_tokens: Optional[float]
    output_tokens: Optional[float]
    total_tokens: Optional[float]
    ram_mb: Optional[float]
    vram_mb: Optional[float]
    memory_mb: Optional[float]
    cost_usd: Optional[float]
    patch_generated: int
    # Optional idle-adjusted energies (per issue) when idle baselines are provided
    cpu_energy_adj_j: Optional[float] = None
    gpu_energy_adj_j: Optional[float] = None
    total_energy_adj_j: Optional[float] = None


@dataclass
class RunSummary:
    source: Path
    scaffold_dir: str
    agent_scaffold: Optional[str]
    model_name: Optional[str]
    issues: int
    resolved: int
    total_energy_j: float
    total_cpu_energy_j: float
    total_gpu_energy_j: float
    mean_energy_j: Optional[float]
    mean_cpu_energy_j: Optional[float]
    mean_gpu_energy_j: Optional[float]
    time_sum_s: float
    mean_time_s: Optional[float]
    llm_calls_total: Optional[float]
    mean_llm_calls: Optional[float]
    total_tokens: Optional[float]
    input_tokens_total: Optional[float]
    output_tokens_total: Optional[float]
    mean_total_tokens: Optional[float]
    mean_input_tokens: Optional[float]
    mean_output_tokens: Optional[float]
    mean_ram_mb: Optional[float]
    mean_vram_mb: Optional[float]
    mean_memory_mb: Optional[float]
    total_cost_usd: Optional[float]
    mean_cost_usd: Optional[float]
    patch_count: int
    failure_counts: Counter
    # Optional idle-adjusted aggregates when idle baselines are provided
    total_energy_adj_j: Optional[float] = None
    total_cpu_energy_adj_j: Optional[float] = None
    total_gpu_energy_adj_j: Optional[float] = None
    mean_energy_adj_j: Optional[float] = None
    mean_cpu_energy_adj_j: Optional[float] = None
    mean_gpu_energy_adj_j: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate metrics from agent benchmark JSONL logs."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[],
        help="List of JSONL files or directories containing JSONL logs.",
    )
    parser.add_argument(
        "--glob",
        default="*.jsonl",
        help="Glob pattern for discovering logs inside input directories (default: %(default)s).",
    )
    parser.add_argument(
        "--energy-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to cpu_energy and gpu_energy to convert to Joules. "
            "For microjoule logs, pass 1e-6."
        ),
    )
    parser.add_argument(
        "--idle-cpu-w",
        type=float,
        default=None,
        help=(
            "Optional CPU idle baseline in Watts (from RAPL). When provided together "
            "with --idle-gpu-w, adjusted energies are computed by subtracting idle*duration."
        ),
    )
    parser.add_argument(
        "--idle-gpu-w",
        type=float,
        default=None,
        help=(
            "Optional GPU idle baseline in Watts (from NVML). When provided together "
            "with --idle-cpu-w, adjusted energies are computed by subtracting idle*duration."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path to write per-run summary metrics as CSV.",
    )
    parser.add_argument(
        "--per-issue",
        type=Path,
        help="Optional path to write per-issue records (without patches) as CSV.",
    )
    parser.add_argument(
        "--include-overall",
        action="store_true",
        help="Print aggregate metrics across all inputs.",
    )
    parser.add_argument(
        "--no-scaffold-summary",
        action="store_true",
        help="Skip scaffold-level aggregation in the textual report.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        help="Optional directory to save summary plots (requires matplotlib).",
    )
    parser.add_argument(
        "--sb-reports",
        nargs="*",
        help="Optional sb-cli report files or directories for resolved issue lookup.",
    )
    parser.add_argument(
        "--table-csv",
        type=Path,
        help="Optional path to export scaffold metrics (mean/min/median/max/std) as CSV.",
    )
    return parser.parse_args()


def is_results_file(path: Path) -> bool:
    """Heuristic to ignore verbose run logs like *_logs.jsonl."""
    stem = path.stem.lower()
    if stem.endswith("_logs") or "_logs_" in stem:
        return False
    return True


def collect_paths(inputs: Sequence[str], file_glob: str) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")
        if path.is_dir():
            for discovered in sorted(path.rglob(file_glob)):
                if discovered.is_file() and is_results_file(discovered):
                    paths.append(discovered)
        else:
            if path.suffix != ".jsonl":
                raise ValueError(f"Expected a .jsonl file, received: {path}")
            if is_results_file(path):
                paths.append(path)
    return paths


def sanitize_scaffold_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def infer_top_level_dir(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(Path.cwd().resolve())
        if relative.parts:
            return relative.parts[0]
    except Exception:
        pass
    parts = [part for part in path.parts if part not in {path.anchor, ""}]
    return parts[0] if parts else path.stem


def load_sb_reports(inputs: Sequence[str]) -> List[SBReport]:
    reports: List[SBReport] = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"SB report input not found: {path}")
        if path.is_dir():
            candidates = sorted(path.glob("*.json"))
        else:
            if path.suffix != ".json":
                raise ValueError(f"Expected .json SB report, received: {path}")
            candidates = [path]
        for candidate in candidates:
            with candidate.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            resolved_ids = set(data.get("resolved_ids") or [])
            unresolved_ids = set(data.get("unresolved_ids") or [])
            base_name = candidate.stem
            clean_base = base_name
            if clean_base.startswith("swe-bench_verified__test__"):
                clean_base = clean_base[len("swe-bench_verified__test__") :]
            if clean_base.endswith("_actual"):
                clean_base = clean_base[: -len("_actual")]
            full_key = sanitize_scaffold_key(clean_base)
            full_key_no_agent = full_key.replace("agent", "")
            scaffold_portion = re.sub(r"\d+", "", clean_base)
            scaffold_key = sanitize_scaffold_key(scaffold_portion)
            scaffold_key_no_agent = scaffold_key.replace("agent", "")
            run_digits = set(re.findall(r"\d+", clean_base))
            reports.append(
                SBReport(
                    path=candidate,
                    base_name=clean_base,
                    full_key=full_key,
                    full_key_no_agent=full_key_no_agent,
                    scaffold_key=scaffold_key,
                    scaffold_key_no_agent=scaffold_key_no_agent,
                    run_digits=run_digits,
                    resolved_ids=resolved_ids,
                    unresolved_ids=unresolved_ids,
                )
            )
    return reports


def match_sb_report(source: Path, reports: Sequence[SBReport]) -> Optional[SBReport]:
    if not reports:
        return None
    scaffold_dir = infer_top_level_dir(source)
    dir_key = sanitize_scaffold_key(scaffold_dir)
    dir_key_no_digits = sanitize_scaffold_key(re.sub(r"\d+", "", scaffold_dir))
    stem_key = sanitize_scaffold_key(source.stem)
    stem_key_no_digits = sanitize_scaffold_key(re.sub(r"\d+", "", source.stem))
    combined_key = sanitize_scaffold_key(f"{scaffold_dir}_{source.stem}")
    combined_key_no_digits = sanitize_scaffold_key(
        re.sub(r"\d+", "", f"{scaffold_dir}_{source.stem}")
    )
    full_candidates = {stem_key, combined_key}
    full_candidates_no_agent = {key.replace("agent", "") for key in full_candidates}
    reduced_candidates = {
        dir_key,
        dir_key_no_digits,
        stem_key_no_digits,
        combined_key_no_digits,
    }
    reduced_candidates.update(full_candidates)
    reduced_candidates_no_agent = {
        key.replace("agent", "") for key in reduced_candidates
    }
    run_digits = set(re.findall(r"\d+", source.stem))
    best_match: Optional[SBReport] = None
    best_score = -1
    for report in reports:
        key_match_score = 0
        if report.full_key in full_candidates:
            key_match_score = 4
        elif report.full_key_no_agent and report.full_key_no_agent in full_candidates_no_agent:
            key_match_score = 3
        elif report.scaffold_key in reduced_candidates:
            key_match_score = 2
        elif report.scaffold_key_no_agent and report.scaffold_key_no_agent in reduced_candidates_no_agent:
            key_match_score = 1
        else:
            continue

        if run_digits and report.run_digits:
            if run_digits & report.run_digits:
                key_match_score += 1
            else:
                continue

        if key_match_score > best_score:
            best_match = report
            best_score = key_match_score
    return best_match


def build_sb_lookup(sources: Sequence[Path], reports: Sequence[SBReport]) -> Dict[Path, SBReport]:
    lookup: Dict[Path, SBReport] = {}
    for source in sources:
        match = match_sb_report(source, reports)
        if match:
            lookup[source] = match
    return lookup


def load_records(paths: Iterable[Path]) -> List[Record]:
    records: List[Record] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:  # pragma: no cover - defensive.
                    raise ValueError(f"Malformed JSON in {path}:{line_no}") from exc
                records.append(Record(payload=payload, source=path))
    return records


def categorize_failure(stop_reason: Optional[str]) -> str:
    if not stop_reason:
        return FAILURE_OTHER
    text = str(stop_reason).strip().lower()
    if "timeout" in text:
        return FAILURE_TIMEOUT
    if "context" in text or "overflow" in text:
        return FAILURE_CONTEXT
    if "agent" in text and "done" in text:
        return FAILURE_AGENT_DONE
    if "marked" in text and "done" in text:
        return FAILURE_AGENT_DONE
    if text == "done":
        return FAILURE_AGENT_DONE
    return FAILURE_OTHER


def is_success(payload: Dict) -> bool:
    outcome = payload.get("outcome")
    if outcome is not None:
        normalized = str(outcome).strip().lower()
        if normalized in {"pass", "passed", "success", "successful", "solved"}:
            return True
        if normalized in {"fail", "failed"}:
            return False
    stop_reason = payload.get("stop_reason")
    if stop_reason is None:
        return False
    return str(stop_reason).strip().lower() in {"successful", "success", "passed"}


def as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return statistics.fmean(values)


def safe_median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    return float(statistics.median(values))


def build_issue_metrics(
    records: Sequence[Record],
    energy_scale: float,
    sb_lookup: Mapping[Path, SBReport],
    idle_cpu_w: Optional[float] = None,
    idle_gpu_w: Optional[float] = None,
) -> List[IssueMetrics]:
    issues: List[IssueMetrics] = []
    for rec in records:
        payload = rec.payload
        instance_id = payload.get("instance_id")
        if not instance_id:
            continue
        scaffold_dir = infer_top_level_dir(rec.source)
        sb_report = sb_lookup.get(rec.source)
        if sb_report:
            resolved = instance_id in sb_report.resolved_ids
        else:
            resolved = is_success(payload)
        stop_reason = payload.get("stop_reason")
        failure_category = categorize_failure(stop_reason) if not resolved else None

        raw_cpu = as_float(payload.get("cpu_energy"))
        if raw_cpu is not None and raw_cpu < 0:
            raw_cpu = None
        raw_gpu = as_float(payload.get("gpu_energy"))
        if raw_gpu is not None and raw_gpu < 0:
            raw_gpu = None
        cpu_energy_j = raw_cpu * energy_scale if raw_cpu is not None else None
        gpu_energy_j = raw_gpu * energy_scale if raw_gpu is not None else None
        total_energy_j = None
        if cpu_energy_j is not None or gpu_energy_j is not None:
            total_energy_j = (cpu_energy_j or 0.0) + (gpu_energy_j or 0.0)

        time_s = as_float(payload.get("time_taken"))
        llm_calls = as_float(payload.get("llm_calls"))
        input_tokens = as_float(payload.get("input_tokens"))
        output_tokens = as_float(payload.get("output_tokens"))
        total_tokens = None
        if input_tokens is not None or output_tokens is not None:
            total_tokens = (input_tokens or 0.0) + (output_tokens or 0.0)

        ram_raw = as_float(payload.get("ram_used"))
        vram_raw = as_float(payload.get("vram_used"))
        ram_mb = ram_raw / BYTES_PER_MB if ram_raw is not None else None
        vram_mb = vram_raw / BYTES_PER_MB if vram_raw is not None else None
        memory_mb = None
        if ram_mb is not None or vram_mb is not None:
            memory_mb = (ram_mb or 0.0) + (vram_mb or 0.0)

        cost_usd = as_float(payload.get("cost"))

        patch_generated = 1 if payload.get("model_patch") else 0

        # Compute idle-adjusted energies if baselines provided
        cpu_adj = None
        gpu_adj = None
        total_adj = None
        if (
            time_s is not None
            and time_s >= 0
            and (idle_cpu_w is not None or idle_gpu_w is not None)
        ):
            if cpu_energy_j is not None and idle_cpu_w is not None:
                cpu_adj = max(cpu_energy_j - idle_cpu_w * time_s, 0.0)
            if gpu_energy_j is not None and idle_gpu_w is not None:
                gpu_adj = max(gpu_energy_j - idle_gpu_w * time_s, 0.0)
            # If only one component available, still compute sensible total
            total_adj = (
                (cpu_adj if cpu_adj is not None else 0.0)
                + (gpu_adj if gpu_adj is not None else 0.0)
            )

        issues.append(
            IssueMetrics(
                instance_id=instance_id,
                source=rec.source,
                scaffold_dir=scaffold_dir,
                agent_scaffold=rec.scaffold,
                model_name=rec.model_name,
                resolved=resolved,
                stop_reason=stop_reason,
                failure_category=failure_category,
                cpu_energy_j=cpu_energy_j,
                gpu_energy_j=gpu_energy_j,
                total_energy_j=total_energy_j,
                time_s=time_s,
                llm_calls=llm_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                ram_mb=ram_mb,
                vram_mb=vram_mb,
                memory_mb=memory_mb,
                cost_usd=cost_usd,
                patch_generated=patch_generated,
                cpu_energy_adj_j=cpu_adj,
                gpu_energy_adj_j=gpu_adj,
                total_energy_adj_j=total_adj,
            )
        )
    return issues


def summarize_runs(issues: Sequence[IssueMetrics]) -> List[RunSummary]:
    grouped: Dict[Path, List[IssueMetrics]] = defaultdict(list)
    for issue in issues:
        grouped[issue.source].append(issue)

    run_summaries: List[RunSummary] = []
    for source, rows in sorted(grouped.items()):
        scaffold_dir = rows[0].scaffold_dir
        agent_scaffold = rows[0].agent_scaffold
        model_name = rows[0].model_name

        issues_count = len(rows)
        resolved_count = sum(1 for row in rows if row.resolved)

        total_energy_j = sum((row.total_energy_j or 0.0) for row in rows if row.total_energy_j is not None)
        total_cpu_energy_j = sum((row.cpu_energy_j or 0.0) for row in rows if row.cpu_energy_j is not None)
        total_gpu_energy_j = sum((row.gpu_energy_j or 0.0) for row in rows if row.gpu_energy_j is not None)
        cpu_energy_values = [row.cpu_energy_j for row in rows if row.cpu_energy_j is not None]
        gpu_energy_values = [row.gpu_energy_j for row in rows if row.gpu_energy_j is not None]
        energy_values = [row.total_energy_j for row in rows if row.total_energy_j is not None]
        mean_energy_j = safe_mean(energy_values)
        mean_cpu_energy_j = safe_mean(cpu_energy_values)
        mean_gpu_energy_j = safe_mean(gpu_energy_values)

        # Optional idle-adjusted aggregates
        total_cpu_energy_adj_j = None
        total_gpu_energy_adj_j = None
        total_energy_adj_j = None
        mean_energy_adj_j = None
        mean_cpu_energy_adj_j = None
        mean_gpu_energy_adj_j = None
        adj_values = [row.total_energy_adj_j for row in rows if row.total_energy_adj_j is not None]
        if adj_values:
            total_energy_adj_j = sum(v or 0.0 for v in adj_values)
            mean_energy_adj_j = safe_mean(adj_values)
        cpu_adj_values = [row.cpu_energy_adj_j for row in rows if row.cpu_energy_adj_j is not None]
        if cpu_adj_values:
            total_cpu_energy_adj_j = sum(v or 0.0 for v in cpu_adj_values)
            mean_cpu_energy_adj_j = safe_mean(cpu_adj_values)
        gpu_adj_values = [row.gpu_energy_adj_j for row in rows if row.gpu_energy_adj_j is not None]
        if gpu_adj_values:
            total_gpu_energy_adj_j = sum(v or 0.0 for v in gpu_adj_values)
            mean_gpu_energy_adj_j = safe_mean(gpu_adj_values)

        time_values = [row.time_s for row in rows if row.time_s is not None]
        time_sum_s = sum(time_values) if time_values else 0.0
        mean_time_s = safe_mean(time_values)

        llm_calls_values = [row.llm_calls for row in rows if row.llm_calls is not None]
        llm_calls_total = sum(llm_calls_values) if llm_calls_values else None
        mean_llm_calls = safe_mean(llm_calls_values)

        input_tokens_values = [row.input_tokens for row in rows if row.input_tokens is not None]
        output_tokens_values = [row.output_tokens for row in rows if row.output_tokens is not None]
        total_tokens_values = [row.total_tokens for row in rows if row.total_tokens is not None]

        input_tokens_total = sum(input_tokens_values) if input_tokens_values else None
        output_tokens_total = sum(output_tokens_values) if output_tokens_values else None
        total_tokens = sum(total_tokens_values) if total_tokens_values else None
        mean_input_tokens = safe_mean(input_tokens_values)
        mean_output_tokens = safe_mean(output_tokens_values)
        mean_total_tokens = safe_mean(total_tokens_values)

        ram_values = [row.ram_mb for row in rows if row.ram_mb is not None]
        vram_values = [row.vram_mb for row in rows if row.vram_mb is not None]
        memory_values = [row.memory_mb for row in rows if row.memory_mb is not None]
        cost_values = [row.cost_usd for row in rows if row.cost_usd is not None]

        mean_ram_mb = safe_mean(ram_values)
        mean_vram_mb = safe_mean(vram_values)
        mean_memory_mb = safe_mean(memory_values)
        total_cost_usd = sum(cost_values) if cost_values else None
        mean_cost_usd = safe_mean(cost_values)

        patch_count = sum(row.patch_generated for row in rows)

        failure_counts: Counter = Counter()
        for row in rows:
            if row.failure_category:
                failure_counts[row.failure_category] += 1

        run_summaries.append(
            RunSummary(
                source=source,
                scaffold_dir=scaffold_dir,
                agent_scaffold=agent_scaffold,
                model_name=model_name,
                issues=issues_count,
                resolved=resolved_count,
                total_energy_j=total_energy_j,
                total_cpu_energy_j=total_cpu_energy_j,
                total_gpu_energy_j=total_gpu_energy_j,
                mean_energy_j=mean_energy_j,
                mean_cpu_energy_j=mean_cpu_energy_j,
                mean_gpu_energy_j=mean_gpu_energy_j,
                time_sum_s=time_sum_s,
                mean_time_s=mean_time_s,
                llm_calls_total=llm_calls_total,
                mean_llm_calls=mean_llm_calls,
                total_tokens=total_tokens,
                input_tokens_total=input_tokens_total,
                output_tokens_total=output_tokens_total,
                mean_total_tokens=mean_total_tokens,
                mean_input_tokens=mean_input_tokens,
                mean_output_tokens=mean_output_tokens,
                mean_ram_mb=mean_ram_mb,
                mean_vram_mb=mean_vram_mb,
                mean_memory_mb=mean_memory_mb,
                total_cost_usd=total_cost_usd,
                mean_cost_usd=mean_cost_usd,
                patch_count=patch_count,
                failure_counts=failure_counts,
                total_energy_adj_j=total_energy_adj_j,
                total_cpu_energy_adj_j=total_cpu_energy_adj_j,
                total_gpu_energy_adj_j=total_gpu_energy_adj_j,
                mean_energy_adj_j=mean_energy_adj_j,
                mean_cpu_energy_adj_j=mean_cpu_energy_adj_j,
                mean_gpu_energy_adj_j=mean_gpu_energy_adj_j,
            )
        )
    return run_summaries


def group_runs_by_scaffold(runs: Sequence[RunSummary]) -> Dict[str, List[RunSummary]]:
    grouped: Dict[str, List[RunSummary]] = defaultdict(list)
    for run in runs:
        grouped[run.scaffold_dir].append(run)
    return grouped


def describe_distribution(values: Sequence[float]) -> Dict[str, Optional[float]]:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return {"mean": None, "min": None, "p50": None, "max": None, "std": None}
    mean = statistics.fmean(filtered)
    min_value = min(filtered)
    max_value = max(filtered)
    median = statistics.median(filtered)
    std = statistics.stdev(filtered) if len(filtered) >= 2 else None
    return {"mean": mean, "min": min_value, "p50": median, "max": max_value, "std": std}


def compute_scaffold_metric_stats(
    grouped_runs: Mapping[str, Sequence[RunSummary]],
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    scaffold_stats: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for scaffold, runs in grouped_runs.items():
        metrics: Dict[str, List[Optional[float]]] = {
            "Time Duration (s)": [run.mean_time_s for run in runs],
            "CPU Energy (J)": [run.mean_cpu_energy_j for run in runs],
            "GPU Energy (J)": [run.mean_gpu_energy_j for run in runs],
            "Energy Usage (J)": [run.mean_energy_j for run in runs],
            "RAM (MB)": [run.mean_ram_mb for run in runs],
            "VRAM (MB)": [run.mean_vram_mb for run in runs],
            "Memory Usage (MB)": [run.mean_memory_mb for run in runs],
            "Token Usage": [run.mean_total_tokens for run in runs],
            "Cost (USD)": [run.mean_cost_usd for run in runs],
            "Patches Generated": [float(run.patch_count) for run in runs],
            "Resolved Issues": [float(run.resolved) for run in runs],
            "Resolve Rate (%)": [
                float(run.resolved) / float(run.issues) * 100.0 if run.issues else None
                for run in runs
            ],
        }
        # Include adjusted energy metrics only if present
        if any(run.mean_cpu_energy_adj_j is not None for run in runs):
            metrics["CPU Energy Adj (J)"] = [run.mean_cpu_energy_adj_j for run in runs]
        if any(run.mean_gpu_energy_adj_j is not None for run in runs):
            metrics["GPU Energy Adj (J)"] = [run.mean_gpu_energy_adj_j for run in runs]
        if any(run.mean_energy_adj_j is not None for run in runs):
            metrics["Energy Usage Adj (J)"] = [run.mean_energy_adj_j for run in runs]
        scaffold_stats[scaffold] = {
            metric: describe_distribution(values)
            for metric, values in metrics.items()
        }
    return scaffold_stats


def run_summary_to_dict(run: RunSummary) -> Dict[str, Optional[float]]:
    resolve_rate = run.resolved / run.issues if run.issues else None
    return {
        "source_file": run.source.as_posix(),
        "scaffold": run.scaffold_dir,
        "model_name": run.model_name,
        "issues": run.issues,
        "resolved": run.resolved,
        "resolve_rate": resolve_rate,
        "total_energy_j": run.total_energy_j,
        "total_cpu_energy_j": run.total_cpu_energy_j,
        "total_gpu_energy_j": run.total_gpu_energy_j,
        "total_energy_adj_j": run.total_energy_adj_j,
        "total_cpu_energy_adj_j": run.total_cpu_energy_adj_j,
        "total_gpu_energy_adj_j": run.total_gpu_energy_adj_j,
        "mean_energy_adj_j": run.mean_energy_adj_j,
        "mean_cpu_energy_adj_j": run.mean_cpu_energy_adj_j,
        "mean_gpu_energy_adj_j": run.mean_gpu_energy_adj_j,
        "mean_energy_j": run.mean_energy_j,
        "mean_cpu_energy_j": run.mean_cpu_energy_j,
        "mean_gpu_energy_j": run.mean_gpu_energy_j,
        "time_sum_s": run.time_sum_s,
        "mean_time_s": run.mean_time_s,
        "total_tokens": run.total_tokens,
        "input_tokens_total": run.input_tokens_total,
        "output_tokens_total": run.output_tokens_total,
        "mean_total_tokens": run.mean_total_tokens,
        "mean_input_tokens": run.mean_input_tokens,
        "mean_output_tokens": run.mean_output_tokens,
        "llm_calls_total": run.llm_calls_total,
        "mean_llm_calls": run.mean_llm_calls,
        "mean_memory_mb": run.mean_memory_mb,
        "mean_ram_mb": run.mean_ram_mb,
        "mean_vram_mb": run.mean_vram_mb,
        "total_cost_usd": run.total_cost_usd,
        "mean_cost_usd": run.mean_cost_usd,
        "patch_count": run.patch_count,
    }


def print_scaffold_metric_table(stats: Mapping[str, Mapping[str, Dict[str, Optional[float]]]]) -> None:
    if not stats:
        print("No scaffold statistics to display.")
        return
    columns = ["Scaffold", "Metric", "Mean", "Min", "50%", "Max", "Std Dev"]
    header = " | ".join(columns)
    print("\nScaffold Metrics Summary")
    print(header)
    print("-" * len(header))
    for scaffold in sorted(stats.keys()):
        metric_map = stats[scaffold]
        for metric_name, summary in metric_map.items():
            row = [
                scaffold,
                metric_name,
                format_value(summary.get("mean")),
                format_value(summary.get("min")),
                format_value(summary.get("p50")),
                format_value(summary.get("max")),
                format_value(summary.get("std")),
            ]
            print(" | ".join(row))


def summarize_group(
    records: List[Record],
    energy_scale: float,
) -> Dict[str, Optional[float]]:
    n = len(records)
    success_count = sum(1 for rec in records if is_success(rec.payload))

    cpu_energy: List[float] = []
    gpu_energy: List[float] = []
    total_energy: List[float] = []
    for rec in records:
        raw_cpu = as_float(rec.payload.get("cpu_energy"))
        raw_gpu = as_float(rec.payload.get("gpu_energy"))
        if raw_cpu is not None:
            cpu_energy.append(raw_cpu * energy_scale)
        if raw_gpu is not None:
            gpu_energy.append(raw_gpu * energy_scale)
        if raw_cpu is not None or raw_gpu is not None:
            total_energy.append(((raw_cpu or 0.0) + (raw_gpu or 0.0)) * energy_scale)

    times = [
        val
        for rec in records
        if (val := as_float(rec.payload.get("time_taken"))) is not None
    ]
    llm_calls = [
        val
        for rec in records
        if (val := as_float(rec.payload.get("llm_calls"))) is not None
    ]
    tool_calls = [
        val
        for rec in records
        if (val := as_float(rec.payload.get("tool_calls"))) is not None
    ]
    input_tokens = [
        val
        for rec in records
        if (val := as_float(rec.payload.get("input_tokens"))) is not None
    ]
    output_tokens = [
        val
        for rec in records
        if (val := as_float(rec.payload.get("output_tokens"))) is not None
    ]
    costs = [
        val
        for rec in records
        if (val := as_float(rec.payload.get("cost"))) is not None
    ]

    total_energy_sum = sum(total_energy)
    resolution_per_kj = None
    if total_energy_sum > ENERGY_SUCCESS_DENOM_SENTINEL and success_count:
        resolution_per_kj = success_count / (total_energy_sum / 1000.0)

    summary = {
        "issues": n,
        "successes": success_count,
        "resolve_rate": success_count / n if n else None,
        "total_energy_j": total_energy_sum if total_energy else None,
        "total_cpu_energy_j": sum(cpu_energy) if cpu_energy else None,
        "total_gpu_energy_j": sum(gpu_energy) if gpu_energy else None,
        "mean_energy_j": safe_mean(total_energy),
        "median_energy_j": safe_median(total_energy),
        "mean_cpu_energy_j": safe_mean(cpu_energy),
        "mean_gpu_energy_j": safe_mean(gpu_energy),
        "resolution_per_kj": resolution_per_kj,
        "mean_time_s": safe_mean(times),
        "median_time_s": safe_median(times),
        "mean_llm_calls": safe_mean(llm_calls),
        "median_llm_calls": safe_median(llm_calls),
        "mean_tool_calls": safe_mean(tool_calls),
        "median_tool_calls": safe_median(tool_calls),
        "mean_input_tokens": safe_mean(input_tokens),
        "median_input_tokens": safe_median(input_tokens),
        "mean_output_tokens": safe_mean(output_tokens),
        "median_output_tokens": safe_median(output_tokens),
        "total_input_tokens": sum(input_tokens) if input_tokens else None,
        "total_output_tokens": sum(output_tokens) if output_tokens else None,
        "total_cost": sum(costs) if costs else None,
    }

    # Derived metric with protection against division by zero.
    if success_count and total_energy_sum > ENERGY_SUCCESS_DENOM_SENTINEL:
        summary["energy_per_success_j"] = total_energy_sum / success_count
    else:
        summary["energy_per_success_j"] = None

    return summary


def format_value(value: Optional[float], precision: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "-"
        return f"{value:.{precision}f}"
    return str(value)


def print_table(rows: List[Dict], title: str, columns: Sequence[str]) -> None:
    if not rows:
        print(f"{title}: no data")
        return

    print(f"\n{title}")
    header = " | ".join(columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        formatted = " | ".join(str(row.get(col, "-")) for col in columns)
        print(formatted)


def summarize_by_source(
    records: Sequence[Record], energy_scale: float
) -> List[Dict[str, Optional[float]]]:
    grouped: Dict[Path, List[Record]] = defaultdict(list)
    for record in records:
        grouped[record.source].append(record)

    summaries: List[Dict[str, Optional[float]]] = []
    for path, recs in sorted(grouped.items()):
        summary = summarize_group(recs, energy_scale)
        summary.update(
            {
                "source_file": path.as_posix(),
                "scaffold": recs[0].scaffold,
                "model_name": recs[0].model_name,
            }
        )
        summaries.append(summary)
    return summaries


def summarize_by_scaffold(
    records: Sequence[Record], energy_scale: float
) -> List[Dict[str, Optional[float]]]:
    grouped: Dict[tuple, List[Record]] = defaultdict(list)
    for record in records:
        key = (record.scaffold, record.model_name)
        grouped[key].append(record)

    summaries: List[Dict[str, Optional[float]]] = []
    for (scaffold, model_name), recs in sorted(grouped.items()):
        summary = summarize_group(recs, energy_scale)
        summary.update({"scaffold": scaffold, "model_name": model_name})
        summaries.append(summary)
    return summaries


def export_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        print(f"Skipping CSV export for {path} (no rows).")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def export_per_issue_metrics(path: Path, issues: Sequence[IssueMetrics]) -> None:
    rows: List[Dict] = []
    for issue in issues:
        rows.append(
            {
                "instance_id": issue.instance_id,
                "source_file": issue.source.as_posix(),
                "scaffold": issue.scaffold_dir,
                "agent_scaffold": issue.agent_scaffold,
                "model_name": issue.model_name,
                "resolved": issue.resolved,
                "stop_reason": issue.stop_reason,
                "failure_category": issue.failure_category,
                "cpu_energy_j": issue.cpu_energy_j,
                "gpu_energy_j": issue.gpu_energy_j,
                "total_energy_j": issue.total_energy_j,
                "cpu_energy_adj_j": issue.cpu_energy_adj_j,
                "gpu_energy_adj_j": issue.gpu_energy_adj_j,
                "total_energy_adj_j": issue.total_energy_adj_j,
                "time_s": issue.time_s,
                "llm_calls": issue.llm_calls,
                "input_tokens": issue.input_tokens,
                "output_tokens": issue.output_tokens,
                "total_tokens": issue.total_tokens,
                "ram_mb": issue.ram_mb,
                "vram_mb": issue.vram_mb,
                "memory_mb": issue.memory_mb,
                "cost_usd": issue.cost_usd,
                "patch_generated": issue.patch_generated,
            }
        )
    export_csv(path, rows)


def attach_display_metrics(
    summaries: List[Dict[str, Optional[float]]],
    precision: int = 3,
) -> List[Dict[str, str]]:
    display_rows: List[Dict[str, str]] = []
    for summary in summaries:
        display_row = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                display_row[key] = format_value(value, precision=precision)
            elif isinstance(value, Path):
                display_row[key] = value.as_posix()
            else:
                display_row[key] = str(value) if value is not None else "-"
        display_rows.append(display_row)
    return display_rows






def main() -> None:
    args = parse_args()
    if not args.inputs:
        raise SystemExit("Provide at least one --inputs path.")

    paths = collect_paths(args.inputs, args.glob)
    if not paths:
        raise SystemExit("No JSONL files discovered from the provided inputs.")

    records = load_records(paths)
    if not records:
        raise SystemExit("No records loaded from the provided JSONL files.")

    sources = sorted({record.source for record in records})

    sb_inputs = list(args.sb_reports or [])
    if not sb_inputs:
        default_reports_dir = Path("sb-cli-reports")
        if default_reports_dir.exists():
            sb_inputs = [str(default_reports_dir)]

    sb_reports: List[SBReport] = []
    if sb_inputs:
        try:
            sb_reports = load_sb_reports(sb_inputs)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Warning: failed to load sb-cli reports ({exc}); falling back to log outcomes.")
            sb_reports = []

    sb_lookup = build_sb_lookup(sources, sb_reports) if sb_reports else {}
    unmatched_sources = sorted({src for src in sources if src not in sb_lookup})
    if sb_reports and unmatched_sources:
        print(
            f"Warning: {len(unmatched_sources)} run(s) had no matching sb-cli report. "
            "Falling back to log stop_reason/outcome for those runs."
        )

    issue_metrics = build_issue_metrics(
        records,
        args.energy_scale,
        sb_lookup,
        idle_cpu_w=args.idle_cpu_w,
        idle_gpu_w=args.idle_gpu_w,
    )
    if not issue_metrics:
        raise SystemExit("No usable issue metrics extracted from the provided JSONL files.")

    run_summaries = summarize_runs(issue_metrics)
    if not run_summaries:
        raise SystemExit("No run-level summaries could be computed.")

    grouped_runs = group_runs_by_scaffold(run_summaries)
    scaffold_stats = compute_scaffold_metric_stats(grouped_runs)

    run_rows = [run_summary_to_dict(run) for run in run_summaries]
    per_run_display = attach_display_metrics(run_rows)
    per_run_columns = [
        "source_file",
        "scaffold",
        "model_name",
        "issues",
        "resolved",
        "resolve_rate",
        "time_sum_s",
        "mean_time_s",
        "total_energy_j",
        "total_cpu_energy_j",
        "total_gpu_energy_j",
        "total_energy_adj_j",
        "total_cost_usd",
        "mean_cost_usd",
        "mean_memory_mb",
        "patch_count",
    ]
    print_table(per_run_display, title="Per-run summary", columns=per_run_columns)

    if not args.no_scaffold_summary:
        print_scaffold_metric_table(scaffold_stats)

    if args.csv:
        export_csv(args.csv, run_rows)

    if args.per_issue:
        export_per_issue_metrics(args.per_issue, issue_metrics)

    if args.table_csv:
        table_rows: List[Dict] = []
        for scaffold, metrics in scaffold_stats.items():
            for metric_name, summary in metrics.items():
                table_rows.append(
                    {
                        "scaffold": scaffold,
                        "metric": metric_name,
                        "mean": summary.get("mean"),
                        "min": summary.get("min"),
                        "p50": summary.get("p50"),
                        "max": summary.get("max"),
                        "std": summary.get("std"),
                    }
                )
        export_csv(args.table_csv, table_rows)

    if args.include_overall:
        resolved_total = sum(1 for issue in issue_metrics if issue.resolved)
        total_issues = len(issue_metrics)
        energy_values = [issue.total_energy_j for issue in issue_metrics if issue.total_energy_j is not None]
        cpu_values = [issue.cpu_energy_j for issue in issue_metrics if issue.cpu_energy_j is not None]
        gpu_values = [issue.gpu_energy_j for issue in issue_metrics if issue.gpu_energy_j is not None]
        cost_values = [issue.cost_usd for issue in issue_metrics if issue.cost_usd is not None]
        time_values = [issue.time_s for issue in issue_metrics if issue.time_s is not None]
        token_values = [issue.total_tokens for issue in issue_metrics if issue.total_tokens is not None]
        memory_values = [issue.memory_mb for issue in issue_metrics if issue.memory_mb is not None]
        print("\nOverall aggregate")
        print(f"Issues: {total_issues}")
        print(f"Resolved: {resolved_total}")
        print(f"Resolve rate: {resolved_total / total_issues:.3f}" if total_issues else "Resolve rate: -")
        print(f"Total energy (J): {sum(energy_values):.3f}" if energy_values else "Total energy (J): -")
        print(f"Total CPU energy (J): {sum(cpu_values):.3f}" if cpu_values else "Total CPU energy (J): -")
        print(f"Total GPU energy (J): {sum(gpu_values):.3f}" if gpu_values else "Total GPU energy (J): -")
        print(f"Total cost (USD): {sum(cost_values):.3f}" if cost_values else "Total cost (USD): -")
        print(f"Mean cost (USD): {statistics.fmean(cost_values):.3f}" if cost_values else "Mean cost (USD): -")
        print(f"Mean time (s): {statistics.fmean(time_values):.3f}" if time_values else "Mean time (s): -")
        print(f"Mean tokens: {statistics.fmean(token_values):.3f}" if token_values else "Mean tokens: -")
        print(f"Mean memory usage (MB): {statistics.fmean(memory_values):.3f}" if memory_values else "Mean memory usage (MB): -")

    if args.plots_dir:
        plot_scatter_tasks_vs_energy(grouped_runs, args.plots_dir)
        plot_scatter_cpu_vs_gpu_energy(grouped_runs, args.plots_dir)
        plot_scatter_patches_vs_energy(grouped_runs, args.plots_dir)
        plot_memory_bar(grouped_runs, args.plots_dir)
        plot_cost_bar(grouped_runs, args.plots_dir)
        plot_memory_token_cost_bubble(grouped_runs, args.plots_dir)
        plot_scatter_time_vs_energy(grouped_runs, args.plots_dir)
        plot_energy_bar(grouped_runs, args.plots_dir)
        plot_energy_boxplot(issue_metrics, args.plots_dir)
        plot_failure_modes_from_analyses(Path("gemini_analyses"), args.plots_dir, issue_metrics)
        plot_token_histograms(issue_metrics, args.plots_dir)
        plot_correlation_heatmaps(issue_metrics, args.plots_dir, sanitize_scaffold_key)
        plot_correlation_summary(issue_metrics, args.plots_dir)


if __name__ == "__main__":
    main()
