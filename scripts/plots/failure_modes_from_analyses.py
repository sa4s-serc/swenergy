from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from scripts.plot_utils import ensure_matplotlib, format_scaffold_label


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        newline_idx = stripped.find("\n")
        if newline_idx != -1:
            stripped = stripped[newline_idx + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")]
    return stripped.strip()


def _load_analysis(path: Path) -> Optional[Dict]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    text = _strip_markdown_fence(raw)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


AGENT_ALIASES: Dict[str, str] = {
    "autocoderover_gemma": "acr_gemma",
    "autocoderover_qwen": "acr_qwen",
}

METADATA_FILES: Dict[str, Set[str]] = {
    "acr_gemma": {"acr_gemma_1.jsonl"},
    "acr_qwen": {"acr_qwen_1.jsonl"},
    "swe_agent_gemma": {"swe-agent_run_1.jsonl"},
    "swe_agent_qwen": {"swe-agent_run_1.jsonl"},
    "mini_swe_agent_gemma": {"results_run_1.jsonl"},
    "mini_swe_agent_qwen": {"results_run_1.jsonl"},
    "openhands_gemma": {"results_run_1.jsonl"},
    "openhands_qwen": {"results_run_1.jsonl"},
}

OPENHANDS_AGENTS = {"openhands_gemma", "openhands_qwen"}

_MODE_PATTERN = re.compile(r"FM[-\s_]?(\d\.\d)", re.IGNORECASE)

FAILURE_MODE_TAXONOMY: Dict[str, str] = {
    "FM-1.1": "follow task requirements",
    "FM-1.2": "follow agent roles",
    "FM-1.3": "step repetition",
    "FM-1.4": "context loss",
    "FM-1.5": "failure to recognize completion",
    "FM-2.1": "unexpected conversation reset",
    "FM-2.2": "wrong assumption",
    "FM-2.3": "task derailment",
    "FM-2.4": "information withholding",
    "FM-2.5": "ignoring inputs",
    "FM-2.6": "reasoning-action mismatch",
    "FM-3.1": "premature termination",
    "FM-3.2": "missing or incomplete verification",
    "FM-3.3": "incorrect verification",
}

FAILURE_LABEL_ALIASES: Dict[str, str] = {
    "failure to follow task requirements": "follow task requirements",
}


def _canonical_mode_code(value: str) -> Optional[str]:
    if not value:
        return None
    match = _MODE_PATTERN.search(value)
    if not match:
        return None
    return f"FM-{match.group(1)}".upper()


def _normalize_failure_label(entry: Dict[str, Any]) -> str:
    mode = entry.get("mode")
    if isinstance(mode, str):
        mode_code = _canonical_mode_code(mode)
        if mode_code and mode_code in FAILURE_MODE_TAXONOMY:
            return FAILURE_MODE_TAXONOMY[mode_code]

    raw_label = entry.get("label")
    if isinstance(raw_label, str):
        cleaned = " ".join(raw_label.strip().lower().replace("_", " ").split())
        if cleaned:
            return FAILURE_LABEL_ALIASES.get(cleaned, cleaned)

    if isinstance(mode, str):
        fallback = " ".join(mode.strip().lower().replace("_", " ").split())
        if fallback:
            return FAILURE_LABEL_ALIASES.get(fallback, fallback)

    return "unlabeled"


def _normalize_agent(name: str) -> str:
    return AGENT_ALIASES.get(name, name)


def _collect_failure_modes(
    analyses_root: Path,
    issues: Sequence["IssueMetrics"],
) -> Tuple[Dict[str, Counter], Dict[str, Set[str]]]:
    counts: Dict[str, Counter] = defaultdict(Counter)
    instance_ids: Dict[str, Set[str]] = defaultdict(set)
    analysis_timeout: Dict[str, Set[str]] = defaultdict(set)

    if analyses_root.exists():
        for analysis_path in analyses_root.rglob("*.analysis.txt"):
            try:
                raw_agent = analysis_path.relative_to(analyses_root).parts[0]
            except (ValueError, IndexError):
                continue
            agent = _normalize_agent(raw_agent)
            stem = analysis_path.stem
            instance_id = stem[:-len(".analysis")] if stem.endswith(".analysis") else stem
            instance_ids[agent].add(instance_id)

            data = _load_analysis(analysis_path)
            if not data:
                counts[agent]["unlabeled"] += 1
                continue

            failures = data.get("failure_analysis") or []
            if not failures:
                counts[agent]["unlabeled"] += 1
                continue
            for entry in failures:
                label = _normalize_failure_label(entry)
                counts[agent][label] += 1
                if "timeout" in label:
                    analysis_timeout[agent].add(instance_id)

    processed: Dict[str, Set[str]] = defaultdict(set)
    expected: Dict[str, Set[str]] = defaultdict(set)

    for issue in issues or []:
        agent = _normalize_agent(issue.scaffold_dir)
        allowed_files = METADATA_FILES.get(agent)
        source_name = Path(issue.source).name
        if allowed_files and source_name not in allowed_files:
            continue
        expected[agent].add(issue.instance_id)
        in_analyses = issue.instance_id in instance_ids[agent]
        forced_timeout = False
        if instance_ids[agent] and not in_analyses:
            if agent not in OPENHANDS_AGENTS:
                continue
            forced_timeout = True
        if issue.instance_id in processed[agent]:
            continue
        processed[agent].add(issue.instance_id)
        instance_ids[agent].add(issue.instance_id)
        if forced_timeout or (
            issue.stop_reason and "timeout" in str(issue.stop_reason).lower()
        ):
            if issue.instance_id not in analysis_timeout[agent]:
                counts[agent]["timeout"] += 1

    for agent, ids in expected.items():
        missing = ids - instance_ids[agent]
        if missing:
            instance_ids[agent].update(missing)
            counts[agent]["timeout"] += len(missing)

    return counts, instance_ids


def _format_failure_label(label: str) -> str:
    cleaned = label.replace("_", " ").replace("-", " ").strip()
    if not cleaned:
        return "Unlabeled"
    return cleaned.title()


def plot_failure_modes_from_analyses(
    analyses_root: Path,
    plots_dir: Path,
    issues: Sequence["IssueMetrics"],
    *,
    top_labels: Optional[int] = None,
) -> None:
    counts, instance_ids = _collect_failure_modes(analyses_root, issues)
    if not counts:
        return

    plt = ensure_matplotlib()
    agents = sorted(counts.keys())

    all_labels: List[str] = []
    for counter in counts.values():
        all_labels.extend(counter.keys())
    label_counter = Counter(all_labels)

    if top_labels is not None and top_labels > 0 and len(label_counter) > top_labels:
        most_common = {label for label, _ in label_counter.most_common(top_labels)}
        for agent in agents:
            counter = counts[agent]
            other_total = sum(count for label, count in counter.items() if label not in most_common)
            counts[agent] = Counter({label: counter[label] for label in most_common if label in counter})
            if other_total:
                counts[agent]["other"] += other_total
        unique_labels = sorted({label for agent in agents for label in counts[agent].keys()})
    else:
        unique_labels = sorted(label_counter.keys())

    display_labels = [_format_failure_label(label) for label in unique_labels]
    colors = plt.cm.get_cmap("tab20", len(unique_labels))

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    bottom = [0] * len(agents)
    positions = list(range(len(agents)))
    for idx, label in enumerate(unique_labels):
        heights = [counts[agent].get(label, 0) for agent in agents]
        ax.bar(positions, heights, bottom=bottom, label=display_labels[idx], color=colors(idx))
        bottom = [b + h for b, h in zip(bottom, heights)]

    for idx, agent in enumerate(agents):
        total = len(instance_ids.get(agent, set()))
        ax.text(idx, bottom[idx] + max(0.3, 0.02 * total), f"{total}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Issue Count")
    ax.set_ylim(0, max(bottom) * 1.15 if bottom else 1)
    ax.set_xticks(positions)
    ax.set_xticklabels([format_scaffold_label(agent) for agent in agents], rotation=45, ha="right")
    ax.legend(title="Failure Mode", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)

    fig.tight_layout()
    fig.savefig(plots_dir / "failure_reasons_stacked.png", dpi=200)
    plt.close(fig)
