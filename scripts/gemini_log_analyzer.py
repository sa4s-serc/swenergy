#!/usr/bin/env python3
"""
Batch analysis of scaffold logs with Gemini 2.5 Flash.

This script discovers per-issue logs for each scaffold, sends them to the
Gemini API with a compact review prompt, persists the model's findings,
and then asks Gemini for a scaffold-level synthesis.

Usage example:
    python3 scripts/gemini_log_analyzer.py \
        --root /home/sustaind/result_analysis \
        --output-dir gemini_analyses \
        --model gemini-2.5-flash \
        --resume

Key features:
- Handles AutoCodeRover (Gemma + Qwen), SWE-Agent (Gemma + Qwen),
  mini SWE-Agent Gemma run_1 artifacts, and OpenHands run logs (Gemma + Qwen).
- Uses a small built-in prompt so Gemini interprets the logs instead of
  role-playing as the agent.
- Stores one analysis file per issue, then composes a scaffold-wide digest
  and asks Gemini for a short scaffold summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency on dry-run
    genai = None  # type: ignore[assignment]


DEFAULT_MODEL = "gemini-2.5-flash"
LOG_CHAR_BUDGET = 250_000  # guardrail for large transcripts
GENERATION_RETRIES = 3
GENERATION_COOLDOWN_SECONDS = 5 
API_KEY = "AIzaSyD935x3_S8Md2D6rAvo6xqLL5WqahFqHzQ"

INSTANCE_PROMPT_HEADER = (
    "You are an analyst reviewing a single agent run log for a multi-agent system.\n"
    "Do not role-play as the agent.\n"
    "Use the Multi-Agent System Failure Taxonomy (MAST) when interpreting the run:\n"
    "- FC1 Specification Issues (design or prompt gaps): FM-1.1 follow task requirements, FM-1.2 follow agent roles, FM-1.3 step repetition, FM-1.4 context loss, FM-1.5 failure to recognize completion.\n"
    "- FC2 Inter-Agent Misalignment (coordination breakdowns): FM-2.1 unexpected conversation reset, FM-2.2 wrong assumption, FM-2.3 task derailment, FM-2.4 information withholding, FM-2.5 ignoring inputs, FM-2.6 reasoning-action mismatch.\n"
    "- FC3 Task Verification (checking and termination flaws): FM-3.1 premature termination, FM-3.2 missing or incomplete verification, FM-3.3 incorrect verification.\n"
    "FC denotes the high-level failure category and FM the specific failure mode within that category.\n"
    "Describe the trajectory of the attempt, highlight pivotal decisions or turning points, and connect them to the final outcome.\n"
    "Explicitly diagnose the failure reason (or explain success) using the taxonomy.\n"
    "Return your analysis as a single JSON object with this structure:\n"
    "{\n"
    '  "trajectory": [\n'
    '    {"step": "<short description>", "evidence": "<log excerpt or observation>"}\n'
    "  ],\n"
    '  "outcome": {"status": "<SUCCESS|FAILURE|TIMEOUT|ERROR>", "evidence": "<log evidence>"},\n'
    '  "failure_analysis": [\n'
    '    {\n'
    '      "category": "FC#",\n'
    '      "mode": "FM-#.#",\n'
    '      "label": "<failure mode name>",\n'
    '      "justification": "<why this mode applies, citing evidence>"\n'
    "    }\n"
    "  ],\n"
    '  "notes": "<optional additional context or prevented failure modes>"\n'
    "}\n"
    "Only output valid JSON; do not include markdown fences or prose."
)

SUMMARY_PROMPT_HEADER = (
    "You are summarizing multiple issue analyses for one scaffold.\n"
    "Pull together the key behaviors and outcomes you observed across those runs, highlighting any common patterns or notable outliers."
)


@dataclass
class AnalysisTask:
    scaffold: str
    run_id: str
    instance_id: str
    metadata: Dict
    log_excerpt: str
    output_path: Path


def load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] Skipping malformed JSON in {path}: {exc}", file=sys.stderr)


def clip_text(text: str, limit: int = LOG_CHAR_BUDGET) -> str:
    """Return the text unchanged so logs are never truncated."""
    return text


def render_autocoderover_logs(log_dir: Path) -> str:
    logs_dir = log_dir / "logs"
    if not logs_dir.exists():
        return f"[log directory missing: {logs_dir}]"
    stdout_path = logs_dir / "scaffold.stdout"
    if not stdout_path.exists():
        return f"[scaffold.stdout missing in {logs_dir}]"
    return clip_text(stdout_path.read_text(errors="replace"))


def render_swe_agent_trajectory(path: Path) -> str:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return f"[failed to parse {path}: {exc}]"

    trajectory = data.get("trajectory") or []
    lines: List[str] = []
    info = data.get("info")
    if isinstance(info, dict):
        exit_status = info.get("exit_status")
        if exit_status:
            lines.append(f"[exit_status] {exit_status}")
        submission = info.get("submission")
        if submission:
            lines.append(f"[submission]\n{submission}")

    if trajectory:
        for idx, step in enumerate(trajectory, 1):
            lines.append(f"--- Step {idx} ---")
            for key in ("thought", "action", "response", "observation"):
                value = step.get(key)
                if value:
                    lines.append(f"{key.capitalize()}: {value}")
    else:
        messages = data.get("messages") or []
        if messages:
            for idx, message in enumerate(messages, 1):
                role = message.get("role", "unknown")
                content = message.get("content")
                lines.append(f"--- Message {idx} [{role}] ---")
                if isinstance(content, str):
                    lines.append(content)
                elif isinstance(content, list):
                    for chunk in content:
                        if isinstance(chunk, str):
                            lines.append(chunk)
                        elif isinstance(chunk, dict) and "text" in chunk:
                            lines.append(str(chunk["text"]))
        else:
            lines.append("[no trajectory or messages recorded]")

    if data.get("final_answer"):
        lines.append(f"Final answer: {data['final_answer']}")
    if data.get("error"):
        lines.append(f"Error: {data['error']}")
    if data.get("metadata"):
        lines.append(f"Metadata: {json.dumps(data['metadata'], indent=2)}")
    return clip_text("\n".join(lines))


def render_openhands_history(entry: Dict) -> str:
    history = entry.get("history") or []
    lines: List[str] = []
    for idx, event in enumerate(history, 1):
        source = event.get("source", "unknown")
        lines.append(f"--- Event {idx} [{source}] ---")
        message = event.get("message")
        if message:
            lines.append(f"message: {message}")
        action = event.get("action")
        if action:
            lines.append(f"action: {action}")
        args = event.get("args")
        if args:
            if isinstance(args, (dict, list)):
                args_str = json.dumps(args, indent=2)
            else:
                args_str = str(args)
            lines.append(f"args: {args_str}")
    metrics = entry.get("metrics")
    if metrics:
        lines.append(f"metrics: {json.dumps(metrics, indent=2)}")
    error = entry.get("error")
    if error:
        lines.append(f"error: {error}")
    return clip_text("\n".join(lines)) if lines else "[no history events]"


def gather_autocoderover_tasks(root: Path, output_dir: Path) -> List[AnalysisTask]:
    tasks: List[AnalysisTask] = []
    metadata_files = sorted((root / "acr_gemma").glob("acr_gemma_*.jsonl"))
    log_root = root / "acr_gemma" / "artifacts" / "acr_runs"
    if not metadata_files or not log_root.exists():
        return tasks

    for meta_file in metadata_files:
        run_id = meta_file.stem
        if run_id != "acr_gemma_1":
            continue
        for record in load_jsonl(meta_file):
            instance_id = record["instance_id"]
            pattern = f"autocoderover_gemma_{instance_id}_rep*"
            log_dirs: List[Path] = sorted(log_root.glob(pattern))
            if not log_dirs:
                log_dirs = sorted(
                    d
                    for d in log_root.iterdir()
                    if d.is_dir()
                    and instance_id in d.name
                    and "autocoderover_gemma_" in d.name
                    and "gemma2" not in d.name
                )
            if not log_dirs:
                log_excerpt = f"[no matching log directories for pattern {pattern}]"
            else:
                excerpts = [render_autocoderover_logs(log_dir) for log_dir in log_dirs]
                log_excerpt = "\n\n".join(
                    f"# Log bundle {idx + 1} ({log_dirs[idx].name})\n{excerpt}"
                    for idx, excerpt in enumerate(excerpts)
                )

            output_path = (
                output_dir / "autocoderover_gemma" / run_id / f"{instance_id}.analysis.txt"
            )
            tasks.append(
                AnalysisTask(
                    scaffold="autocoderover_gemma",
                    run_id=run_id,
                    instance_id=instance_id,
                    metadata=record,
                    log_excerpt=log_excerpt,
                    output_path=output_path,
                )
            )
    return tasks


def gather_autocoderover_qwen_tasks(root: Path, output_dir: Path) -> List[AnalysisTask]:
    tasks: List[AnalysisTask] = []
    meta_root = root / "acr_qwen"
    log_root = root / "acr_gemma" / "artifacts" / "qwen_runs"
    if not meta_root.exists() or not log_root.exists():
        return tasks

    metadata_files = sorted(meta_root.glob("acr_qwen_*.jsonl"))
    for meta_path in metadata_files:
        run_id = meta_path.stem
        if run_id != "acr_qwen_1":
            continue
        for record in load_jsonl(meta_path):
            instance_id = record["instance_id"]
            pattern = f"autocoderover_qwen_{instance_id}_rep*"
            log_dirs: List[Path] = sorted(log_root.glob(pattern))
            if not log_dirs:
                log_dirs = sorted(
                    d
                    for d in log_root.iterdir()
                    if d.is_dir() and instance_id in d.name and "autocoderover_qwen_" in d.name
                )
            if not log_dirs:
                log_excerpt = f"[no matching log directories for pattern {pattern}]"
            else:
                excerpts = [render_autocoderover_logs(log_dir) for log_dir in log_dirs]
                log_excerpt = "\n\n".join(
                    f"# Log bundle {idx + 1} ({log_dirs[idx].name})\n{excerpt}"
                    for idx, excerpt in enumerate(excerpts)
                )

            output_path = (
                output_dir / "autocoderover_qwen" / run_id / f"{instance_id}.analysis.txt"
            )
            tasks.append(
                AnalysisTask(
                    scaffold="autocoderover_qwen",
                    run_id=run_id,
                    instance_id=instance_id,
                    metadata=record,
                    log_excerpt=log_excerpt,
                    output_path=output_path,
                )
            )
    return tasks


def gather_big_swe_agent_tasks(root: Path, output_dir: Path) -> List[AnalysisTask]:
    candidate_bases = [
        root / "swe_agent_gemma",
        root / "big_swe_agent_gemma",
    ]
    base = next((path for path in candidate_bases if path.exists()), None)
    if base is None:
        return []

    metadata_files = sorted(base.glob("swe-agent_run_*.jsonl"))
    tasks: List[AnalysisTask] = []
    for meta_file in metadata_files:
        run_id = meta_file.stem
        if run_id != "swe-agent_run_1":
            continue
        run_folder_name = run_id.replace("swe-agent_", "")
        run_folder = base / run_folder_name
        if not run_folder.exists():
            continue
        for record in load_jsonl(meta_file):
            instance_id = record["instance_id"]
            trajectory_path = run_folder / f"{instance_id}_trajectory.json"
            if trajectory_path.exists():
                log_excerpt = render_swe_agent_trajectory(trajectory_path)
            else:
                log_excerpt = f"[trajectory log missing: {trajectory_path}]"
            output_path = (
                output_dir / "swe_agent_gemma" / run_id / f"{instance_id}.analysis.txt"
            )
            tasks.append(
                AnalysisTask(
                    scaffold="swe_agent_gemma",
                    run_id=run_id,
                    instance_id=instance_id,
                    metadata=record,
                    log_excerpt=log_excerpt,
                    output_path=output_path,
                )
            )
    return tasks


def gather_big_swe_agent_qwen_tasks(root: Path, output_dir: Path) -> List[AnalysisTask]:
    candidate_bases = [
        root / "swe_agent_qwen" / "qwen",
        root / "big_swe_agent_qwen" / "qwen",
    ]
    base = next((path for path in candidate_bases if path.exists()), None)
    if base is None:
        return []

    metadata_files = sorted(base.glob("swe-agent_run_*.jsonl"))
    tasks: List[AnalysisTask] = []
    for metadata_file in metadata_files:
        run_id = metadata_file.stem
        if run_id != "swe-agent_run_1":
            continue
        run_folder_name = run_id.replace("swe-agent_", "")
        run_folder = base / run_folder_name
        if not run_folder.exists():
            continue
        for record in load_jsonl(metadata_file):
            instance_id = record["instance_id"]
            trajectory_path = run_folder / f"{instance_id}_trajectory.json"
            if trajectory_path.exists():
                log_excerpt = render_swe_agent_trajectory(trajectory_path)
            else:
                log_excerpt = f"[trajectory log missing: {trajectory_path}]"
            output_path = (
                output_dir / "swe_agent_qwen" / run_id / f"{instance_id}.analysis.txt"
            )
            tasks.append(
                AnalysisTask(
                    scaffold="swe_agent_qwen",
                    run_id=run_id,
                    instance_id=instance_id,
                    metadata=record,
                    log_excerpt=log_excerpt,
                    output_path=output_path,
                )
            )
    return tasks


def gather_mini_swe_agent_tasks(root: Path, output_dir: Path) -> List[AnalysisTask]:
    configs = [
        ("mini_swe_agent_gemma", "mini_swe_agent_gemma"),
        ("mini_swe_agent_qwen", "mini_swe_agent_qwen"),
    ]
    tasks: List[AnalysisTask] = []
    for subdir, scaffold_name in configs:
        base = root / subdir
        if not base.exists():
            continue
        metadata_files = sorted(base.glob("results_run_*.jsonl"))
        logs_root = base / "logs"
        for meta_file in metadata_files:
            run_id = meta_file.stem
            run_suffix = run_id.split("_")[-1]
            if run_suffix != "1":
                continue
            trajectory_dir = logs_root / f"run_{run_suffix}"
            for record in load_jsonl(meta_file):
                instance_id = record["instance_id"]
                trajectory_path = trajectory_dir / f"{instance_id}_trajectory.json"
                if trajectory_path.exists():
                    log_excerpt = render_swe_agent_trajectory(trajectory_path)
                else:
                    log_excerpt = f"[trajectory log missing: {trajectory_path}]"
                output_path = (
                    output_dir / scaffold_name / run_id / f"{instance_id}.analysis.txt"
                )
                tasks.append(
                    AnalysisTask(
                        scaffold=scaffold_name,
                        run_id=run_id,
                        instance_id=instance_id,
                        metadata=record,
                        log_excerpt=log_excerpt,
                        output_path=output_path,
                    )
                )
    return tasks


def gather_openhands_tasks(root: Path, output_dir: Path) -> List[AnalysisTask]:
    configs = [
        (root / "openhands_gemma" / "run", "openhands_gemma", lambda path: True),
        (
            root / "openhands_qwen" / "qwen",
            "openhands_qwen",
            lambda path: path.stem.startswith("openhands_logs_") and "run_1" in path.stem,
        ),
    ]

    tasks: List[AnalysisTask] = []
    for run_root, scaffold_name, file_filter in configs:
        if not run_root.exists():
            continue
        for log_file in sorted(run_root.glob("*.jsonl")):
            if not file_filter(log_file):
                continue
            run_id = log_file.stem.replace("_logs", "")
            for record in load_jsonl(log_file):
                instance_id = record["instance_id"]
                metadata = {
                    "status": (record.get("instance") or {}).get("status"),
                    "metrics": record.get("metrics"),
                    "error": record.get("error"),
                    "history_events": len(record.get("history") or []),
                }
                log_excerpt = render_openhands_history(record)
                output_path = (
                    output_dir / scaffold_name / run_id / f"{instance_id}.analysis.txt"
                )
                tasks.append(
                    AnalysisTask(
                        scaffold=scaffold_name,
                        run_id=run_id,
                        instance_id=instance_id,
                        metadata=metadata,
                        log_excerpt=log_excerpt,
                        output_path=output_path,
                    )
                )
    return tasks


def build_instance_prompt(task: AnalysisTask) -> str:
    return (
        f"{INSTANCE_PROMPT_HEADER}\n\n"
        "Log excerpt:\n```text\n"
        f"{task.log_excerpt}\n"
        "```\n\n"
        "Respond with the JSON object described above. Populate at least one trajectory step, always include an outcome, and list all applicable failure modes (or an empty array if the run is clearly successful)."
    )


def build_summary_prompt(scaffold: str, combined_text: str) -> str:
    return (
        f"{SUMMARY_PROMPT_HEADER}\n\n"
        f"Scaffold: {scaffold}\n"
        "Analyst findings:\n```markdown\n"
        f"{combined_text}\n"
        "```\n\n"
        "Reference specific issue IDs when calling out examples."
    )


def generate_text(
    model: Any,
    prompt: str,
    *,
    max_output_tokens: Optional[int] = None,
) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(1, GENERATION_RETRIES + 1):
        try:
            generation_config = {
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 40,
            }
            if max_output_tokens is not None:
                generation_config["max_output_tokens"] = max_output_tokens

            response = model.generate_content(
                [prompt],
                generation_config=generation_config,
                safety_settings={
                    "HARASSMENT": "BLOCK_NONE",
                    "HATE": "BLOCK_NONE",
                    "SEXUAL": "BLOCK_NONE",
                    "DANGEROUS": "BLOCK_NONE",
                },
            )
            if response and response.text:
                return response.text
            raise RuntimeError("Empty response from Gemini")
        except Exception as exc:  # pragma: no cover - defensive path
            last_error = exc
            if attempt < GENERATION_RETRIES:
                time.sleep(GENERATION_COOLDOWN_SECONDS)
            else:
                raise
    raise RuntimeError("Unreachable") from last_error


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse scaffold logs with Gemini.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Workspace root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gemini_analyses"),
        help="Directory to store per-issue analyses and summaries.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name.")
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit for the number of per-issue analyses (debugging aid).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip analyses whose output files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover tasks and report what would be analysed without calling Gemini.",
    )
    args = parser.parse_args(argv)

    api_key = API_KEY
    if not api_key and not args.dry_run:
        raise SystemExit("API_KEY is empty. Set it near the top of gemini_log_analyzer.py.")

    all_tasks: List[AnalysisTask] = []
    all_tasks.extend(gather_autocoderover_tasks(args.root, args.output_dir))
    all_tasks.extend(gather_autocoderover_qwen_tasks(args.root, args.output_dir))
    all_tasks.extend(gather_big_swe_agent_tasks(args.root, args.output_dir))
    all_tasks.extend(gather_big_swe_agent_qwen_tasks(args.root, args.output_dir))
    all_tasks.extend(gather_mini_swe_agent_tasks(args.root, args.output_dir))
    all_tasks.extend(gather_openhands_tasks(args.root, args.output_dir))

    if not all_tasks:
        print("No analysis tasks found. Check directory paths.", file=sys.stderr)
        return 1

    if args.limit is not None:
        all_tasks = all_tasks[: args.limit]

    if args.dry_run:
        print(f"[dry-run] Planned analyses: {len(all_tasks)}")
        for task in all_tasks:
            print(f"- {task.scaffold} | {task.run_id} | {task.instance_id} -> {task.output_path}")
        return 0

    if genai is None:
        raise SystemExit(
            "google-generativeai package is required. Install with `pip install google-generativeai`."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    for task in all_tasks:
        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.resume and task.output_path.exists():
            continue
        prompt = build_instance_prompt(task)
        response_text = generate_text(model, prompt)
        task.output_path.write_text(response_text)

    # Scaffold-level aggregation.
    scaffold_groups: Dict[str, List[AnalysisTask]] = defaultdict(list)
    for task in all_tasks:
        scaffold_groups[task.scaffold].append(task)

    for scaffold, tasks in scaffold_groups.items():
        combined_parts: List[str] = []
        for task in tasks:
            if task.output_path.exists():
                text = task.output_path.read_text()
                combined_parts.append(
                    f"## {task.run_id} :: {task.instance_id}\n{text.strip()}"
                )
        if not combined_parts:
            continue
        combined_text = "\n\n".join(combined_parts)
        combined_path = args.output_dir / scaffold / f"{scaffold}_combined.txt"
        combined_path.write_text(combined_text)

        summary_prompt = build_summary_prompt(scaffold, combined_text)
        summary_path = args.output_dir / scaffold / f"{scaffold}_summary.txt"
        if args.resume and summary_path.exists():
            continue
        summary_text = generate_text(model, summary_prompt)
        summary_path.write_text(summary_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
