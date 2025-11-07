#!/usr/bin/env python3
"""
Utility for patching malformed JSONL rows produced by SWE-agent style runs.

It scans each line, validates that it is valid JSON, and applies a small
set of string fixes when parsing fails (currently replacing `/dev/""` with
`/dev/null` and escaping stray double quotes inside `model_patch` diffs).
The script can be pointed at one or more JSONL files and optionally run in
dry-run mode to preview the changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple


# (bad, good) replacements to try when JSON decoding fails.
REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ('--- /dev/""', "--- /dev/null"),
    ('+++ /dev/""', "+++ /dev/null"),
)


def escape_unescaped_quotes(raw: str) -> Tuple[str, bool]:
    """Escape any double quote characters that are not already escaped."""
    out: list[str] = []
    modified = False
    for idx, ch in enumerate(raw):
        if ch != '"':
            out.append(ch)
            continue

        # Count preceding backslashes to determine escape parity.
        backslashes = 0
        j = idx - 1
        while j >= 0 and raw[j] == "\\":
            backslashes += 1
            j -= 1

        if backslashes % 2 == 0:
            out.append("\\")
            out.append('"')
            modified = True
        else:
            out.append('"')
    return "".join(out), modified


def fix_model_patch_quotes(line: str) -> Tuple[str, bool]:
    """Ensure all quotes inside model_patch string are properly escaped."""
    marker = '"model_patch": "'
    start = line.find(marker)
    if start == -1:
        return line, False

    prefix_end = start + len(marker)

    suffix_start = line.rfind('"')
    if suffix_start <= prefix_end:
        return line, False

    patch = line[prefix_end:suffix_start]
    fixed_patch, changed = escape_unescaped_quotes(patch)
    if not changed:
        return line, False

    return line[:prefix_end] + fixed_patch + line[suffix_start:], True


def ensure_model_patch_string(line: str) -> Tuple[str, bool]:
    """Quote unquoted model_patch values."""
    marker = '"model_patch":'
    start = line.find(marker)
    if start == -1:
        return line, False

    value_start = start + len(marker)
    # Skip whitespace after colon.
    while value_start < len(line) and line[value_start].isspace():
        value_start += 1
    if value_start >= len(line):
        return line, False

    if line[value_start] == '"':
        return line, False

    # Find the end of the unquoted token (either comma or closing brace).
    end = value_start
    while end < len(line) and line[end] not in ',}':
        end += 1

    raw_value = line[value_start:end].strip()
    if not raw_value:
        return line, False

    quoted = json.dumps(raw_value)
    return line[:value_start] + quoted + line[end:], True


def fix_line(line: str) -> Tuple[str, bool]:
    """Return (possibly) corrected line and whether a modification occurred."""
    try:
        json.loads(line)
        return line, False
    except json.JSONDecodeError:
        candidate = line
        changed = False
        for bad, good in REPLACEMENTS:
            updated = candidate.replace(bad, good)
            if updated != candidate:
                changed = True
                candidate = updated
            else:
                candidate = updated

        candidate, quotes_changed = fix_model_patch_quotes(candidate)
        changed = changed or quotes_changed

        candidate, quoted = ensure_model_patch_string(candidate)
        changed = changed or quoted

        if candidate != line:
            try:
                json.loads(candidate)
                return candidate, True
            except json.JSONDecodeError:
                return line, False

        return line, changed


def process_file(path: Path, dry_run: bool) -> bool:
    """Process a JSONL file; return True if any changes were written."""
    original_lines = path.read_text().splitlines(keepends=True)
    updated_lines = []
    modified = False
    for line in original_lines:
        fixed, changed = fix_line(line)
        updated_lines.append(fixed)
        modified |= changed

    if modified and not dry_run:
        path.write_text("".join(updated_lines))

    return modified


def iter_targets(paths: Iterable[str]) -> Iterable[Path]:
    """Yield each path as a Path object, expanding directories to JSONL files."""
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            for file_path in sorted(path.rglob("*.jsonl")):
                if file_path.is_file():
                    yield file_path
        else:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix common JSONL diff issues.")
    parser.add_argument(
        "targets",
        nargs="+",
        help="JSONL files or directories to process (directories searched recursively).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would change without overwriting them.",
    )
    args = parser.parse_args()

    any_changes = False
    for target in iter_targets(args.targets):
        try:
            changed = process_file(target, args.dry_run)
        except (UnicodeDecodeError, json.JSONDecodeError):
            print(f"[error] failed to read {target}")
            continue

        if changed:
            status = "would change" if args.dry_run else "updated"
            print(f"[{status}] {target}")
            any_changes = True
        else:
            print(f"[clean] {target}")

    if args.dry_run and any_changes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
