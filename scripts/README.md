# Result Analysis Scripts

This folder contains small utilities for working with the JSONL logs produced by the experiment runs documented in `plan.md`.

- `analyze_results.py` – Loads one or more JSONL files, prints per-run summaries, optional scaffold aggregates, and can export CSV/issue-level data. Use `--plots-dir DIR` to generate bar/scatter charts (requires `matplotlib`).
- `contribution_analysis.py` – Computes Pearson correlations between agent activity (LLM calls, tool calls, tokens) and energy/latency outcomes. Add `--per-scaffold` to split by scaffold/model, and `--plots-dir DIR` for correlation plots.
- `efficiency_frontier.py` – Applies optional resource constraints and reports energy-success trade-offs plus a Pareto frontier; `--plots-dir DIR` produces scatter charts highlighting the frontier.

All scripts accept `--energy-scale` to convert stored energy units (e.g., pass `1e-6` when the logs are in microjoules) and expect Python 3 with `matplotlib` installed when plotting is requested.
