Result Analysis
===============

This repository contains helpers for aggregating and plotting metrics collected from SWE-bench Verified runs. The most common workflow is to regenerate the summary tables and figures after adding new JSONL logs.

## Prerequisites

- Python 3.10 or newer available as `python3`
- `matplotlib` installed in the active environment (`pip install matplotlib`)
- Run log files stored as JSONL under one or more experiment directories (for example `acr_gemma/`, `mini_swe_agent_gemma/`, …)

## Collecting New Runs

When you want to generate fresh JSONL logs before running the analysis scripts, you can launch each scaffold locally with the standalone helpers in the repository root:

- `openhands_main.py` – runs OpenHands on the SWE-bench verified mini split while tracking energy and memory usage.
- `swe_agent_main.py` – executes SWE-Agent with identical instrumentation, using the pre-generated slice indices in `mini_verified_line_numbers.txt`.
- `mini_swe_agent_main.py` – drives the lighter-weight SWE-Agent configuration against the mini dataset.
- `autocoderover_main.py` – runs AutoCodeRover locally using the parameters from `agentic-efficiency/configs/matrix.yaml`, writing per-issue metrics to `autocoderover.jsonl` and archiving logs under `autocoderover_runs/`.

Each script is intended to be run from the repository root via `python3 <script>` and appends results to a scaffold-specific JSONL file (for example `openhands.jsonl`, `swe-agent.jsonl`, `mini-swe-agent.jsonl`, or `autocoderover.jsonl`). They automatically skip issues that already appear in their output file so you can rerun them safely.

> **Tip:** The scripts assume the relevant scaffold installations (OpenHands, SWE-Agent, AutoCodeRover) are already configured on the machine and that any paths or API keys referenced inside the scripts exist. Adjust the constants near the top of each script if your environment differs.

## Running the analysis

Use `scripts/analyze_results.py` to rebuild the CSV outputs and plots. Pass every directory (or individual JSONL file) you want to include:

```bash
python3 scripts/analyze_results.py \
  --inputs acr_gemma big_swe_agent_gemma big_swe_agent_qwen \
           mini_swe_agent_gemma openhands_gemma openhands_qwen \
  --energy-scale 1e-6 \
  --idle-cpu-w 51.69 --idle-gpu-w 2.70 \
  --plots-dir plots/all_scaffolds \
  --csv run_summary_all.csv \
  --per-issue per_issue.csv \
  --table-csv tables_all.csv
```

- `--energy-scale 1e-6` converts microjoule readings stored in the logs to joules. Adjust the value (`1.0` is the default) if your data already uses joules.
- `--idle-cpu-w` and `--idle-gpu-w` (optional) subtract a fixed idle baseline (Watts) per trial duration to compute idle-adjusted energies. When provided, the per-issue CSV includes `*_adj_j` columns and summaries include `Energy Usage Adj (J)`.
- `--plots-dir` controls where figures (including `scatter_tasks_vs_energy.png`) are saved.
- `--csv`, `--per-issue`, and `--table-csv` export run-level, issue-level, and scaffold aggregate tables respectively.

The script prints a per-run summary and a scaffold summary to the terminal while writing the selected artifacts to disk.

## Troubleshooting

- **Missing matplotlib:** Install it with `pip install matplotlib`.
- **No JSONL files found:** Confirm the directories passed to `--inputs` exist and contain `*.jsonl` files.
- **Label overlap in scatter plots:** The helper automatically spaces labels, but extreme clustering can still occur. Slightly increasing figure size (for example `--plots-dir plots && edit scripts/analyze_results.py` to adjust `figsize`) can help.
