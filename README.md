

This repository contains helpers for aggregating and plotting metrics collected from SWE-bench Verified runs, and provides a full replication package including the source code, logs, and measurements for each phase. The most common workflow is to regenerate the summary tables and figures after adding new JSONL logs.

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

