import json
import os
import re
import shlex
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from string import Template

import fcntl
import psutil
import pyRAPL
import pynvml
import threading
import yaml


def send_sigint_twice(process, message=None):
    """
    Send SIGINT to the process group. If it keeps running, escalate.
    """
    if message:
        print(message)

    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGINT)
        time.sleep(1)
        if process.poll() is None:
            print(f"Process still running, sending SIGTERM to process group {pgid}")
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(2)
            if process.poll() is None:
                print(f"Process still running, sending SIGKILL to process group {pgid}")
                os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        print("Process already terminated")
    except Exception as exc:  # pragma: no cover - best effort fallback
        print(f"Error terminating process: {exc}")
        process.terminate()
        time.sleep(2)
        if process.poll() is None:
            process.kill()


def monitor_memory(gpu_handle, ram_peak, vram_peak, stop_event):
    """
    Track peak RAM/VRAM usage for the current process.
    """
    proc = psutil.Process()
    while not stop_event.is_set():
        try:
            ram_usage = proc.memory_info().rss
            if ram_usage > ram_peak[0]:
                ram_peak[0] = ram_usage
        except psutil.Error:
            pass

        try:
            vram_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            vram_usage = vram_info.used
            if vram_usage > vram_peak[0]:
                vram_peak[0] = vram_usage
        except pynvml.NVMLError:
            pass

        time.sleep(0.1)


def load_matrix_settings(matrix_path):
    """
    Pull dataset/task configuration and AutoCodeRover env from matrix.yaml.
    """
    with open(matrix_path, "r", encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle)

    dataset_split = Path(matrix["dataset"]["split"]).expanduser()
    timeout_minutes = matrix.get("timeouts", {}).get("per_issue_minutes", 30)

    acr_env = None
    for scaffold in matrix.get("scaffolds", []):
        if scaffold.get("name") == "autocoderover":
            acr_env = scaffold.get("env", {}).copy()
            break

    if acr_env is None:
        raise RuntimeError("AutoCodeRover scaffold configuration not found in matrix.yaml")

    return dataset_split, timeout_minutes, acr_env


def render_env(env_cfg, context):
    """
    Replace $VARIABLE placeholders in env config with context values.
    """
    rendered = {}
    for key, value in env_cfg.items():
        if isinstance(value, str):
            rendered[key] = Template(value).safe_substitute(context)
        else:
            rendered[key] = value
    return rendered


def build_acr_command(issue_id, run_root, env_cfg):
    """
    Construct the AutoCodeRover command, returning the shell string and output dir.
    """
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_dir = logs_dir / "acr_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    acr_root = Path(env_cfg.get("ACR_ROOT", "/home/sustaind/auto-code-rover")).expanduser()
    python_bin = env_cfg.get("ACR_PYTHON", "python")
    task_mode = env_cfg.get("ACR_TASK", "swe-bench")
    activate_cmd = env_cfg.get("ACR_ACTIVATE")

    args = ["--output-dir", str(output_dir)]

    def _add_arg(flag, key):
        val = env_cfg.get(key)
        if val:
            args.extend([flag, str(val)])

    _add_arg("--model", "ACR_MODEL")
    _add_arg("--model-temperature", "ACR_MODEL_TEMPERATURE")
    _add_arg("--setup-map", "ACR_SETUP_MAP")
    _add_arg("--tasks-map", "ACR_TASKS_MAP")

    task_list = env_cfg.get("ACR_TASK_LIST_FILE")
    task_name = env_cfg.get("ACR_TASK_NAME") or issue_id
    if task_list:
        args.extend(["--task-list-file", str(task_list)])
    else:
        args.extend(["--task", str(task_name)])

    extra_args = env_cfg.get("ACR_EXTRA_ARGS")
    if extra_args:
        args.extend(shlex.split(extra_args))

    command_parts = [f"cd {shlex.quote(str(acr_root))}"]
    if activate_cmd:
        command_parts.append(activate_cmd)

    autop_parts = ["PYTHONPATH=.", python_bin, "app/main.py", task_mode, *args]
    quoted_autop = " ".join(shlex.quote(str(part)) for part in autop_parts)
    command_parts.append(quoted_autop)

    command = " && ".join(part for part in command_parts if part)
    return command, output_dir, logs_dir


def collect_output_patch(issue_id, output_dir):
    """
    Extract the best available patch from AutoCodeRover's output directory.
    """
    if not output_dir.exists():
        return ""

    reference_patches = set()

    def _normalize_patch(text):
        return text.strip()

    for meta_path in output_dir.rglob("meta.json"):
        try:
            raw_meta = meta_path.read_text(encoding="utf-8")
            meta = json.loads(raw_meta)
        except Exception:
            continue

        task_info = meta.get("task_info") or {}
        meta_task_id = meta.get("task_id")
        meta_issue_ids = {
            meta_task_id,
            task_info.get("instance_id"),
            task_info.get("task_id"),
        }
        if issue_id and issue_id not in meta_issue_ids:
            continue

        meta_patch = (
            task_info.get("patch")
            or meta.get("patch")
            or (task_info.get("report") or {}).get("model_patch")
        )
        if isinstance(meta_patch, str) and meta_patch.strip():
            reference_patches.add(_normalize_patch(meta_patch))

    candidates = []

    for pattern in ("*.diff", "*.patch"):
        for path in output_dir.rglob(pattern):
            if path.name == "developer_patch.diff":
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            if not text.strip():
                continue
            norm = _normalize_patch(text)
            if reference_patches and norm in reference_patches:
                continue
            score = 0
            path_str = str(path)
            if issue_id and issue_id in path.name:
                score += 2
            if issue_id and issue_id in path_str:
                score += 1
            try:
                mtime = path.stat().st_mtime
            except Exception:
                mtime = 0.0
            candidates.append((score, mtime, text))

    if candidates:
        candidates.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return candidates[0][2]

    predictions_path = output_dir / "predictions_for_swebench.json"
    if predictions_path.exists():
        try:
            data = json.loads(predictions_path.read_text(encoding="utf-8"))
        except Exception:
            data = None
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                if issue_id and issue_id not in {
                    entry.get("instance_id"),
                    entry.get("issue_id"),
                    entry.get("task_id"),
                }:
                    continue
                patch = entry.get("model_patch") or entry.get("patch") or entry.get("git_patch")
                if isinstance(patch, str) and patch.strip():
                    norm = _normalize_patch(patch)
                    if not reference_patches or norm not in reference_patches:
                        return patch

    return ""


def parse_run_metrics(logs_dir, output_dir, issue_id):
    """
    Parse token usage, LLM/tool call counts, and collected diff.
    """
    stdout_path = logs_dir / "scaffold.stdout"
    stderr_path = logs_dir / "scaffold.stderr"

    texts = []
    for path in (stdout_path, stderr_path):
        if path.exists():
            try:
                texts.append(path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass

    total_in = 0
    total_out = 0
    llm_calls = 0
    total_cost = 0.0

    patterns_in = [
        re.compile(r"\binput_tokens\s*=\s*(\d+)", re.IGNORECASE),
        re.compile(r"\bprompt_tokens\s*=\s*(\d+)", re.IGNORECASE),
    ]
    patterns_out = [
        re.compile(r"\boutput_tokens\s*=\s*(\d+)", re.IGNORECASE),
        re.compile(r"\bcompletion_tokens\s*=\s*(\d+)", re.IGNORECASE),
    ]
    llm_pattern = re.compile(r"Model \([^)]*\) API request cost info", re.IGNORECASE)
    cost_pattern = re.compile(r"cost\s*=\s*([0-9.+-eE]+)")

    for text in texts:
        for pattern in patterns_in:
            for match in pattern.finditer(text):
                total_in += int(match.group(1))

        for pattern in patterns_out:
            for match in pattern.finditer(text):
                total_out += int(match.group(1))

        llm_matches = llm_pattern.finditer(text)
        for match in llm_matches:
            llm_calls += 1
            cost_match = cost_pattern.search(match.group(0))
            if cost_match:
                try:
                    total_cost += float(cost_match.group(1))
                except ValueError:
                    pass

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("{") or '"usage"' not in stripped:
                continue
            try:
                payload = json.loads(stripped)
            except Exception:
                continue
            usage = payload.get("usage", {})
            if "prompt_tokens" in usage:
                total_in += int(usage.get("prompt_tokens", 0) or 0)
            if "completion_tokens" in usage:
                total_out += int(usage.get("completion_tokens", 0) or 0)

    tool_calls = 0
    if output_dir.exists():
        tool_files = sorted(output_dir.rglob("tool_call_layers.json"))
        for path in tool_files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, list):
                tool_calls += sum(len(layer) for layer in data if isinstance(layer, list))

    diff = collect_output_patch(issue_id, output_dir)
    return {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "cost": total_cost,
        "model_patch": diff,
    }


def ensure_processed_set(path):
    """
    Load already processed instance IDs from the output jsonl file.
    """
    processed = set()
    if not path.exists():
        return processed

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            instance_id = entry.get("instance_id")
            if instance_id:
                processed.add(instance_id)
    return processed


def main():
    matrix_path = Path("agentic-efficiency/configs/matrix.yaml")
    dataset_split, timeout_minutes, acr_env = load_matrix_settings(matrix_path)

    if not dataset_split.exists():
        raise FileNotFoundError(f"Dataset split file not found: {dataset_split}")

    issues = [line.strip() for line in dataset_split.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not issues:
        print("No issue IDs found in dataset split file.")
        return

    output_path = Path("autocoderover.jsonl")
    processed_ids = ensure_processed_set(output_path)

    pyRAPL.setup()
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    timeout_seconds = timeout_minutes * 60
    runs_root = Path("autocoderover_runs")
    runs_root.mkdir(exist_ok=True)

    try:
        for idx, issue_id in enumerate(issues, 1):
            if issue_id in processed_ids:
                print(f"Skipping {issue_id} as already processed.")
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_root = runs_root / f"{issue_id}_{timestamp}"
            run_root.mkdir(parents=True, exist_ok=True)

            logs_dir = run_root / "logs"
            context = {
                "ISSUE_ID": issue_id,
                "HARNESS_LOGS_DIR": str(logs_dir),
                "RUN_DIR": str(run_root),
            }
            rendered_env = render_env(acr_env, context)
            command, output_dir, logs_dir = build_acr_command(issue_id, run_root, rendered_env)

            env_for_process = os.environ.copy()
            forward_keys = set()
            forward_cfg = rendered_env.get("ACR_FORWARD_ENV")
            if forward_cfg:
                forward_keys.update(k.strip() for k in str(forward_cfg).split(","))

            for key, value in rendered_env.items():
                if key.startswith("ACR_"):
                    continue
                if isinstance(value, str):
                    env_for_process[key] = value

            for key in forward_keys:
                if key and key in os.environ:
                    env_for_process[key] = os.environ[key]

            print(f"[{idx}/{len(issues)}] Processing {issue_id}")
            print(f"Running command: {command}")

            stdout_path = logs_dir / "scaffold.stdout"
            stderr_path = logs_dir / "scaffold.stderr"

            process = subprocess.Popen(
                ["bash", "-lc", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid,
                env=env_for_process,
            )

            fcntl.fcntl(process.stdout, fcntl.F_SETFL, fcntl.fcntl(process.stdout, fcntl.F_GETFL) | os.O_NONBLOCK)
            fcntl.fcntl(process.stderr, fcntl.F_SETFL, fcntl.fcntl(process.stderr, fcntl.F_GETFL) | os.O_NONBLOCK)

            measurement_started = False
            monitor_thread = None
            stop_event = threading.Event()
            meter = None
            initial_gpu_energy = None
            start_time = None

            ram_peak = [0]
            vram_peak = [0]

            process_start_time = time.time()
            deadline = process_start_time + timeout_seconds
            stop_reason = "successful"

            with stdout_path.open("w", encoding="utf-8") as stdout_log, stderr_path.open("w", encoding="utf-8") as stderr_log:
                while time.time() < deadline and process.poll() is None:
                    try:
                        chunk = process.stdout.read()
                    except Exception:
                        chunk = None
                    if chunk:
                        stdout_log.write(chunk)
                        stdout_log.flush()
                        print(chunk, end="")
                        if (
                            not measurement_started
                            and (
                                "=========== Starting AutoCodeRover" in chunk
                                or "Model (" in chunk
                            )
                        ):
                            measurement_started = True
                            initial_gpu_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
                            meter = pyRAPL.Measurement("cpu")
                            meter.begin()
                            start_time = time.time()
                            stop_event.clear()
                            monitor_thread = threading.Thread(
                                target=monitor_memory,
                                args=(gpu_handle, ram_peak, vram_peak, stop_event),
                                daemon=True,
                            )
                            monitor_thread.start()
                            deadline = time.time() + timeout_seconds

                        if "litellm.BadRequestError" in chunk:
                            send_sigint_twice(process, "Encountered BadRequestError, terminating run.")
                            stop_reason = "badrequest"
                            break

                    try:
                        chunk_err = process.stderr.read()
                    except Exception:
                        chunk_err = None
                    if chunk_err:
                        stderr_log.write(chunk_err)
                        stderr_log.flush()
                        print(chunk_err, end="")
                        if (
                            not measurement_started
                            and (
                                "=========== Starting AutoCodeRover" in chunk_err
                                or "Model (" in chunk_err
                            )
                        ):
                            measurement_started = True
                            initial_gpu_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
                            meter = pyRAPL.Measurement("cpu")
                            meter.begin()
                            start_time = time.time()
                            stop_event.clear()
                            monitor_thread = threading.Thread(
                                target=monitor_memory,
                                args=(gpu_handle, ram_peak, vram_peak, stop_event),
                                daemon=True,
                            )
                            monitor_thread.start()
                            deadline = time.time() + timeout_seconds

                        if "litellm.BadRequestError" in chunk_err:
                            send_sigint_twice(process, "Encountered BadRequestError, terminating run.")
                            stop_reason = "badrequest"
                            break

                    time.sleep(0.1)

                if process.poll() is None and time.time() >= deadline:
                    stop_reason = "timeout"
                    send_sigint_twice(process, f"Instance {issue_id} timed out after {timeout_minutes} minutes.")

                # Drain remaining output
                try:
                    remaining_out = process.stdout.read()
                    if remaining_out:
                        stdout_log.write(remaining_out)
                        print(remaining_out, end="")
                except Exception:
                    pass
                try:
                    remaining_err = process.stderr.read()
                    if remaining_err:
                        stderr_log.write(remaining_err)
                        print(remaining_err, end="")
                except Exception:
                    pass

            process.wait()
            print(f"Process for {issue_id} exited with code {process.returncode}")

            if measurement_started:
                stop_event.set()
                if monitor_thread:
                    monitor_thread.join()

            end_time = time.time()
            if measurement_started and meter:
                meter.end()

            cpu_energy = -1
            if meter:
                try:
                    cpu_result = meter.result
                    if hasattr(cpu_result, "pkg") and cpu_result.pkg:
                        cpu_energy = int(cpu_result.pkg[0])
                except Exception:
                    cpu_energy = -1

            gpu_energy_used = -1
            if initial_gpu_energy is not None:
                try:
                    final_gpu_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
                    gpu_energy_used = final_gpu_energy - initial_gpu_energy
                except pynvml.NVMLError:
                    pass

            time_taken = 0.0
            if start_time:
                time_taken = round(end_time - start_time, 2)
            else:
                time_taken = round(end_time - process_start_time, 2)

            if stop_reason == "successful":
                if process.returncode != 0:
                    stop_reason = f"EXIT_{process.returncode}"

            metrics = parse_run_metrics(logs_dir, output_dir, issue_id)

            result_entry = {
                "instance_id": issue_id,
                "scaffold": "autocoderover",
                "model_name": rendered_env.get("ACR_MODEL", "unknown"),
                "cpu_energy": cpu_energy,
                "gpu_energy": gpu_energy_used,
                "time_taken": time_taken,
                "llm_calls": metrics["llm_calls"],
                "tool_calls": metrics["tool_calls"],
                "vram_used": vram_peak[0],
                "ram_used": ram_peak[0],
                "input_tokens": metrics["input_tokens"],
                "output_tokens": metrics["output_tokens"],
                "cost": metrics["cost"],
                "stop_reason": stop_reason,
                "model_patch": metrics["model_patch"],
            }

            with output_path.open("a", encoding="utf-8") as out_file:
                out_file.write(json.dumps(result_entry) + "\n")

            processed_ids.add(issue_id)
            print(f"Saved metrics for {issue_id} to {output_path}")

    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
