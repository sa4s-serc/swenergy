import json
import subprocess
import os
import pyRAPL
import pynvml
import time
import threading
import psutil
import signal
import fcntl


def send_sigint_twice(process, message=None):
    """
    Send SIGINT to the process group, wait 1 second, and escalate to SIGTERM if still running.
    """
    if message:
        print(message)

    try:
        # Try to terminate the entire process group
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGINT)
        time.sleep(1)

        if process.poll() is None:
            # Still running, try SIGTERM
            print(
                f"Process still running, sending SIGTERM to process group {pgid}")
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(2)

            if process.poll() is None:
                # Last resort: SIGKILL
                print(
                    f"Process still running, sending SIGKILL to process group {pgid}")
                os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        # Process already terminated
        print("Process already terminated")
    except Exception as e:
        print(f"Error terminating process: {e}")
        # Fallback to regular terminate
        process.terminate()
        time.sleep(2)
        if process.poll() is None:
            process.kill()


def monitor_memory(gpu_handle, ram_peak, vram_peak, stop_event):
    """
    Monitor RAM and VRAM usage and track peaks.

    Args:
        gpu_handle: NVML GPU handle.
        ram_peak: List to store peak RAM usage.
        vram_peak: List to store peak VRAM usage.
        stop_event: Threading event to stop monitoring.
    """
    process = psutil.Process()
    while not stop_event.is_set():
        ram = process.memory_info().rss
        vram_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        vram = vram_info.used
        if ram > ram_peak[0]:
            ram_peak[0] = ram
        if vram > vram_peak[0]:
            vram_peak[0] = vram
        time.sleep(0.1)  # Sample every 0.1 seconds


def get_results_from_output_jsonl(output_jsonl_path, instance_id):
    """
    Extract token counts and diff from the output.jsonl file.

    Args:
        output_jsonl_path (str): Path to the output JSONL file.
        instance_id (str): The instance ID for logging purposes.

    Returns:
        tuple: (input_tokens, output_tokens, llm_calls, diff, cost)
    """
    try:
        # Expand the tilde in the path
        expanded_path = os.path.expanduser(output_jsonl_path)

        # Find the entry matching the instance_id
        total_input_tokens = 0
        total_output_tokens = 0
        llm_calls = 0
        diff = ""
        cost = 0.0

        with open(expanded_path, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("instance_id") == instance_id:
                        # Extract metrics from the output.jsonl entry
                        metrics = entry.get("metrics", {})
                        accumulated_token_usage = metrics.get(
                            "accumulated_token_usage", {})

                        # Get prompt_tokens as input tokens
                        total_input_tokens = accumulated_token_usage.get(
                            "prompt_tokens", 0)

                        # Get completion_tokens as output tokens
                        total_output_tokens = accumulated_token_usage.get(
                            "completion_tokens", 0)

                        # Get accumulated_cost as cost
                        cost = metrics.get("accumulated_cost", 0.0)

                        # Count the number of LLM calls from token_usages array
                        llm_calls = len(metrics.get("token_usages", []))

                        # Get the model patch (diff)
                        diff = entry.get("test_result", {}).get(
                            "git_patch", "")

                        break

        return total_input_tokens, total_output_tokens, llm_calls, diff, cost

    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error reading output file: {e}")
        return -1, -1, -1, "", 0.0


def copy_output_jsonl_to_logs(output_jsonl_path, instance_id):
    """
    Copy only the line for the current instance from output.jsonl to logs folder as openhands_logs.jsonl.

    Args:
        output_jsonl_path (str): Path to the output JSONL file.
        instance_id (str): The instance ID to find and copy.
    """
    try:
        # Expand the tilde in the path
        expanded_path = os.path.expanduser(output_jsonl_path)

        # Create logs directory if it doesn't exist
        logs_dir = "./logs"
        os.makedirs(logs_dir, exist_ok=True)

        # Find and append the line for the current instance
        dest_path = os.path.join(logs_dir, "openhands_logs.jsonl")
        with open(expanded_path, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("instance_id") == instance_id:
                        with open(dest_path, "a") as out_f:
                            out_f.write(line)
                        print(f"Copied instance {instance_id} to {dest_path}")
                        break

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error copying output file: {e}")


def create_config_toml(instance_id, config_path):
    """
    Create a config.toml file with the specified instance_id in selected_ids.

    Args:
        instance_id (str): The instance ID to include in the config.
        config_path (str): Path to the config.toml file.
    """
    config_content = f"selected_ids = ['{instance_id}']\n"
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"Created config.toml with selected_ids = ['{instance_id}']")


def main():
    # Load the mini dataset
    f = open("swe-bench-verified-mini.jsonl")
    data = [json.loads(line) for line in f]
    f.close()

    # Load existing processed instance_ids to skip duplicates
    processed_ids = set()
    if os.path.exists("openhands.jsonl"):
        with open("openhands.jsonl", "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    print(
                        f"Found existing entry for instance_id {entry['instance_id']}, skipping.")
                    processed_ids.add(entry["instance_id"])

    # Setup energy measurement
    pyRAPL.setup()
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    for idx, entry in enumerate(data):
        instance_id = entry["instance_id"]

        if instance_id in processed_ids:
            print(f"Skipping {instance_id} as already processed.")
            continue

        print(f"Processing instance {instance_id} ({idx + 1}/{len(data)})")

        # Create config.toml with the current instance_id
        config_path = "/home/sustaind/Desktop/openhands-test/OpenHands/evaluation/benchmarks/swe_bench/config.toml"
        create_config_toml(instance_id, config_path)

        # Initialize measurement variables
        initial_gpu_energy = None
        meter = None
        start_time = None
        measurement_started = False

        # Initialize peak memory tracking
        ram_peak = [0]
        vram_peak = [0]
        stop_event = threading.Event()
        monitor_thread = None

        MODEL_NAME = "unsloth/Qwen3-1.7B-GGUF:BF16"

        # Run the command ./evaluation/benchmarks/swe_bench/scripts/run_infer.sh llm.llm HEAD CodeActAgent 500 1000000000000000000000 1 princeton-nlp/SWE-bench_Verified test
        # Start the mini-extra command in the background
        # Use preexec_fn to create a new process group so we can kill the entire tree
        process = subprocess.Popen(
            [
                "./evaluation/benchmarks/swe_bench/scripts/run_infer.sh",
                "llm.llm",
                "HEAD",
                "CodeActAgent",
                "500",
                "1000000000000000000000",
                "1",
                "princeton-nlp/SWE-bench_Verified",
                "test",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="OpenHands",
            preexec_fn=os.setsid  # Create new session/process group
        )

        # Set stdout and stderr to non-blocking
        fcntl.fcntl(
            process.stdout,
            fcntl.F_SETFL,
            fcntl.fcntl(process.stdout, fcntl.F_GETFL) | os.O_NONBLOCK,
        )
        fcntl.fcntl(
            process.stderr,
            fcntl.F_SETFL,
            fcntl.fcntl(process.stderr, fcntl.F_GETFL) | os.O_NONBLOCK,
        )

        # Monitor output and wait
        stop_reason = "successful"
        process_start_time = time.time()
        timer_start_time = process_start_time  # Initialize timer start time

        while time.time() - timer_start_time < 1800 and process.poll() is None:
            # Read stdout
            try:
                output = process.stdout.read(1024)
                if output:
                    print(output, end="")

                    # Start measurement when we see "BEGIN Runtime Initialization Fn"
                    if "BEGIN Runtime Initialization Fn" in output and not measurement_started:
                        print("\n[MEASUREMENT] Started measurement\n")
                        initial_gpu_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(
                            gpu_handle)
                        meter = pyRAPL.Measurement("cpu")
                        meter.begin()
                        start_time = time.time()
                        timer_start_time = time.time()  # Reset timer start time
                        monitor_thread = threading.Thread(
                            target=monitor_memory, args=(
                                gpu_handle, ram_peak, vram_peak, stop_event)
                        )
                        monitor_thread.start()
                        measurement_started = True

                    # Stop measurement when we see "END Runtime Completion Fn"
                    if "END Runtime Completion Fn" in output and measurement_started:
                        print("\n[MEASUREMENT] Ended measurement\n")
                        stop_event.set()
                        if monitor_thread:
                            monitor_thread.join()
                        measurement_started = False

                    if "litellm.BadRequestError" in output:
                        send_sigint_twice(process)
                        stop_reason = "badrequest"
                        break
            except OSError:
                pass
            # Read stderr
            try:
                output = process.stderr.read(1024)
                if output:
                    print(output, end="")

                    # Start measurement when we see "BEGIN Runtime Initialization Fn"
                    if "BEGIN Runtime Initialization Fn" in output and not measurement_started:
                        print("\n[MEASUREMENT] Started measurement\n")
                        initial_gpu_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(
                            gpu_handle)
                        meter = pyRAPL.Measurement("cpu")
                        meter.begin()
                        start_time = time.time()
                        timer_start_time = time.time()  # Reset timer start time
                        monitor_thread = threading.Thread(
                            target=monitor_memory, args=(
                                gpu_handle, ram_peak, vram_peak, stop_event)
                        )
                        monitor_thread.start()
                        measurement_started = True

                    # Stop measurement when we see "END Runtime Completion Fn"
                    if "END Runtime Completion Fn" in output and measurement_started:
                        print("\n[MEASUREMENT] Ended measurement\n")
                        stop_event.set()
                        if monitor_thread:
                            monitor_thread.join()
                        measurement_started = False

                    if "litellm.BadRequestError" in output:
                        send_sigint_twice(process)
                        stop_reason = "badrequest"
                        break
            except OSError:
                pass

            time.sleep(0.1)

        # If still running, timeout
        if process.poll() is None and not stop_reason == "badrequest":
            send_sigint_twice(
                process,
                f"Instance {instance_id} timed out after 30 minutes. Sending SIGINT.",
            )
            stop_reason = "timeout"

        # Wait for process to finish
        process.wait()

        print(f"Process for instance {instance_id} completed.")

        # Stop memory monitoring if still running
        if measurement_started:
            print("\n[MEASUREMENT] Ended measurement (script completed)\n")
            stop_event.set()
            if monitor_thread:
                monitor_thread.join()

        # End energy measurement
        end_time = time.time() if start_time else process_start_time
        time_taken = round(end_time - start_time, 2) if start_time else 0
        if meter:
            meter.end()
            cpu_energy = meter.result
        else:
            cpu_energy = None

        if initial_gpu_energy is not None:
            final_gpu_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(
                gpu_handle)
            gpu_energy_consumed = final_gpu_energy - initial_gpu_energy
        else:
            gpu_energy_consumed = -1

        # Get the output and input token counts from output.jsonl file
        output_jsonl_path = "./OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/Qwen3-1.7B-GGUF_BF16_maxiter_1000000000000000000000_N_v0.57.0-no-hint-run_1/output.jsonl"
        input_tokens, output_tokens, llm_calls, diff, cost = (
            get_results_from_output_jsonl(
                output_jsonl_path,
                instance_id,
            )
        )

        # Copy the output.jsonl file to logs folder
        copy_output_jsonl_to_logs(output_jsonl_path, instance_id)

        # Save the predictions to a JSONL file with instance_id, model_name, and the diff (model_patch)
        output_entry = {
            "instance_id": instance_id,
            "scaffold": "openhands",
            "model_name": MODEL_NAME,
            "cpu_energy": int(cpu_energy.pkg[0]) if cpu_energy and cpu_energy.pkg else -1,
            "gpu_energy": gpu_energy_consumed,
            "time_taken": time_taken,
            "llm_calls": llm_calls,
            "vram_used": vram_peak[0],
            "ram_used": ram_peak[0],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "stop_reason": stop_reason,
            "model_patch": diff,
        }
        output_file = "openhands.jsonl"
        with open(output_file, "a") as out_file:
            out_file.write(json.dumps(output_entry) + "\n")
        print(
            f"Saved model prediction for instance {instance_id} to {output_file}\n")

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
