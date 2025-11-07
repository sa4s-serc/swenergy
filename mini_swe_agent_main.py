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

# Columns:
# - 'repo'
# - 'instance_id'
# - 'base_commit'
# - 'patch'
# - 'test_patch'
# - 'problem_statement'
# - 'hints_text'
# - 'created_at'
# - 'version'
# - 'FAIL_TO_PASS'
# - 'PASS_TO_PASS'
# - 'environment_setup_commit'


def send_sigint_twice(process, message=None):
    """
    Send SIGINT to the process, wait 1 second, and send again if still running.
    """
    process.send_signal(signal.SIGINT)
    time.sleep(1)
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    if message:
        print(message)


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


def get_results_from_trajectory(traj_file_path, instance_id):
    """
    Extract token counts from the mini-swe-agent trajectory file.
    Also, copy the file to the logs directory for future reference.

    Args:
        traj_file_path (str): Path to the trajectory JSON file.
        instance_id (str): The instance ID for logging purposes.

    Returns:
        tuple: (input_tokens, output_tokens, llm_calls)
    """
    try:
        # Expand the tilde in the path
        expanded_path = os.path.expanduser(traj_file_path)

        with open(expanded_path, "r") as f:
            traj_data = json.load(f)

        total_input_tokens = 0
        total_output_tokens = 0
        llm_calls = 0

        diff = traj_data.get("info", {}).get("submission", "")
        cost = traj_data.get("info", {}).get("model_stats", {}).get("instance_cost", -1)

        # Extract tokens from messages with 'extra' field containing response data
        messages = traj_data.get("messages", [])
        for message in messages:
            if message.get("role") == "assistant" and "extra" in message:
                response_data = message["extra"].get("response", {})
                usage = response_data.get("usage", {})

                if usage:
                    llm_calls += 1
                    total_input_tokens += usage.get("prompt_tokens", 0)
                    total_output_tokens += usage.get("completion_tokens", 0)

        # Copy the trajectory file to logs directory, rename it with instance_id
        logs_dir = "./logs"
        os.makedirs(logs_dir, exist_ok=True)
        dest_path = os.path.join(logs_dir, f"{instance_id}_trajectory.json")
        subprocess.run(["cp", expanded_path, dest_path], check=True)

        return total_input_tokens, total_output_tokens, llm_calls, diff, cost

    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error reading trajectory file: {e}")
        return -1, -1, -1, "", 0.0


def main():
    # Load the mini dataset
    f = open("swe-bench-verified-mini.jsonl")
    data = [json.loads(line) for line in f]
    f.close()

    # Load the verified line numbers
    f = open("mini_verified_line_numbers.txt", "r")
    numbers = [int(line.strip()) for line in f]
    f.close()

    # Load existing processed instance_ids to skip duplicates
    processed_ids = set()
    if os.path.exists("mini-swe-agent.jsonl"):
        with open("mini-swe-agent.jsonl", "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    print(f"Found existing entry for instance_id {entry['instance_id']}, skipping.")
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

        # Start energy measurement
        initial_gpu_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
        meter = pyRAPL.Measurement("cpu")
        meter.begin()
        start_time = time.time()

        # Initialize peak memory tracking
        ram_peak = [0]
        vram_peak = [0]
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_memory, args=(gpu_handle, ram_peak, vram_peak, stop_event)
        )
        monitor_thread.start()

        MODEL_NAME = "unsloth/Qwen3-1.7B-GGUF:BF16"

        number = numbers[idx]

        # Run the command mini-extra swebench-single --subset verified --split test -i {number} --config swebench.yaml --exit-immediately
        # Start the mini-extra command in the background
        process = subprocess.Popen(
            [
                "mini-extra",
                "swebench-single",
                "--subset",
                "verified",
                "--split",
                "test",
                "-i",
                str(number),
                "--config",
                "swebench.yaml",
                "--exit-immediately",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
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
        start_time = time.time()
        while time.time() - start_time < 1800 and process.poll() is None:
            # Read stdout
            try:
                output = process.stdout.read(1024)
                if output:
                    print(output, end="")
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

        # Stop memory monitoring
        stop_event.set()
        monitor_thread.join()

        # End energy measurement
        end_time = time.time()
        time_taken = round(end_time - start_time, 2)
        meter.end()
        cpu_energy = meter.result
        final_gpu_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
        gpu_energy_consumed = final_gpu_energy - initial_gpu_energy

        # Get the output and input token counts from trajectory file
        input_tokens, output_tokens, llm_calls, diff, cost = (
            get_results_from_trajectory(
                "~/.config/mini-swe-agent/last_swebench_single_run.traj.json",
                instance_id,
            )
        )

        # Save the predictions to a JSONL file with instance_id, model_name, and the diff (model_patch)
        output_entry = {
            "instance_id": instance_id,
            "scaffold": "mini-swe-agent",
            "model_name": MODEL_NAME,
            "cpu_energy": int(cpu_energy.pkg[0]) if cpu_energy.pkg else -1,
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
        output_file = "mini-swe-agent.jsonl"
        with open(output_file, "a") as out_file:
            out_file.write(json.dumps(output_entry) + "\n")
        print(f"Saved model prediction for instance {instance_id} to {output_file}\n")

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
