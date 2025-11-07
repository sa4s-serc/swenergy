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
import re


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


def get_results_from_trajectory(instance_id):
    """
    Extract token counts from the swe-agent trajectory file.
    Also, copy the file to the logs directory for future reference.

    Args:
        instance_id (str): The instance ID for logging purposes.

    Returns:
        tuple: (input_tokens, output_tokens, llm_calls, diff, cost, predicted_n)
    """
    try:
        # Construct the path to the trajectory file
        base_path = "/home/sustaind/Desktop/big-swe-agent/SWE-agent/trajectories/sustaind"
        traj_file = None
        instance_path = None
        for model_folder in os.listdir(base_path):
            model_path = os.path.join(base_path, model_folder)
            # ⚠️⚠️⚠️ IMPORTANT: REMOVE THE OTHER MODEL FOLDERS WHEN TESTING WITH ANOTHER MODEL ⚠️⚠️⚠️
            if os.path.isdir(model_path):
                instance_path = os.path.join(model_path, instance_id)
                if os.path.isdir(instance_path):
                    potential_traj = os.path.join(
                        instance_path, f"{instance_id}.traj")
                    if os.path.isfile(potential_traj):
                        traj_file = potential_traj
                        break
        if traj_file is None:
            print(f"Trajectory file not found for {instance_id}")
            return -1, -1, -1, "", 0.0, 0

        with open(traj_file, "r") as f:
            traj_data = json.load(f)

        # Extract totals from model_stats
        model_stats = traj_data.get("info", {}).get("model_stats", {})
        total_input_tokens = model_stats.get("tokens_sent", 0)
        total_output_tokens = model_stats.get("tokens_received", 0)
        llm_calls = model_stats.get("api_calls", 0)

        # Debug: Print if tokens are 0
        if total_input_tokens == 0 and total_output_tokens == 0:
            exit_status = traj_data.get("info", {}).get(
                "exit_status", "unknown")
            traj_length = len(traj_data.get("trajectory", []))
            print(
                f"WARNING: {instance_id} has 0 tokens. Exit status: {exit_status}, Trajectory length: {traj_length}")
            print(f"  model_stats: {model_stats}")

        # Get diff from submission
        diff = traj_data.get("info", {}).get("submission", "")
        cost = model_stats.get("instance_cost", 0.0)

        # Parse debug.log for predicted_n
        debug_file = os.path.join(instance_path, f"{instance_id}.debug.log")
        total_predicted_n = 0
        if instance_path:
            debug_file = os.path.join(
                instance_path, f"{instance_id}.debug.log")
            if os.path.isfile(debug_file):
                try:
                    with open(debug_file, "r") as f:
                        content = f.read()
                    predicted_n_matches = re.findall(
                        r"'predicted_n': (\d+)", content)
                    total_predicted_n = sum(int(x)
                                            for x in predicted_n_matches)
                except Exception as e:
                    print(f"Error parsing debug file: {e}")

        # Copy the trajectory file to logs directory, rename it with instance_id
        try:
            logs_dir = "./logs"
            os.makedirs(logs_dir, exist_ok=True)
            dest_path = os.path.join(
                logs_dir, f"{instance_id}_trajectory.json")
            subprocess.run(["cp", traj_file, dest_path], check=True)
        except Exception as e:
            print(f"Error copying trajectory file: {e}")
        return total_input_tokens, total_output_tokens, llm_calls, diff, cost, total_predicted_n

    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error reading trajectory file: {e}")
        return -1, -1, -1, "", 0.0, 0


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
    output_file = "swe-agent.jsonl"
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
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
        initial_gpu_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(
            gpu_handle)
        meter = pyRAPL.Measurement("cpu")
        meter.begin()
        start_time = time.time()

        # Initialize peak memory tracking
        ram_peak = [0]
        vram_peak = [0]
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_memory, args=(
                gpu_handle, ram_peak, vram_peak, stop_event)
        )
        monitor_thread.start()

        MODEL_NAME = "unsloth/Qwen3-1.7B-GGUF:BF16"
        MODEL_INPUT_PRICE = 0.00000011
        MODEL_OUTPUT_PRICE = 0.00000126

        number = numbers[idx]

        # Run the command sweagent run-batch --config swebench.yaml --instances.slice {number}:{number+1}
        # Start the sweagent command in the background
        process = subprocess.Popen(
            [
                "sweagent",
                "run-batch",
                "--config",
                "swebench.yaml",
                "--instances.slice",
                f"{number}:{number+1}",
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
        final_gpu_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(
            gpu_handle)
        gpu_energy_consumed = final_gpu_energy - initial_gpu_energy

        # Get the output and input token counts from trajectory file
        input_tokens, output_tokens, llm_calls, diff, cost, predicted_n = (
            get_results_from_trajectory(instance_id)
        )

        if cost == 0:
            cost = (input_tokens * MODEL_INPUT_PRICE) + \
                (output_tokens * MODEL_OUTPUT_PRICE)

        # Save the predictions to a JSONL file
        output_entry = {
            "instance_id": instance_id,
            "scaffold": "swe-agent",
            "model_name": MODEL_NAME,
            "cpu_energy": int(cpu_energy.pkg[0]) if cpu_energy.pkg else -1,
            "gpu_energy": gpu_energy_consumed,
            "time_taken": time_taken,
            "llm_calls": llm_calls,
            "vram_used": vram_peak[0],
            "ram_used": ram_peak[0],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens + predicted_n,
            "cost": cost,
            "stop_reason": stop_reason,
            "model_patch": diff,
        }
        with open(output_file, "a") as out_file:
            out_file.write(json.dumps(output_entry) + "\n")
        print(
            f"Saved model prediction for instance {instance_id} to {output_file}\n")

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
