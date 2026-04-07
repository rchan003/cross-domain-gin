import itertools
import math
import subprocess
import sys
import time
from argparse import Namespace
from datetime import timedelta
from pathlib import Path

from train import RunManager

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import JSSP_FIXED_PARAMS, JSSP_SEARCH_SPACE

# RUN COMMAND:  nohup python3 -u train.py > runner.log 2>&1 &
# KILL PROCESS: kill <PID>
# GET PID:      ps aux | grep train.py

# Current pid: 1794318 (CHECK BEFORE KILLING!!!)

# Relative paths to relevant directories
JSSP_DIR = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = JSSP_DIR / "scripts" / "train.py"
LOG_DIR = JSSP_DIR / "logs"
RESULTS_DIR = JSSP_DIR / "results"


def build_command(config: dict) -> list[str]:
    cmd = [sys.executable, "-u", TRAIN_SCRIPT]

    for key, value in config.items():
        cmd.append(f"--{key}")

        if isinstance(value, bool):
            cmd.append(str(value).lower())
        else:
            cmd.append(str(value))

    return cmd


def run_experiment(run_name: str, config: dict) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / f"{run_name}.log"
    cmd = build_command(config)

    print(f"Running experiment: {run_name}")
    print("Command:", " ".join(cmd))
    print(f"Logging to: {log_path}")

    with open(log_path, "w") as f:
        process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    return process.returncode


def main():
    search_space = JSSP_SEARCH_SPACE
    fixed_params = JSSP_FIXED_PARAMS

    # Create search space
    keys = list(search_space.keys())
    values_product = itertools.product(*(search_space[k] for k in keys))

    # To store results
    results = []

    # For ETA estimates
    num_experiments = math.prod(len(search_space[k]) for k in keys)
    start_time = time.time()

    # Run Experiments
    for i, values in enumerate(values_product):
        print(f"\nEXPERIMENT {i+1} OF {num_experiments}")

        # Load config
        config = fixed_params.copy()
        config.update(dict(zip(keys, values)))

        # Create run manager to check if this is finished
        args = Namespace(**config)
        run_manager = RunManager(args, base_dir=RESULTS_DIR)
        run_name = run_manager.run_name()

        # Skip if this experiment has already been done
        if run_manager.is_complete():
            print(f"Skipping existing experiment: {run_name}")
            results.append((config, "SKIPPED"))
            continue

        # Run the Experiment
        print("Experiment was not complete before.")
        return_code = run_experiment(run_name=run_name, config=config)
        results.append((config, return_code))

        # Determine the status
        experiment_status = (
            "SUCCESS"
            if return_code == 0 or "SKIPPED"
            else f"FAILED (code={return_code})"
        )

        # Calculate ETA
        t2 = time.time()
        avg_experiment_time = (t2 - start_time) / (i + 1)
        eta = timedelta(seconds=int((num_experiments - (i + 1)) * avg_experiment_time))
        print(f"Finished Experiment {i+1} | Status: {experiment_status} | ETA: {eta}")

    # Output result of all experiments
    print("\nAll experiments finished.\n")
    for config, code in results:
        status = "SUCCESS" if code == 0 or "SKIPPED" else f"FAILED (code={code})"
        print(f"{status}: {config}")


if __name__ == "__main__":
    main()
