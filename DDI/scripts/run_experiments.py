import itertools
import math
import subprocess
import sys
import time
from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from typing import List

# TODO: make some learning rate schedule and hyperparameter tuning thing then run exerpiments with best settings

#python3 src/main_rachel_gin.py --dataset "ogbl-ddi" --pretrained_path "/mnt/4tb/rachel_thesis/L2S-main/saved_rachel_model/20x15[1,99]_fdd-divide-mwkr_yaoxin_1.0_256_4_4_gin_NAN_0.0001_10_500_64_640_1/checkpoints/best_gin_incumbent.pth" --init_mode "pretrained" --epochs 100 --batch_size 18000 --lr 1e-6

# cd /mnt/4tb/rachel_thesis/DDI
# RUN COMMAND:  nohup python3 -u src/runner_rachel.py > runner.log 2>&1 &
# KILL PROCESS: kill <PID>
# GET PID:      ps aux | grep runner_rachel.py 
# Get nvidia processes: nvidia-smi

# Current pid: 2044683 (CHECK BEFORE KILLING!!!)

# Name of training file
EXPERIMENT_SCRIPT = "/mnt/4tb/rachel_thesis/DDI/src/main_rachel_gin.py"
SAVED_MODEL_DIR = "/mnt/4tb/rachel_thesis/L2S-main/saved_rachel_model"

# Root directories for results 
LOG_ROOT_DIR = Path("/mnt/4tb/rachel_thesis/DDI/src/rachel/experiment_run_logs")
RESULTS_ROOT_DIR = Path("/mnt/4tb/rachel_thesis/DDI/src/rachel/experiment_results")

def build_command(config: dict) -> List[str]:
    cmd = [sys.executable, "-u", EXPERIMENT_SCRIPT]

    for key, value in config.items():
        cmd.append(f"--{key}")

        if isinstance(value, bool):
            cmd.append(str(value).lower())
        else:
            cmd.append(str(value))

    return cmd

def build_run_name(config: dict) -> str:
    # 1. Map long keys to short identifiers
    key_map = {
        "init_mode": "init",
        "dataset": "ds",
        "loss_function": "loss",
        "batch_size": "bs",
        "epochs": "ep",
        "lr": "lr",
        "num_neg_per_pos": "neg",
        "pretrained_path": "pre"
    }

    # 2. Define what is actually important to see in the filename
    # (Exclude things like device, eval_steps, or runs)
    important_keys = ["init_mode", "loss_function", "lr", "batch_size", "pretrained_path"]
    
    parts = []
    for k in important_keys:
        if k not in config:
            continue
            
        v = config[k]
        short_k = key_map.get(k, k)

        # Handle the pretrained path string specifically
        if k == "pretrained_path" and v is not None:
            # Gets just the specific experiment folder name
            v = Path(v).parents[1].name 
            # If the folder name itself is too long, take the last 30 chars
            if len(v) > 30:
                v = "..." + v[-27:]
        
        # Shorten loss function names
        if v == "binary_cross_entropy": v = "bce"
        if v == "pairwise_ranking": v = "rank"

        parts.append(f"{short_k}={v}")

    # 3. Add a timestamp at the front so files sort chronologically
    timestamp = time.strftime("%m%d_%H%M")
    return f"{timestamp}__{'__'.join(parts)}"

def run_experiment(log_path, run_name: str, config: dict) -> int:
    cmd = build_command(config)

    print(f"Running experiment: {run_name}")
    print("Command:", " ".join(cmd))
    print(f"Logging to: {log_path}")

    with open(log_path, "w") as f:
        process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    return process.returncode

def get_saved_models():
    paths = [
        p.resolve()
        for p in Path(SAVED_MODEL_DIR).glob("*/checkpoints/best_gin_incumbent.pth")
    ]
    return paths

def main():
    # Name of this set of this experiments 
    experiment_set_name: str = "ogbl-ddi" # NOTE: Update this
    save_results: bool = True                # NOTE: Update this

    # Update output paths
    log_dir = LOG_ROOT_DIR / experiment_set_name
    results_dir = RESULTS_ROOT_DIR / experiment_set_name

    # Get all the model checkpoints
    saved_models_list = get_saved_models()

    # Put the parameters you want to vary here
    search_space = {
        "init_mode": ["pretrained", "random"],
        #"batch_size": [10000],
    }

    # Optional fixed parameters for every run
    # TODO: change dataset and loss function
    fixed_params = {
        "dataset": "ogbl-ddi",
        "loss_function": "binary_cross_entropy",
        "runs": 20,
        "device": 0,
        "batch_size": 18000,
        "epochs": 100,
        "eval_steps": 1,
        "lr": 1e-6, # TODO: try to tune lr 
        "dropout": 0.1,
        "num_neg_per_pos": 1,
        "results_dir": results_dir,
        "save_results": save_results,
    }

    # Create search space
    keys = list(search_space.keys())
    values_product = list(itertools.product(*(search_space[k] for k in keys)))

    # To store results
    results = []
    global_idx = 0

    # For ETA estimates
    num_experiments = math.prod(len(search_space[k]) for k in keys) * len(saved_models_list)
    start_time = time.time()

    # To store runner logs
    log_dir.mkdir(parents=True, exist_ok=True)

    # Run Experiments
    for saved_model in saved_models_list:
        for values in values_product:
            global_idx += 1
            print(f"\nEXPERIMENT {global_idx} OF {num_experiments}")

            # Load config
            config = fixed_params.copy()
            config.update(dict(zip(keys, values)))
            config.update({"pretrained_path": str(saved_model)})

            # Parse arguments and create run name 
            args = Namespace(**config)
            run_name = build_run_name(config)

            # Create the runner log file
            log_path = log_dir / f"{run_name}.log"

            #Run the Experiment
            return_code = run_experiment(log_path=log_path, run_name=run_name, config=config)
            results.append((config, return_code))

            # Determine the status
            experiment_status = "SUCCESS" if return_code == 0 else f"FAILED (code={return_code})"

            # Calculate ETA
            t2 = time.time()
            avg_experiment_time = (t2 - start_time) / (global_idx)
            eta = timedelta(seconds=int((num_experiments - (global_idx)) * avg_experiment_time))
            print(f"Finished Experiment {global_idx} | Status: {experiment_status} | ETA: {eta}")

    # Output result of all experiments
    print("\nAll experiments finished.\n")
    end_time = time.time()
    total_time = timedelta(seconds=int(start_time - end_time))
    print(f"Total Time: {total_time}")
    for config, code in results:
        status = "SUCCESS" if code == 0 else f"FAILED (code={code})"
        print(f"{status}: {config}")


if __name__ == "__main__":
    main()