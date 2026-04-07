import itertools
import math
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DDI_EXPERIMENT_SET_NAME, DDI_FIXED_PARAMS, DDI_SEARCH_SPACE

# cd /mnt/4tb/rachel_thesis/cross-domain-gin/DDI
# RUN COMMAND:  nohup python3 -u scripts/run_experiments.py > runner.log 2>&1 &
# KILL PROCESS: kill <PID>
# GET PID:      ps aux | grep run_experiments.py
# Get nvidia processes: nvidia-smi

# Project directories
DDI_DIR = Path(__file__).resolve().parent.parent
JSSP_DIR = Path(__file__).resolve().parent.parent.parent / "JSSP"

# Input paths
EXPERIMENT_SCRIPT = DDI_DIR / "scripts" / "train.py"
SAVED_MODEL_DIR = JSSP_DIR / "results"

# Output paths
LOG_DIR = DDI_DIR / "logs"
RESULTS_DIR = DDI_DIR / "results"


def build_command(config: dict) -> List[str]:
    cmd = [sys.executable, "-u", str(EXPERIMENT_SCRIPT)]

    for key, value in config.items():
        cmd.append(f"--{key}")
        if isinstance(value, bool):
            cmd.append(str(value).lower())
        else:
            cmd.append(str(value))

    return cmd


def build_run_name(config: dict) -> str:
    key_map = {
        "init_mode": "init",
        "dataset": "ds",
        "loss_function": "loss",
        "batch_size": "bs",
        "epochs": "ep",
        "lr": "lr",
        "num_neg_per_pos": "neg",
        "pretrained_path": "pre",
        "split_seed": "split",
        "model_seed": "seed",
    }

    important_keys = [
        "init_mode",
        "loss_function",
        "lr",
        "batch_size",
        "pretrained_path",
        "split_seed",
        "model_seed",
    ]

    parts = []
    for k in important_keys:
        if k not in config:
            continue

        v = config[k]
        short_k = key_map.get(k, k)

        if k == "pretrained_path" and v is not None:
            v = Path(v).parents[1].name
            if len(v) > 30:
                v = "..." + v[-27:]

        if v == "binary_cross_entropy":
            v = "bce"
        elif v == "pairwise_ranking":
            v = "rank"

        parts.append(f"{short_k}={v}")

    timestamp = time.strftime("%m%d_%H%M")
    return f"{timestamp}__{'__'.join(parts)}"


def run_experiment(log_path: Path, run_name: str, config: dict) -> int:
    cmd = build_command(config)

    print(f"Running experiment: {run_name}")
    print("Command:", " ".join(cmd))
    print(f"Logging to: {log_path}")

    with open(log_path, "w") as f:
        process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    return process.returncode


def get_saved_models():
    return [
        p.resolve()
        for p in Path(SAVED_MODEL_DIR).glob("*/checkpoints/best_gin_incumbent.pth")
    ]


def main():
    experiment_set_name = DDI_EXPERIMENT_SET_NAME

    exp_log_dir = LOG_DIR / experiment_set_name
    exp_results_dir = RESULTS_DIR / experiment_set_name
    exp_log_dir.mkdir(parents=True, exist_ok=True)

    saved_models_list = get_saved_models()  # [:1]  # TODO: remove [:1]

    search_space = DDI_SEARCH_SPACE
    fixed_params = DDI_FIXED_PARAMS.copy()
    fixed_params["results_dir"] = str(exp_results_dir)

    # Create search space
    keys = list(search_space.keys())
    values_product = list(itertools.product(*(search_space[k] for k in keys)))

    # To store results
    results = []
    global_idx = 0

    # For ETA estimates
    num_experiments = math.prod(len(search_space[k]) for k in keys) * len(
        saved_models_list
    )
    start_time = time.time()

    print(f"Dataset: {fixed_params['dataset']}")
    print(f"Saved models: {len(saved_models_list)}")
    print(f"Runs per experiment: {fixed_params['runs']}")
    print(f"Total experiments: {num_experiments}")

    # Seed scheduling
    base_split_seed = 10_000
    base_model_seed = 100_000

    for model_idx, saved_model in enumerate(saved_models_list):
        # One paired seed schedule per saved model
        split_seed = base_split_seed + model_idx * 1000
        model_seed = base_model_seed + model_idx * 1000

        for values in values_product:
            global_idx += 1
            print(f"\nEXPERIMENT {global_idx} OF {num_experiments}")

            # Load config
            config = fixed_params.copy()
            config.update(dict(zip(keys, values)))
            config.update({"pretrained_path": str(saved_model)})

            # Same split schedule for pretrained/random on the same saved model
            config["split_seed"] = split_seed
            config["model_seed"] = model_seed

            # Create run name and log path
            run_name = build_run_name(config)
            log_path = exp_log_dir / f"{run_name}.log"

            # Run the Experiment
            return_code = run_experiment(
                log_path=log_path,
                run_name=run_name,
                config=config,
            )
            results.append((config, return_code))

            # Determine the status
            experiment_status = (
                "SUCCESS" if return_code == 0 else f"FAILED (code={return_code})"
            )

            elapsed = time.time() - start_time
            avg_experiment_time = elapsed / global_idx
            eta = timedelta(
                seconds=int((num_experiments - global_idx) * avg_experiment_time)
            )

            print(
                f"Finished Experiment {global_idx} | "
                f"Status: {experiment_status} | ETA: {eta}"
            )

    print("\nAll experiments finished.\n")
    total_time = timedelta(seconds=int(time.time() - start_time))
    print(f"Total Time: {total_time}")

    for config, code in results:
        status = "SUCCESS" if code == 0 else f"FAILED (code={code})"
        print(f"{status}: {config}")


if __name__ == "__main__":
    main()
