import argparse
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.actor import Actor
from src.environment import BatchGraph, JsspN5
from src.generateJSP import uni_instance_gen

# For quickly updating log file when this is run
FLUSH: bool = True
# TODO: UPDATE PATH HERE
DATA_PATH = Path("/mnt/4tb/rachel_thesis/cross-domain-gin/JSSP/data")
BASE_RUN_DIR = Path("/mnt/4tb/rachel_thesis/cross-domain-gin/JSSP/results")


class RunManager:
    """
    Manages training runs including:
        - Initalizes directories
        - Saves checkpoints
        - Saves logs
        etc.
    """

    def __init__(self, args, base_dir: str):
        self.args = args
        self.base_dir = Path(base_dir)

        # Directory Paths
        self.run_dir = self.base_dir / self.run_name()
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.log_dir = self.run_dir / "logs"

        # File Paths
        self.metadata_file = self.run_dir / "metadata.txt"
        self.finish_file = self.run_dir / "finished.txt"

    def mark_complete(self):
        self.finish_file.write_text("success!\n")

    def is_complete(self) -> bool:
        return self.finish_file.exists()

    def initialize_run_directory(self):
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize or update metadata file
        self._add_run_to_metadata()

    def save_log(self, name, log):
        log_path = self.log_dir / f"{name}.npy"
        np.save(log_path, np.array(log))

    def save_checkpoint(self, policy, prefix, batch_i: int, is_improved: bool):
        # Save full actor if improved
        if is_improved:
            print(
                f"Found better model w.r.t {prefix} objs, updating actor weights...",
                flush=FLUSH,
            )
            actor_path = self.checkpoint_dir / f"best_{prefix}.pth"
            torch.save(policy.state_dict(), actor_path)

        # Save GIN-only module every time called
        gin_module = self._get_gin_module(policy)
        if gin_module is not None:
            print(f"Saving {prefix} GIN model for batch {batch_i}...", flush=FLUSH)
            gin_path = self.checkpoint_dir / f"gin_{prefix}_batch_{batch_i}.pth"
            torch.save(gin_module.state_dict(), gin_path)

            if is_improved:
                print(f"Updating best model for {prefix} GIN...", flush=FLUSH)
                gin_best_path = self.checkpoint_dir / f"best_gin_{prefix}.pth"
                torch.save(gin_module.state_dict(), gin_best_path)

    def update_run_metadata(self, text: str):
        with open(self.metadata_file, "a") as f:
            f.write(f"{text}\n")

    def run_name(self):
        gamma = float(self.args.gamma)
        lr = float(self.args.lr)

        dghan_param_for_saved_model = self._get_dghan_param_for_saved_model()
        return (
            f"{self.args.j}x{self.args.m}[{self.args.l},{self.args.h}]_"
            f"{self.args.init_type}_{self.args.reward_type}_{gamma}_"
            f"{self.args.hidden_dim}_{self.args.embedding_layer}_{self.args.policy_layer}_{self.args.embedding_type}_"
            f"{dghan_param_for_saved_model}_"
            f"{lr}_{self.args.steps_learn}_{self.args.transit}_{self.args.batch_size}_{self.args.episodes}_{self.args.step_validation}"
        )

    def _add_run_to_metadata(self):
        mode = "a" if self.metadata_file.exists() else "w"
        with open(self.metadata_file, mode) as f:
            if mode == "w":
                f.write("Run Metadata\n")
                f.write("====================\n")
                f.write(f"Run Name: {self.run_name()}\n")
                f.write("\nRun Parameters\n")
                for key, value in vars(self.args).items():
                    f.write(f"{key}: {value}\n")

            f.write("\n--------------------\n")
            f.write(f"Update Time: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")

    def _get_gin_module(self, policy):
        if self.args.embedding_type == "gin":
            return policy.embedding
        if self.args.embedding_type == "gin+dghan":
            return policy.embedding_gin
        return None

    def _get_dghan_param_for_saved_model(self):
        if self.args.embedding_type == "gin":
            return "NAN"
        if self.args.embedding_type in {"dghan", "gin+dghan"}:
            return f"{self.args.heads}_{self.args.drop_out}"
        raise ValueError(
            'embedding_type should be one of "gin", "dghan", or "gin+dghan".'
        )


class RL2S4JSSP:
    def __init__(self, args):
        self.args = args

        # Check that this is a valid instance to test first
        self.validation_optimal = self._load_optimal_validation_data()
        self.validation_data = self._load_or_create_validation_data()

        # Load environments
        self.env_training = JsspN5(
            n_job=self.args.j,
            n_mch=self.args.m,
            low=self.args.l,
            high=self.args.h,
            reward_type=self.args.reward_type,
        )
        self.env_validation = JsspN5(
            n_job=self.args.j,
            n_mch=self.args.m,
            low=self.args.l,
            high=self.args.h,
            reward_type=self.args.reward_type,
        )

        # Comparisons for training
        self.eps = np.finfo(np.float32).eps.item()
        self.incumbent_validation_result = np.inf
        self.current_validation_result = np.inf

        # Create run manager and initialize run directory
        self.run_manager = RunManager(self.args, base_dir=BASE_RUN_DIR)
        self.run_manager.initialize_run_directory()

        # Save the batch where the best occured
        self.best_incumbent_batch = -1
        self.best_last_step_batch = -1

    def _load_optimal_validation_data(self):
        path = (
            DATA_PATH
            / "validation_data"
            / f"validation{self.args.j}x{self.args.m}_ortools_result.npy"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Optimal validation data doesn't exist for: j={self.args.j}, m={self.args.m}. Exiting training."
            )
        return np.load(path)

    def _load_or_create_validation_data(self):
        path = (
            DATA_PATH
            / "validation_data"
            / f"validation_instance_{self.args.j}x{self.args.m}[{self.args.l},{self.args.h}].npy"
        )
        if path.is_file():
            return np.load(path)

        print(
            f"No validation data for {self.args.j}x{self.args.m}[{self.args.l},{self.args.h}], generating new one.",
            flush=FLUSH,
        )
        validation_data = np.array(
            [
                uni_instance_gen(
                    n_j=self.args.j, n_m=self.args.m, low=self.args.l, high=self.args.h
                )
                for _ in range(100)
            ]
        )
        np.save(path, validation_data)
        return validation_data

    def _evaluate_policy(self, env, policy, dev, instances):
        batch_data = BatchGraph()
        states, feasible_actions, _ = env.reset(
            instances=instances,
            init_type=self.args.init_type,
            device=dev,
        )

        while env.itr < self.args.transit:
            batch_data.wrapper(*states)
            actions, _ = policy(batch_data, feasible_actions)
            states, _, feasible_actions, _ = env.step(actions, dev)

        batch_data.clean()

        incumbent_result = env.incumbent_objs.mean().cpu().item()
        current_result = env.current_objs.mean().cpu().item()
        return incumbent_result, current_result

    def learn(self, rewards, log_probs, dones, optimizer):
        R = torch.zeros_like(rewards[0], dtype=torch.float, device=rewards[0].device)
        returns = []

        for r in rewards[::-1]:
            R = r + self.args.gamma * R
            returns.insert(0, R)

        returns = torch.cat(returns, dim=-1)
        dones = torch.cat(dones, dim=-1)
        log_probs = torch.cat(log_probs, dim=-1)

        losses = []
        for b in range(returns.shape[0]):
            masked_R = torch.masked_select(returns[b], ~dones[b])

            if masked_R.numel() > 1:
                masked_R = (masked_R - masked_R.mean()) / (
                    torch.std(masked_R, unbiased=False) + self.eps
                )
            else:
                masked_R = masked_R - masked_R.mean()

            masked_log_prob = torch.masked_select(log_probs[b], ~dones[b])
            loss = (-masked_log_prob * masked_R).sum()
            losses.append(loss)

        optimizer.zero_grad()
        mean_loss = torch.stack(losses).mean()
        mean_loss.backward()
        optimizer.step()

    def validation(self, policy, dev, batch_i: int):
        validation_start = time.time()

        validation_result_incumbent, validation_result_current = self._evaluate_policy(
            env=self.env_validation,
            policy=policy,
            dev=dev,
            instances=self.validation_data,
        )

        incumbent_gap = (
            (self.env_validation.incumbent_objs.cpu().numpy() - self.validation_optimal)
            / self.validation_optimal
        ).mean()

        current_gap = (
            (self.env_validation.current_objs.cpu().numpy() - self.validation_optimal)
            / self.validation_optimal
        ).mean()

        validation_end = time.time()

        print("\nValidation Results", flush=FLUSH)
        print(
            "\tIncumbent objs:  {:.2f}".format(validation_result_incumbent), flush=FLUSH
        )
        print(
            "\tFinal objs:      {:.2f}".format(validation_result_current), flush=FLUSH
        )
        print("\tIncumbent gap: {:.2%}".format(incumbent_gap), flush=FLUSH)
        print("\tFinal gap:     {:.2%}".format(current_gap), flush=FLUSH)

        # Update best incumbent score
        is_incumbent_improved = False
        if validation_result_incumbent < self.incumbent_validation_result:
            is_incumbent_improved = True
            self.incumbent_validation_result = validation_result_incumbent
            self.best_incumbent_batch = batch_i

        # Update best last-step score
        is_last_improved = False
        if validation_result_current < self.current_validation_result:
            is_last_improved = True
            self.current_validation_result = validation_result_current
            self.best_last_step_batch = batch_i

        # Save model checkpoint for both incumbent and last
        self.run_manager.save_checkpoint(
            policy,
            prefix="incumbent",
            batch_i=batch_i,
            is_improved=is_incumbent_improved,
        )
        self.run_manager.save_checkpoint(
            policy, prefix="last-step", batch_i=batch_i, is_improved=is_last_improved
        )

        print(
            "Validation took {:.2f}s\n".format(validation_end - validation_start),
            flush=FLUSH,
        )

        return validation_result_incumbent, validation_result_current

    def train(self):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {dev}", flush=FLUSH)
        print(f"Run Parameters: {self.args}", flush=FLUSH)

        torch.manual_seed(1)
        random.seed(1)
        np.random.seed(1)

        policy = Actor(
            in_dim=3,
            hidden_dim=self.args.hidden_dim,
            embedding_l=self.args.embedding_layer,
            policy_l=self.args.policy_layer,
            embedding_type=self.args.embedding_type,
            heads=self.args.heads,
            dropout=self.args.drop_out,
        ).to(dev)

        # NOTE: Saving the gin architecture in metadata
        gin = (
            policy.embedding
            if self.args.embedding_type == "gin"
            else policy.embedding_gin
        )
        self.run_manager.update_run_metadata(text="GIN Architecture:")
        for i, layer in enumerate(gin.GIN_layers):
            self.run_manager.update_run_metadata(text=f"\nLayer {i}")
            self.run_manager.update_run_metadata(text=str(layer))

        optimizer = optim.Adam(policy.parameters(), lr=self.args.lr)

        batch_data = BatchGraph()
        log = []
        validation_log = []

        total_num_batches = self.args.episodes // self.args.batch_size
        print()
        start_time = time.time()

        for batch_i in range(1, total_num_batches + 1):
            t1 = time.time()

            instances = np.array(
                [
                    uni_instance_gen(self.args.j, self.args.m, self.args.l, self.args.h)
                    for _ in range(self.args.batch_size)
                ]
            )
            states, feasible_actions, dones = self.env_training.reset(
                instances=instances,
                init_type=self.args.init_type,
                device=dev,
            )

            rewards_buffer = []
            log_probs_buffer = []
            dones_buffer = [dones]

            while self.env_training.itr < self.args.transit:
                batch_data.wrapper(*states)
                actions, log_ps = policy(batch_data, feasible_actions)
                states, rewards, feasible_actions, dones = self.env_training.step(
                    actions, dev
                )

                rewards_buffer.append(rewards)
                log_probs_buffer.append(log_ps)
                dones_buffer.append(dones)

                if self.env_training.itr % self.args.steps_learn == 0:
                    self.learn(
                        rewards_buffer, log_probs_buffer, dones_buffer[:-1], optimizer
                    )
                    rewards_buffer = []
                    log_probs_buffer = []
                    dones_buffer = [dones]

            # Calculate ETA
            t2 = time.time()
            avg_batch_time = (t2 - start_time) / batch_i
            eta = timedelta(seconds=int((total_num_batches - batch_i) * avg_batch_time))

            # Retreive mean performance
            mean_perf = self.env_training.current_objs.mean().cpu().item()

            print(
                "Batch {} of {}".format(batch_i, total_num_batches),
                "| Training took: {:.2f}s".format(t2 - t1),
                "| Mean Performance: {:.2f}".format(mean_perf),
                "| ETA: {}".format(eta),
                flush=FLUSH,
            )

            log.append(mean_perf)

            if batch_i % self.args.step_validation == 0:
                validation_result1, validation_result2 = self.validation(
                    policy, dev, batch_i
                )
                validation_log.append([validation_result1, validation_result2])

                self.run_manager.save_log(name="training_log", log=log)
                self.run_manager.save_log(name="validation_log", log=validation_log)

        # Update metadata with the best last batch
        best_batch_message = (
            f"Best Incumbent Batch: {self.best_incumbent_batch}\n"
            + f"Best Last Step Batch: {self.best_last_step_batch}"
        )
        self.run_manager.update_run_metadata(text=best_batch_message)
        print(best_batch_message)

        # Mark the training as finished
        finish_message = f"[{datetime.now().strftime('%H:%M:%S')}] TRAINING FINISHED"
        self.run_manager.update_run_metadata(text=finish_message)
        self.run_manager.mark_complete()
        print(finish_message)


def main(args):
    agent = RL2S4JSSP(args)
    agent.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRL-LSJSP")
    # env parameters
    parser.add_argument("--j", type=int, default=10)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--l", type=int, default=1)
    parser.add_argument("--h", type=int, default=99)
    parser.add_argument("--init_type", type=str, default="fdd-divide-mwkr")
    parser.add_argument("--reward_type", type=str, default="yaoxin")
    parser.add_argument("--gamma", type=float, default=1)
    # model parameters
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embedding_layer", type=int, default=4)
    parser.add_argument("--policy_layer", type=int, default=4)
    parser.add_argument(
        "--embedding_type", type=str, default="gin+dghan"
    )  # TODO: edited here 'gin+dghan')  # 'gin', 'dghan', 'gin+dghan'
    parser.add_argument("--heads", type=int, default=1)  # dghan parameters
    parser.add_argument("--drop_out", type=float, default=0.0)  # dghan parameters
    # training parameters
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--steps_learn", type=int, default=10)
    parser.add_argument("--transit", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--episodes", type=int, default=64
    )  # TODO: edited, used to be 128000
    parser.add_argument(
        "--step_validation", type=int, default=1
    )  # TODO: Edited, used to be 10
    args = parser.parse_args()

    main(args)
