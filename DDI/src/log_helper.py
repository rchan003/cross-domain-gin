import os
import shutil
from datetime import datetime
from pathlib import Path

from logger import Logger


class LogHelper:
    def __init__(self, args, model_name: str, result_dir: str, experiment_name: str, mode_part, model, arch, in_channels):
        self.args = args
        self.dataset_name = args.dataset
        self.model_name = model_name
        self.model = model
        self.arch = arch
        self.in_channels = in_channels

        parent_folder = os.path.join(result_dir, experiment_name)
        if args.pretrained_path is not None:
            checkpoint_experiment_name = Path(args.pretrained_path).parent.parent.name
            parent_folder += f"_{checkpoint_experiment_name}"
        self.log_dir = os.path.join(parent_folder, mode_part)
        self.finished_flag_path = os.path.join(self.log_dir, "finished.txt")

        # Skip logic
        if os.path.exists(self.finished_flag_path) and not args.force_rerun:
            print(f"\n=== Experiment already completed: {self.log_dir} ===")
            self.skip = True
            return
        else:
            self.skip = False
        
        # Exit if not saving results
        if not args.save_results:
            return
        
        # Remove old folder if rerunning
        if os.path.exists(self.log_dir):
            print(f"\n=== Clearing existing experiment folder: {self.log_dir} ===")
            shutil.rmtree(self.log_dir)

        os.makedirs(self.log_dir, exist_ok=True)

        self.loggers = self.__create_loggers(args)
        self.__write_metadata_file()

        # overall results
        self.overall_results = [[] for _ in range(args.runs)]

    def __write_metadata_file(self):
        metadata_path = os.path.join(self.log_dir, "metadata.txt")

        with open(metadata_path, "w", encoding="utf-8") as f:
            now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            f.write(f"Run started at: {now}\n\n")
            f.write("=== RUN ARGS ===\n")
            for k, v in sorted(vars(self.args).items()):
                f.write(f"  {k}: {v}\n")
            f.write("\n")
            
            f.write("=== ARCHITECTURE ===\n")
            if self.arch is not None:
                for k, v in self.arch.items():
                    f.write(f"{k}: {v}\n")
            
            f.write(f"in_channels: {self.in_channels}\n")
            f.write(f"hidden_channels: {self.args.hidden_channels}\n")
            f.write(f"num_layers: {self.args.num_layers}\n\n")
            
            f.write("=== MODEL STRUCTURE ===\n")
            f.write(str(self.model) + "\n\n")

            f.write("=== PARAMETER SHAPES ===\n")
            for name, param in self.model.named_parameters():
                f.write(f"{name}: {tuple(param.shape)}\n")
            f.write("\n")

    def mark_finished(self):
        with open(self.finished_flag_path, "w") as f:
            f.write("done\n")

    def __create_loggers(self, args):
        return {
            "Hits@1": Logger(args.runs, args),
            "Hits@3": Logger(args.runs, args),
            "Hits@10": Logger(args.runs, args),
            "Hits@20": Logger(args.runs, args),
            "Hits@30": Logger(args.runs, args),
            "Hits@40": Logger(args.runs, args),
            "Hits@50": Logger(args.runs, args),
        }

    def log_results(self, results, run, epoch, loss):
        log_file_path = os.path.join(
            self.log_dir,
            f"run{run+1}_{self.dataset_name}_{self.model_name}_train_validation.csv",
        )
        file_exists = os.path.exists(log_file_path)

        with open(log_file_path, "a") as log_file:
            if not file_exists:
                log_file.write("run,epoch,loss,accuracy,auc,ap,f1,recall,hits@k,train,valid,test\n")
            
            # Get overall results first
            acc, auc, ap, f1, recall = results["Overall"]
            print_run_message = (
                f"Run: {run + 1:02d}, "
                f"Epoch: {epoch:02d}, "
                f"Loss: {loss:.4f} "
                f"|| TEST RESULTS: "
                f"Accuracy: {100 * acc:.4f}, "
                f"AUC: {100 * auc:.4f}, "
                f"AP: {100 * ap:.4f}, "
                f"F1: {100 * f1:.4f}, "
                f"Recall: {100 * ap:.4f}"
            )
            print(print_run_message)

            for key, result in results.items():
                if key == "Overall":
                    self.overall_results[run].append(result)
                    continue
                self.loggers[key].add_result(run, result)
                train_hits, valid_hits, test_hits = result

                # Update the csv file
                k_value = int(key.split("@")[1])
                log_file.write(
                    f"{run+1},{epoch},{loss:.4f},"
                    f"{100*acc:.4f},{100*auc:.4f},{100*ap:.4f},"
                    f"{100*f1:.4f},{100*recall:.4f},"
                    f"{k_value},"
                    f"{100*train_hits:.4f},{100*valid_hits:.4f},{100*test_hits:.4f}\n"
                )

                print_hits_message = (
                    f"{key:>10}  "
                    f"Train: {100 * train_hits:.2f}%  "
                    f"Valid: {100 * valid_hits:.2f}%  "
                    f"Test: {100 * test_hits:.2f}%"
                )
                print(f"{print_hits_message}")

            #log_file.write("---\n")
        print("---")

    def save_summary(self):
        summary_folder = os.path.join(self.log_dir, "summary")
        os.makedirs(summary_folder, exist_ok=True)

        summary_path = os.path.join(
            summary_folder,
            f"{self.dataset_name}_{self.model_name}_results_summary.csv",
        )
        
        isFirst = True
        with open(summary_path, "w") as summary_file:
            for key, logger in self.loggers.items():
                # Log overall stats only the first time 
                if isFirst:
                    summary_file.write(f"Overall stats:\n")
                    Logger.print_overall_statistics(overall_results=self.overall_results, file=summary_file)
                    summary_file.write("\n---\n")
                    isFirst = False

                summary_file.write(f"Key: {key}\n")
                logger.print_statistics(file=summary_file)
                summary_file.write("\n---\n")
