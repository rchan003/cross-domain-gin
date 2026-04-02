import torch


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)
    
    def print_statistics(self, run=None, file=None):
        def write_line(s):
            if file:
                file.write(s + "\n")
            else:
                print(s)

        if run is not None:
            result = torch.tensor(self.results[run], dtype=torch.float32)
            argmax = result[:, 1].argmax().item()

            write_line(f"Run {run + 1:02d}:")
            write_line(f"Highest Train: {result[:, 0].max():.2f}")
            write_line(f"Highest Valid: {result[:, 1].max():.2f}")
            write_line(f"  Final Train: {result[argmax, 0]:.2f}")
            write_line(f"   Final Test: {result[argmax, 2]:.2f}")

        else:
            result = torch.tensor(self.results, dtype=torch.float32)
            
            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results, dtype=torch.float32)

            write_line("All runs:")
            r = best_result[:, 0]
            write_line(f"Highest Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 1]
            write_line(f"Highest Valid: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 2]
            write_line(f"  Final Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 3]
            write_line(f"   Final Test: {r.mean():.2f} ± {r.std():.2f}")

    @staticmethod
    def print_overall_statistics(overall_results, file=None, run=None):
        def write_line(s):
            if file:
                file.write(s + "\n")
            else:
                print(s)

        if run is not None:
            result = torch.tensor(overall_results[run], dtype=torch.float32)
            # choose best epoch by AUC, for example
            argmax = result[:, 1].argmax().item()

            write_line(f"Run {run + 1:02d}:")
            write_line(f"Best Accuracy: {result[:, 0].max():.4f}")
            write_line(f"Best AUC: {result[:, 1].max():.4f}")
            write_line(f"Best AP: {result[:, 2].max():.4f}")
            write_line(f" Best F1: {result[:, 3].max():.4f}")
            write_line(f"Best Recall: {result[:, 4].max():.4f}")
            write_line(f"Final Overall (best AUC epoch): {result[argmax].tolist()}")

        else:
            result = torch.tensor(overall_results, dtype=torch.float32)

            best_results = []
            for r in result:
                argmax = r[:, 1].argmax().item()  # best epoch by AUC
                best_results.append(r[argmax].tolist())

            best_result = torch.tensor(best_results, dtype=torch.float32)

            names = ["Accuracy", "AUC", "AP", "F1", "Recall"]
            write_line("All runs:")
            for i, name in enumerate(names):
                col = best_result[:, i]
                write_line(f"Highest {name:>8}: {col.mean():.4f} ± {col.std():.4f}")