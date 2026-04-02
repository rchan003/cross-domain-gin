import argparse
import gc
import os
import random
import time

import torch
import torch.nn.functional as F
from dgl.dataloading.negative_sampler import GlobalUniform
from src.dataset_helper import DatasetHelper
from src.log_helper import LogHelper
from src.models import DGLGraphModel, LinkPredictor
from src.pretrain_helper import PretrainHelper
from src.seed_setting_for_reproducibility import seed_torch
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryRecall,
)

# Run using:
# cd /mnt/4tb/rachel_thesis/DDI
# python3 src/main_rachel_gin.py --dataset "biosnapddi" --pretrained_path "/mnt/4tb/rachel_thesis/L2S-main/saved_rachel_model/20x15[1,99]_fdd-divide-mwkr_yaoxin_1.0_256_4_4_gin_NAN_0.0001_10_500_64_640_1/checkpoints/best_gin_incumbent.pth" --init_mode "pretrained"

# Profile:
# python3 -m cProfile -s cumulative src/main_rachel_gin.py --dataset "ogbl-ddi" --pretrained_path "/mnt/4tb/rachel_thesis/L2S-main/saved_rachel_model/20x15[1,99]_fdd-divide-mwkr_yaoxin_1.0_256_4_4_gin_NAN_0.0001_10_500_64_640_1/checkpoints/best_gin_incumbent.pth" --init_mode "pretrained" --epochs 5 > profile_ogbl.txt

# python3 src/main_rachel_gin.py --dataset "ogbl-ddi" --pretrained_path "/mnt/4tb/rachel_thesis/L2S-main/saved_rachel_model/20x15[1,99]_fdd-divide-mwkr_yaoxin_1.0_256_4_4_gin_NAN_0.0001_10_500_64_640_1/checkpoints/best_gin_incumbent.pth" --init_mode "pretrained" --epochs 5 --batch_size 10000 --lr 1e-7

# 35 seconds for bs = 5000
# --- Final epoch 
# Run: 02, Epoch: 05, Loss: 0.0972 || TEST RESULTS: Accuracy: 85.5130, AUC: 99.5320, AP: 99.6003, F1: 85.8453, Recall: 99.6003
#     Hits@1  Train: 1.11%  Valid: 0.70%  Test: 1.99%
#     Hits@3  Train: 10.84%  Valid: 8.25%  Test: 2.12%
#    Hits@10  Train: 25.02%  Valid: 20.22%  Test: 6.85%
#    Hits@20  Train: 39.52%  Valid: 32.55%  Test: 13.48%
#    Hits@30  Train: 47.38%  Valid: 39.55%  Test: 19.07%
#    Hits@40  Train: 53.98%  Valid: 45.61%  Test: 24.41%
#    Hits@50  Train: 56.74%  Valid: 48.06%  Test: 28.10%

# 149 seconds for bs = 1024
# ---
# Run: 02, Epoch: 05, Loss: 0.0834 || TEST RESULTS: Accuracy: 90.2496, AUC: 99.7663, AP: 99.7861, F1: 90.8842, Recall: 99.7861
#     Hits@1  Train: 0.75%  Valid: 0.32%  Test: 1.13%
#     Hits@3  Train: 5.59%  Valid: 3.62%  Test: 2.93%
#    Hits@10  Train: 23.55%  Valid: 17.54%  Test: 10.95%
#    Hits@20  Train: 46.63%  Valid: 37.41%  Test: 15.49%
#    Hits@30  Train: 66.16%  Valid: 55.70%  Test: 20.16%
#    Hits@40  Train: 69.53%  Valid: 58.98%  Test: 24.72%
#    Hits@50  Train: 72.79%  Valid: 62.38%  Test: 31.96%
# ---
# End time: 1774495359.8699143, device = cuda:0
# Duration: 149.18063068389893, device = cuda:0


DATA_ROOT_PATH = "/mnt/4tb/rachel_thesis/DDI/dataset"

class ModelRunner():
    def __init__(self, args, model, predictor, evaluator, graph, split_edge, device, pretrained_state=None):
        # Config
        self.batch_size = args.batch_size
        self.dataset_name = args.dataset
        self.learning_rate = args.lr
        self.hidden_channels = args.hidden_channels
        self.device = device
        self.init_mode = args.init_mode
        self.num_neg_per_pos = args.num_neg_per_pos
        self.loss_function = args.loss_function

        # Objects
        self.model = model
        self.predictor = predictor
        self.evaluator = evaluator
        self.graph = graph     # graph object (who is connected to who)
        self.split_edge = split_edge
        self.pretrained_state = pretrained_state
        
        # Set when reset_and_initialize_runner is called
        self.optimizer = None
        self.x = None  # node features (what each node contains)

        # Optimization 1: Initialize GPU Metrics
        self.rocauc_m = BinaryAUROC().to(device)
        self.ap_m = BinaryAveragePrecision().to(device)
        self.acc_m = BinaryAccuracy().to(device)
        self.f1_m = BinaryF1Score().to(device)
        self.recall_m = BinaryRecall().to(device)
    
    def reset_and_initialize_runner(self):
        '''
        Must be called before any new run. 
        '''
        # Reset model & predictor 
        self.model.reset_parameters()
        self.predictor.reset_parameters()

        # Reload pretrained GIN weights after reset
        if self.pretrained_state is not None and self.init_mode == "pretrained":
            PretrainHelper.load_matching_weights(
                self.model.gin_encoder,
                self.pretrained_state,
                verbose=True,
            )

        # Create learnable embedding weights for ogbl-ddi dataset
        if self.dataset_name == "ogbl-ddi":
            emb = self.__create_embedder()
            torch.nn.init.xavier_uniform_(emb.weight)
            self.graph.ndata["feat"] = emb.weight

        # Create optimizer
        optimizer_params = list(self.model.parameters()) + list(self.predictor.parameters())
        if self.dataset_name == "ogbl-ddi":
            optimizer_params += list(emb.parameters())

        self.optimizer = torch.optim.Adam(optimizer_params, lr=self.learning_rate)

        # Set the node features 
        self.x = self.graph.ndata["feat"]
    
    def train(self):
        self.model.train()
        self.predictor.train()

        pos_train_edge = self.split_edge["train"]["edge"].to(self.device)
        neg_sampler = GlobalUniform(self.num_neg_per_pos)

        total_loss = 0.0
        total_examples = 0

        for perm in DataLoader(range(pos_train_edge.size(0)), self.batch_size, shuffle=True, drop_last=True):
            self.optimizer.zero_grad()

            h = self.model(self.graph, self.x)

            pos_edge = pos_train_edge[perm].t()   # shape: [2, B]
            pos_score = self.predictor(h[pos_edge[0]], h[pos_edge[1]])   # shape: [B]

            neg_edge = neg_sampler(self.graph, pos_edge[0])  # expected shape: [2, B * num_neg_per_pos]
            neg_score = self.predictor(h[neg_edge[0]], h[neg_edge[1]])   # shape: [B * num_neg_per_pos]

            batch_size = pos_edge.size(1)
            neg_score = neg_score.view(batch_size, self.num_neg_per_pos)       # shape: [B, K]

            if self.loss_function == "pairwise_ranking":
                # Ranking needs the negative scores to be [Batch, Num_Negatives]
                batch_size = pos_score.size(0)
                neg_score = neg_score.view(batch_size, self.num_neg_per_pos) 
                loss = -torch.log(torch.sigmoid(pos_score.unsqueeze(1) - neg_score) + 1e-15).mean()
            else: 
                # Binary Cross Entropy logic
                # 1. Flatten both to 1D vectors
                pos_preds = pos_score.view(-1)
                neg_preds = neg_score.view(-1)
                
                # 2. Combine into one long prediction vector
                predictions = torch.cat([pos_preds, neg_preds], dim=0)

                # 3. Create matching labels (1s for positive, 0s for negative)
                pos_labels = torch.ones(pos_preds.size(0), device=self.device)
                neg_labels = torch.zeros(neg_preds.size(0), device=self.device)
                labels = torch.cat([pos_labels, neg_labels], dim=0)

                # 4. Use the stable BCE loss
                loss = F.binary_cross_entropy_with_logits(predictions, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)

            if self.dataset_name == "ogbl-ddi":
                torch.nn.utils.clip_grad_norm_(self.x, 1.0)

            self.optimizer.step()

            
            num_examples = pos_score.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

        return total_loss / total_examples

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.predictor.eval()

        h = self.model(self.graph, self.x)

        preds = {
            "train": self.__get_batch_predictions_gpu(h, self.split_edge["eval_train"]["edge"]),
            "val_pos": self.__get_batch_predictions_gpu(h, self.split_edge["valid"]["edge"]),
            "val_neg": self.__get_batch_predictions_gpu(h, self.split_edge["valid"]["edge_neg"]),
            "test_pos": self.__get_batch_predictions_gpu(h, self.split_edge["test"]["edge"]),
            "test_neg": self.__get_batch_predictions_gpu(h, self.split_edge["test"]["edge_neg"])
        }

        # Optimization 6: GPU-native Metric Calculation
        test_all_logits = torch.cat([preds["test_pos"], preds["test_neg"]])
        test_all_preds = torch.sigmoid(test_all_logits)
        test_all_labels = torch.cat([
            torch.ones_like(preds["test_pos"]), 
            torch.zeros_like(preds["test_neg"])
        ]).long()

        results = {
            "Overall": (
                self.acc_m(test_all_preds, test_all_labels).item(),
                self.rocauc_m(test_all_preds, test_all_labels).item(),
                self.ap_m(test_all_preds, test_all_labels).item(),
                self.f1_m(test_all_preds, test_all_labels).item(),
                self.recall_m(test_all_preds, test_all_labels).item() 
            )
        }

        if self.dataset_name != "ogbl-ddi":
            train_hits = self.evaluator.eval({"y_pred_pos": preds["train"], "y_pred_neg": preds["val_neg"]})
            valid_hits = self.evaluator.eval({"y_pred_pos": preds["val_pos"], "y_pred_neg": preds["val_neg"]})
            test_hits = self.evaluator.eval({"y_pred_pos": preds["test_pos"], "y_pred_neg": preds["test_neg"]})

        for k in [1, 3, 10, 20, 30, 40, 50]:
            if self.dataset_name == "ogbl-ddi":
                self.evaluator.K = k #TODO: this doesn't appear in TDC check if this is a value of the obgi evaluator
                train_hits = self.evaluator.eval({"y_pred_pos": preds["train"], "y_pred_neg": preds["val_neg"]})
                valid_hits = self.evaluator.eval({"y_pred_pos": preds["val_pos"], "y_pred_neg": preds["val_neg"]})
                test_hits = self.evaluator.eval({"y_pred_pos": preds["test_pos"], "y_pred_neg": preds["test_neg"]})

            results[f"Hits@{k}"] = (
                train_hits[f"hits@{k}"], 
                valid_hits[f"hits@{k}"], 
                test_hits[f"hits@{k}"], 
            )
        
        # Clear GPU memory after evaluation
        self.acc_m.reset()
        self.rocauc_m.reset()
        self.ap_m.reset()
        self.f1_m.reset()
        self.recall_m.reset()

        return results
    
    def __create_embedder(self):
        ''' 
        Creates trainable embedding vector for each node during training.
        Used for ogbl-ddi dataset since the nodes don't have features. 
        '''
        emb = torch.nn.Embedding(self.graph.num_nodes(), self.hidden_channels).to(self.device)
        return emb

    def __safe_get_edge(self, edge):
        return edge[0] if isinstance(edge, (tuple, list)) else edge
    
    def __get_batch_predictions_gpu(self, h, edges):
        # Optimization 7: Avoid .cpu() calls and DataLoader overhead for evaluation
        edges = self.__safe_get_edge(edges).to(self.device)
        preds = []
        for i in range(0, edges.size(0), self.batch_size):
            batch_edges = edges[i:i + self.batch_size].t()
            pred = self.predictor(h[batch_edges[0]], h[batch_edges[1]]).squeeze()
            preds.append(pred)
        return torch.cat(preds)


def get_in_channels(args, graph):
    if (args.dataset == "ogbl-ddi") or ("feat" not in graph.ndata):
        in_channels = args.hidden_channels
    else:
        in_channels = graph.ndata["feat"].size(-1) 
    return in_channels

def create_model_and_predictor(device, args, graph):
    in_channels = get_in_channels(args, graph)

    # Optional: inspect pretrained checkpoint first
    if args.pretrained_path is not None:
        pretrained_state = PretrainHelper.load_pretrained_state(
            args.pretrained_path,
            map_location=device,
        )
        arch = PretrainHelper.infer_gin_architecture(pretrained_state)

        print("\n=== Inferred pretrained GIN architecture ===")
        print(f"num_layers: {arch['num_layers']}")
        print(f"pretrained_input_dim: {arch['pretrained_input_dim']}")
        print(f"hidden_dim: {arch['hidden_dim']}")
        print(f"has_batchnorm: {arch['has_batchnorm']}")

        # Use pretrained hidden_dim / num_layers if you want to match the checkpoint
        model_hidden_channels = arch["hidden_dim"]
        model_num_layers = arch["num_layers"]
    else:
        pretrained_state = None
        model_hidden_channels = args.hidden_channels
        model_num_layers = args.num_layers

    model = DGLGraphModel(
        input_features=in_channels,
        hidden_features=model_hidden_channels,
        output_features=model_hidden_channels,
        num_layers=model_num_layers,
        use_gin=True,
    )

    predictor = LinkPredictor(
        in_channels=model_hidden_channels,
        hidden_channels=model_hidden_channels,
        out_channels=1,
        num_layers=3,
        dropout=args.dropout,
    )

    return model.to(device), predictor.to(device), pretrained_state, arch, in_channels

def build_experiment_name(args, arch, in_channels):
    dataset_part = args.dataset

    arch_part = (
        f"ginL{arch['num_layers']}"
        f"_H{arch['hidden_dim']}"
        f"_in{in_channels}"
    )

    train_part = (
        f"lr{args.lr}"
        f"_bs{args.batch_size}"
        f"_ep{args.epochs}"
        f"_runs{args.runs}"
    )

    if args.pretrained_path is not None:
        ckpt_name = os.path.splitext(os.path.basename(args.pretrained_path))[0]
        if args.init_mode == "pretrained":
            mode_part = f"pretrained-{ckpt_name}"
        else:
            mode_part = f"control-randominit_from-{ckpt_name}"
    else:
        mode_part = "randominit_no-pretrain"

    return f"{dataset_part}__{arch_part}__{train_part}", mode_part

def main(args):
    device = torch.device(
        f"cuda:{args.device}" if args.device != -1 and torch.cuda.is_available() else "cpu"
    )

    start_time = time.time()
    print(f"Start time: {start_time}, device = {device}")

    print(f"\n=== Using device: {device} ===")

    print(f"\n=== Processing dataset: {args.dataset} ===")
    # Create dataset helper
    dataset_helper = DatasetHelper(args=args, device=device, data_root_path=DATA_ROOT_PATH)

    # Get dataset graph and evaluator
    dataset, graph, evaluator = dataset_helper.load_dataset()

    # Split dataset for train test and validation 
    split_edge = dataset_helper.get_train_test_val_edge_split(dataset)

    print(f"\n=== Running GIN model on {args.dataset} ===")
    model, predictor, pretrained_state, arch, in_channels = create_model_and_predictor(device, args, graph)

    experiment_name, mode_part = build_experiment_name(args, arch, in_channels)

    # Create model runner 
    model_runner = ModelRunner(
        args=args,
        model=model, 
        predictor=predictor, 
        evaluator=evaluator,
        graph=graph, 
        split_edge=split_edge,
        device=device,
        pretrained_state=pretrained_state)

    #return

    # Create log helper
    log_helper = LogHelper(
        args=args,
        model_name="GIN",
        result_dir=args.results_dir,
        experiment_name=experiment_name,
        mode_part=mode_part,
        model=model, 
        arch=arch,
        in_channels=in_channels,
    )

    # Don't run experiment if already completed before
    if log_helper.skip:
        return 

    for run in range(args.runs):
        random_seed = random.randint(1, 2**6 - 1)
        seed_torch(random_seed)

        model_runner.reset_and_initialize_runner()

        for epoch in range(1, args.epochs + 1):
            loss = model_runner.train()

            if epoch % args.eval_steps == 0:
                results = model_runner.test()
                if args.save_results:
                    log_helper.log_results(results=results, run=run, epoch=epoch, loss=loss)
        
        # Clear GPU memory after each run to prevent potential OOM issues in subsequent runs
        torch.cuda.empty_cache()
        gc.collect()

    if args.save_results:
        log_helper.save_summary()
        log_helper.mark_finished()
    
    end_time = time.time()
    print(f"End time: {end_time}, device = {device}")
    print(f"Duration: {end_time - start_time}, device = {device}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OGBL / DrugBank / BioSnap DDI link prediction"
    )

    # LOG SETTINGS
    parser.add_argument("--save_results", type=bool, default=True)
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/mnt/4tb/rachel_thesis/DDI/src/rachel/experiment_results/tmp"
    )
    
    # RUN SETTINGS  
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--runs", type=int, default=2) # TODO: used to be 5
    parser.add_argument(
        "--dataset",
        type=str,
        default="biosnapddi",
        choices=["ogbl-ddi", "drugbankddi", "biosnapddi"],
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0, # TODO: upddated here
        help="GPU device ID. Use -1 for CPU.",
    )

    # MODEL ARCHITECTURE
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.01)

    # TRAINING PARAMETERS
    parser.add_argument(
        "--loss_function",
        type=str, 
        default="binary_cross_entropy",
        choices=["pairwise_ranking", "binary_cross_entropy"]
    )
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=100) 
    parser.add_argument(
        "--init_mode",
        type=str,
        default="random",
        choices=["random", "pretrained"],
    )
    parser.add_argument("--num_neg_per_pos", type=int, default=1)

    # TEST SETTINGS
    parser.add_argument("--eval_steps", type=int, default=1)

    args = parser.parse_args()
    main(args)