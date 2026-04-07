import dgl
import torch
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator

from .biosnap_data_loader import BioSnapDDIDataset
from .evaluator import TDCDDIEvaluator


class DatasetHelper:
    def __init__(self, args, device: str, data_root_path: str):
        self.dataset_name = args.dataset
        self.device = device
        self.data_root_path = data_root_path

    def __convert_to_dgl_graph(self, pyg_data):
        graph = dgl.graph((pyg_data.edge_index[0], pyg_data.edge_index[1]))
        if hasattr(pyg_data, "x") and pyg_data.x is not None:
            graph.ndata["feat"] = pyg_data.x
        if hasattr(pyg_data, "edge_attr") and pyg_data.edge_attr is not None:
            graph.edata["label"] = pyg_data.edge_attr
        return graph

    def load_dataset(self):
        if self.dataset_name == "ogbl-ddi":
            dataset = DglLinkPropPredDataset(
                name=self.dataset_name, root=self.data_root_path
            )
            graph = dataset[0]
            evaluator = Evaluator(name=dataset.name)

        elif self.dataset_name in {"biosnapddi", "drugbankddi"}:
            dataset = BioSnapDDIDataset(
                name=self.dataset_name, path=self.data_root_path
            )
            graph = self.__convert_to_dgl_graph(dataset.get(0))
            evaluator = TDCDDIEvaluator(name="TDC_DDI")

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        return dataset, graph.to(self.device), evaluator

    def _split_edges(self, edges, seed, train_ratio=0.8, valid_ratio=0.1):
        g = torch.Generator()
        g.manual_seed(seed)

        perm = torch.randperm(edges.size(0), generator=g)
        edges = edges[perm]

        n = edges.size(0)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)

        train_edges = edges[:n_train]
        valid_edges = edges[n_train : n_train + n_valid]
        test_edges = edges[n_train + n_valid :]

        return train_edges, valid_edges, test_edges

    def _random_negative_edges(self, num_nodes, num_samples, positive_edges, seed):
        g = torch.Generator()
        g.manual_seed(seed)

        pos_set = set((u.item(), v.item()) for u, v in positive_edges)
        neg_edges = []

        while len(neg_edges) < num_samples:
            src = torch.randint(0, num_nodes, (1,), generator=g).item()
            dst = torch.randint(0, num_nodes, (1,), generator=g).item()

            if src == dst:
                continue
            if (src, dst) in pos_set:
                continue

            neg_edges.append((src, dst))

        return torch.tensor(neg_edges, dtype=torch.long)

    def get_train_test_val_edge_split(self, dataset, seed):
        if self.dataset_name in {"biosnapddi", "drugbankddi"}:
            # These datasets already store labels in edge_attr
            train_data = dataset.get(0)
            test_data = dataset.get(1)

            all_edge_index = torch.cat(
                [train_data.edge_index, test_data.edge_index], dim=1
            )
            all_edge_attr = torch.cat(
                [train_data.edge_attr, test_data.edge_attr], dim=0
            )

            pos_edges = all_edge_index[:, all_edge_attr == 1].t().contiguous()
            neg_edges = all_edge_index[:, all_edge_attr == 0].t().contiguous()

            train_pos, valid_pos, test_pos = self._split_edges(pos_edges, seed=seed)
            train_neg, valid_neg, test_neg = self._split_edges(neg_edges, seed=seed + 1)

            split_edge = {
                "train": {"edge": train_pos, "edge_neg": train_neg},
                "valid": {"edge": valid_pos, "edge_neg": valid_neg},
                "test": {"edge": test_pos, "edge_neg": test_neg},
            }

        elif self.dataset_name == "ogbl-ddi":
            # Custom randomized split from all positive graph edges
            # Warning: this ignores the official OGB fixed split
            pos_edges = dataset[0].edges()
            pos_edges = torch.stack(pos_edges, dim=1).contiguous()

            train_pos, valid_pos, test_pos = self._split_edges(pos_edges, seed=seed)

            num_nodes = dataset[0].num_nodes()
            train_neg = self._random_negative_edges(
                num_nodes, train_pos.size(0), pos_edges, seed=seed + 1
            )
            valid_neg = self._random_negative_edges(
                num_nodes, valid_pos.size(0), pos_edges, seed=seed + 2
            )
            test_neg = self._random_negative_edges(
                num_nodes, test_pos.size(0), pos_edges, seed=seed + 3
            )

            split_edge = {
                "train": {"edge": train_pos, "edge_neg": train_neg},
                "valid": {"edge": valid_pos, "edge_neg": valid_neg},
                "test": {"edge": test_pos, "edge_neg": test_neg},
            }

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        # Split based on seed
        g = torch.Generator()
        g.manual_seed(seed + 999)
        idx = torch.randperm(split_edge["train"]["edge"].size(0), generator=g)
        idx = idx[: split_edge["valid"]["edge"].size(0)]
        split_edge["eval_train"] = {"edge": split_edge["train"]["edge"][idx]}

        return split_edge
