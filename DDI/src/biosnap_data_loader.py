import dgl
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset


class BioSnapDDIDataset(Dataset):
    """
    Train and test CSVs must share one global drug-id space: DatasetHelper concatenates
    train and test edge_index for splits and trains on a single graph. Building separate
    per-file node indices (the old behavior) misaligned test edges with the wrong nodes.
    """

    def __init__(self, path, name="biosnapddi", transform=None, pre_transform=None):
        self.name = name
        self.transform = transform
        self.pre_transform = pre_transform
        self.path = path
        self.train_df = pd.read_csv(f"{path}/{name}/raw/train.csv")
        self.test_df = pd.read_csv(f"{path}/{name}/raw/test.csv")
        self.process()

    def _global_smiles_vocab(self):
        s = set(self.train_df["smile1"].tolist() + self.train_df["smile2"].tolist())
        s.update(self.test_df["smile1"].tolist())
        s.update(self.test_df["smile2"].tolist())
        return sorted(s)

    def process(self):
        global_smiles = self._global_smiles_vocab()
        self.smiles_to_index = {smile: idx for idx, smile in enumerate(global_smiles)}

        self.train_data = self._convert_to_pyg_data(self.train_df, self.smiles_to_index)
        self.test_data = self._convert_to_pyg_data(self.test_df, self.smiles_to_index)

    def _convert_to_pyg_data(self, df, smiles_to_index):
        smiles1 = df["smile1"].tolist()
        smiles2 = df["smile2"].tolist()
        labels = df["label"].tolist()

        n = len(smiles_to_index)
        node_features = torch.eye(n)

        edge_index = []
        edge_attr = []
        for smile1, smile2, label in zip(smiles1, smiles2, labels):
            idx1 = smiles_to_index[smile1]
            idx2 = smiles_to_index[smile2]
            edge_index.append([idx1, idx2])
            edge_attr.append(label)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

        data.smiles_mapping = smiles_to_index
        data.index_to_smiles = {idx: smile for smile, idx in smiles_to_index.items()}

        return data

    def get(self, idx):
        if idx == 0:
            return self.train_data
        elif idx == 1:
            return self.test_data
        else:
            raise IndexError("Index out of range")

    def len(self):
        return 2  # Train, Test

    def __len__(self):
        return self.len()

    def to_dgl_graph(self, pyg_data):
        n = pyg_data.x.shape[0]
        g = dgl.graph(
            (pyg_data.edge_index[0], pyg_data.edge_index[1]), num_nodes=n
        )
        g.ndata["feat"] = pyg_data.x
        g.edata["label"] = pyg_data.edge_attr

        # Store SMILES mappings in the DGL graph
        g.smiles_to_index = pyg_data.smiles_mapping  # {SMILES: node_id}
        g.index_to_smiles = pyg_data.index_to_smiles  # {node_id: SMILES}

        return g

    def to_dgl_graph_with_node_asgraph(self, pyg_data):
        # Convert PyTorch Geometric data to DGL graph
        n = pyg_data.x.shape[0]
        g = dgl.graph(
            (pyg_data.edge_index[0], pyg_data.edge_index[1]), num_nodes=n
        )
        g.ndata["feat"] = pyg_data.x

        # Create a new graph for each node
        node_graphs = []
        for i in range(g.num_nodes()):
            # Create a new graph with a single node
            node_g = dgl.graph(([], []), num_nodes=1)
            node_g.ndata["feat"] = g.ndata["feat"][i].view(1, -1)
            node_graphs.append(node_g)

        return g, node_graphs

    def get_edge_split(self, seed=42):
        # Convert PyTorch Geometric data to DGL graphs
        train_g = self.to_dgl_graph(self.train_data)
        test_g = self.to_dgl_graph(self.test_data)

        # Extract positive and negative edges from the training and test sets
        train_pos_edges = self.train_data.edge_index[
            :, self.train_data.edge_attr == 1
        ].t()
        train_neg_edges = self.train_data.edge_index[
            :, self.train_data.edge_attr == 0
        ].t()
        test_pos_edges = self.test_data.edge_index[:, self.test_data.edge_attr == 1].t()
        test_neg_edges = self.test_data.edge_index[:, self.test_data.edge_attr == 0].t()

        # Randomly sample 20% of the positive and negative edges for validation
        train_pos_edges, valid_pos_edges = train_test_split(
            train_pos_edges.numpy(), test_size=0.2, random_state=seed
        )
        train_neg_edges, valid_neg_edges = train_test_split(
            train_neg_edges.numpy(), test_size=0.2, random_state=seed
        )

        # Convert back to torch tensors
        train_pos_edges = torch.tensor(train_pos_edges, dtype=torch.long)
        valid_pos_edges = torch.tensor(valid_pos_edges, dtype=torch.long)
        train_neg_edges = torch.tensor(train_neg_edges, dtype=torch.long)
        valid_neg_edges = torch.tensor(valid_neg_edges, dtype=torch.long)

        # Create the split_edge dictionary
        split_edge = {
            "train": {"edge": train_pos_edges, "edge_neg": train_neg_edges},
            "valid": {"edge": valid_pos_edges, "edge_neg": valid_neg_edges},
            "test": {"edge": test_pos_edges, "edge_neg": test_neg_edges},
        }

        # Check class imbalance
        self._check_class_imbalance(split_edge)

        return split_edge

    def _check_class_imbalance(self, split_edge):
        # Calculate the number of positive and negative edges in each split
        train_pos_count = split_edge["train"]["edge"].size(0)
        train_neg_count = split_edge["train"]["edge_neg"].size(0)
        valid_pos_count = split_edge["valid"]["edge"].size(0)
        valid_neg_count = split_edge["valid"]["edge_neg"].size(0)
        test_pos_count = split_edge["test"]["edge"].size(0)
        test_neg_count = split_edge["test"]["edge_neg"].size(0)

        # Print the counts
        print(
            f"Training set - Positive edges: {train_pos_count}, Negative edges: {train_neg_count}"
        )
        print(
            f"Validation set - Positive edges: {valid_pos_count}, Negative edges: {valid_neg_count}"
        )
        print(
            f"Test set - Positive edges: {test_pos_count}, Negative edges: {test_neg_count}"
        )

        # Calculate the ratio of positive to negative edges
        train_ratio = (
            train_pos_count / train_neg_count if train_neg_count > 0 else float("inf")
        )
        valid_ratio = (
            valid_pos_count / valid_neg_count if valid_neg_count > 0 else float("inf")
        )
        test_ratio = (
            test_pos_count / test_neg_count if test_neg_count > 0 else float("inf")
        )

        print(f"Training set - Positive to Negative ratio: {train_ratio}")
        print(f"Validation set - Positive to Negative ratio: {valid_ratio}")
        print(f"Test set - Positive to Negative ratio: {test_ratio}")
