import pandas as pd
import torch
from torch_geometric.data import Data, Dataset


class DDIDataset(Dataset):
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
