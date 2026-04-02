import dgl
import torch
from biosnap_data_loader import BioSnapDDIDataset
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from evaluator import TDCDDIEvaluator


class DatasetHelper():
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
            dataset = DglLinkPropPredDataset(name=self.dataset_name, root=self.data_root_path)
            graph = dataset[0]
            evaluator = Evaluator(name=dataset.name)
        elif self.dataset_name in {"biosnapddi", "drugbankddi"}:
            dataset = BioSnapDDIDataset(name=self.dataset_name, path=self.data_root_path)
            graph = self.__convert_to_dgl_graph(dataset.get(0))
            evaluator = TDCDDIEvaluator(name="TDC_DDI")
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
        return dataset, graph.to(self.device), evaluator
    
    def get_train_test_val_edge_split(self, dataset):
        split_edge = dataset.get_edge_split()
        idx = torch.randperm(split_edge["train"]["edge"].size(0))
        idx = idx[: split_edge["valid"]["edge"].size(0)]
        split_edge["eval_train"] = {"edge": split_edge["train"]["edge"][idx]}
        return split_edge

