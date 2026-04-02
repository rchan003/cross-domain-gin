from typing import Union

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, Size


class ForwardPass(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault("aggr", "max")
        super(ForwardPass, self).__init__(**kwargs)

    def forward(
        self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        return out


class BackwardPass(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault("aggr", "max")
        super(BackwardPass, self).__init__(**kwargs)

    def forward(
        self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        out = self.propagate(edge_index, x=x, size=size)
        return out


class Evaluator:
    def __init__(self):
        self.forward_pass = ForwardPass(aggr="max", flow="source_to_target")
        self.backward_pass = BackwardPass(aggr="max", flow="target_to_source")

    def forward(self, edge_index, duration, n_j, n_m):
        """
        support batch version
        edge_index: [2, n_edges] tensor
        duration: [n_nodes, 1] tensor
        """
        n_nodes = duration.shape[0]
        n_nodes_each_graph = n_j * n_m + 2
        device = edge_index.device

        # forward pass...
        index_S = (
            np.arange(n_nodes // n_nodes_each_graph, dtype=int) * n_nodes_each_graph
        )
        earliest_start_time = torch.zeros_like(
            duration, dtype=torch.float32, device=device
        )
        mask_earliest_start_time = torch.ones_like(
            duration, dtype=torch.int8, device=device
        )
        mask_earliest_start_time[index_S] = 0
        for _ in range(n_nodes):
            if mask_earliest_start_time.sum() == 0:
                break
            x_forward = duration + earliest_start_time.masked_fill(
                mask_earliest_start_time.bool(), 0
            )
            earliest_start_time = self.forward_pass(x=x_forward, edge_index=edge_index)
            mask_earliest_start_time = self.forward_pass(
                x=mask_earliest_start_time, edge_index=edge_index
            )

        # backward pass...
        index_T = (
            np.cumsum(
                np.ones(shape=[n_nodes // n_nodes_each_graph], dtype=int)
                * n_nodes_each_graph
            )
            - 1
        )
        make_span = earliest_start_time[index_T]
        # latest_start_time = torch.zeros_like(duration, dtype=torch.float32, device=device)
        latest_start_time = -torch.ones_like(
            duration, dtype=torch.float32, device=device
        )
        latest_start_time[index_T] = -make_span
        mask_latest_start_time = torch.ones_like(
            duration, dtype=torch.int8, device=device
        )
        mask_latest_start_time[index_T] = 0
        for _ in range(n_nodes):
            if mask_latest_start_time.sum() == 0:
                break
            x_backward = latest_start_time.masked_fill(mask_latest_start_time.bool(), 0)
            latest_start_time = (
                self.backward_pass(x=x_backward, edge_index=edge_index) + duration
            )
            latest_start_time[index_T] = -make_span
            mask_latest_start_time = self.backward_pass(
                x=mask_latest_start_time, edge_index=edge_index
            )

        return earliest_start_time, torch.abs(latest_start_time), make_span


def __forward_pass(graph, topological_order=None):  # graph is a nx.DiGraph;
    # assert (graph.in_degree(topological_order[0]) == 0)
    earliest_ST = dict.fromkeys(graph.nodes, -float("inf"))
    if topological_order is None:
        topo_order = list(nx.topological_sort(graph))
    else:
        topo_order = topological_order
    earliest_ST[topo_order[0]] = 0.0
    for n in topo_order:
        for s in graph.successors(n):
            if earliest_ST[s] < earliest_ST[n] + graph.edges[n, s]["weight"]:
                earliest_ST[s] = earliest_ST[n] + graph.edges[n, s]["weight"]
    # return is a dict where key is each node's ID, value is the length from source node s
    return earliest_ST


def __backward_pass(graph, makespan, topological_order=None):
    if topological_order is None:
        reverse_order = list(reversed(list(nx.topological_sort(graph))))
    else:
        reverse_order = list(reversed(topological_order))
    latest_ST = dict.fromkeys(graph.nodes, float("inf"))
    latest_ST[reverse_order[0]] = float(makespan)
    for n in reverse_order:
        for p in graph.predecessors(n):
            if latest_ST[p] > latest_ST[n] - graph.edges[p, n]["weight"]:
                # assert latest_ST[n] - graph.edges[p, n]['weight'] >= 0, 'latest start times should is negative, BUG!'  # latest start times should be non-negative
                latest_ST[p] = latest_ST[n] - graph.edges[p, n]["weight"]
    return latest_ST


def __forward_and_backward_pass(G):
    # calculate topological order
    topological_order = list(nx.topological_sort(G))
    # forward and backward pass
    est = np.fromiter(
        __forward_pass(graph=G, topological_order=topological_order).values(),
        dtype=np.float32,
    )
    lst = np.fromiter(
        __backward_pass(
            graph=G, topological_order=topological_order, makespan=est[-1]
        ).values(),
        dtype=np.float32,
    )
    # assert np.where(est > lst)[0].shape[0] == 0, 'latest starting time is smaller than earliest starting time, bug!'  # latest starting time should be larger or equal to earliest starting time
    return est, lst, est[-1]


def CPM_batch_G(Gs, dev):
    multi_est = []
    multi_lst = []
    multi_makespan = []
    for G in Gs:
        est, lst, makespan = __forward_and_backward_pass(G)
        multi_est.append(est)
        multi_lst.append(lst)
        multi_makespan.append([makespan])
    multi_est = torch.from_numpy(np.concatenate(multi_est, axis=0)).view(-1, 1).to(dev)
    multi_lst = torch.from_numpy(np.concatenate(multi_lst, axis=0)).view(-1, 1).to(dev)
    multi_makespan = torch.tensor(multi_makespan, device=dev)
    return multi_est, multi_lst, multi_makespan
