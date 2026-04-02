import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GINConv, TAGConv
from torch.nn import Linear


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = F.relu(lin(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x.squeeze(-1) # torch.sigmoid(x) TODO: Changed to give raw score, used to apply sigmoid 


class DGLMPNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DGLMPNNLayer, self).__init__()
        self.message_func = fn.copy_u("h", "m")
        self.reduce_func = fn.sum("m", "h")
        self.fc = nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            graph.ndata["h"] = feat
            graph.update_all(self.message_func, self.reduce_func)
            x = graph.ndata["h"]
            x = self.fc(x)
            return x


class DGLGINEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, layer_gin=4):
        super(DGLGINEncoder, self).__init__()
        self.layer_gin = layer_gin
        self.GIN_layers = nn.ModuleList()

        if layer_gin < 1:
            raise ValueError("layer_gin must be >= 1")

        # First GIN layer: in_dim -> hidden_dim
        first_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.GIN_layers.append(
            GINConv(
                apply_func=first_mlp,
                aggregator_type="mean",
                init_eps=0.0,
                learn_eps=False,
            )
        )

        # Remaining GIN layers: hidden_dim -> hidden_dim
        for _ in range(layer_gin - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.GIN_layers.append(
                GINConv(
                    apply_func=mlp,
                    aggregator_type="mean",
                    init_eps=0.0,
                    learn_eps=False,
                )
            )

        self.reset_parameters()

    def reset_parameters(self):
        for gin in self.GIN_layers:
            mlp = gin.apply_func
            for layer in mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm1d):
                    layer.reset_parameters()

            if hasattr(gin, "reset_parameters"):
                gin.reset_parameters()

    def forward(self, graph, x):
        hidden_rep = []
        node_pool_over_layer = 0

        # First layer
        h = self.GIN_layers[0](graph, x)
        node_pool_over_layer = node_pool_over_layer + h
        hidden_rep.append(h)

        # Remaining layers
        for layer in range(1, self.layer_gin):
            h = self.GIN_layers[layer](graph, h)
            node_pool_over_layer = node_pool_over_layer + h
            hidden_rep.append(h)

        return node_pool_over_layer, hidden_rep

class DGLGraphModel(nn.Module):
    def __init__(
        self,
        input_features,
        hidden_features,
        output_features,
        num_layers,
        use_gin=False,
    ):
        super(DGLGraphModel, self).__init__()
        self.use_gin = use_gin
        self.activation = nn.ReLU()

        if self.use_gin:
            self.gin_encoder = DGLGINEncoder(
                in_dim=input_features,
                hidden_dim=hidden_features,
                layer_gin=num_layers,
            )
            self.tagConv = TAGConv(hidden_features, hidden_features)
        else:
            self.tagConv = TAGConv(input_features, hidden_features)

        self.mpnn_layer = DGLMPNNLayer(hidden_features, output_features)
        self.fc_out = nn.Linear(output_features, output_features)

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_gin:
            self.gin_encoder.reset_parameters()

        self.tagConv.reset_parameters()
        self.mpnn_layer.reset_parameters()

        nn.init.xavier_normal_(self.fc_out.weight)
        if self.fc_out.bias is not None:
            nn.init.zeros_(self.fc_out.bias)

    def forward(self, graph, feat, edge_weight=None):
        x = feat

        if self.use_gin:
            x, hidden_rep = self.gin_encoder(graph, x)

        x = self.tagConv(graph, x)
        x = self.activation(x)

        x = self.mpnn_layer(graph, x, edge_weight)
        x = self.fc_out(x)
        return x