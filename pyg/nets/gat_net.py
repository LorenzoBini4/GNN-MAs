import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GATLayer

class GATNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params["in_dim"]
        hidden_dim = net_params["hidden_dim"]
        n_classes = net_params["n_classes"]
        n_layers = net_params["n_layers"]
        n_heads = net_params["n_heads"]
        dropout = net_params["dropout"]
        attn_dropout = net_params.get("attn_dropout", dropout)
        residual = net_params.get("residual", True)

        layers = []
        # input layer
        layers.append(
            GATLayer(
                in_dim=in_dim,
                out_dim=hidden_dim,
                heads=n_heads,
                dropout=dropout,
                attn_dropout=attn_dropout,
                residual=residual,
                concat=True,
            )
        )
        # hidden layers
        for _ in range(n_layers - 2):
            layers.append(
                GATLayer(
                    in_dim=hidden_dim * n_heads,
                    out_dim=hidden_dim,
                    heads=n_heads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    residual=residual,
                    concat=True,
                )
            )
        # output layer: single head, do not concat
        layers.append(
            GATLayer(
                in_dim=hidden_dim * n_heads,
                out_dim=n_classes,
                heads=1,
                dropout=dropout,
                attn_dropout=attn_dropout,
                residual=False,
                concat=False,
            )
        )
        self.layers = nn.ModuleList(layers)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

    def loss(self, pred, label, mask):
        return F.cross_entropy(pred[mask], label[mask])

    def malog(self, enable: bool):
        for layer in self.layers:
            layer.malog = enable
            layer.malog_h = []
            layer.malog_e = []
