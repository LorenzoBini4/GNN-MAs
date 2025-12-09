import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import torch.nn.functional as F

class GATLayer(MessagePassing):
    """
    Custom GAT layer that exposes the pre-softmax attention logits for MA logging.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        residual: bool = True,
        concat: bool = True,
    ):
        super().__init__(node_dim=0, aggr="add")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.concat = concat
        self.residual = residual

        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim if not concat else heads * out_dim))

        if residual:
            res_dim = heads * out_dim if concat else out_dim
            self.res_proj = None if in_dim == res_dim else nn.Linear(in_dim, res_dim, bias=False)
        else:
            self.res_proj = None

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

        self.malog = False
        self.malog_h = []
        self.malog_e = []

    def reset_parameters(self):
        nn.init.xavier_normal_(self.lin.weight)
        nn.init.xavier_normal_(self.att_src)
        nn.init.xavier_normal_(self.att_dst)
        if self.res_proj is not None:
            nn.init.xavier_normal_(self.res_proj.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        H, D = self.heads, self.out_dim
        x_proj = self.lin(x).view(-1, H, D)
        alpha_src = (x_proj * self.att_src).sum(-1)
        alpha_dst = (x_proj * self.att_dst).sum(-1)

        alpha_logits = self.leaky_relu(alpha_src[edge_index[0]] + alpha_dst[edge_index[1]])
        if self.malog:
            self.malog_e.append({"attention": alpha_logits.detach().cpu()})

        alpha = softmax(alpha_logits, edge_index[1], num_nodes=x_proj.size(0))
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)

        out = self.propagate(edge_index, x=x_proj, alpha=alpha)
        if self.concat:
            out = out.view(-1, H * D)
        else:
            out = out.mean(dim=1)
        out = out + self.bias
        out = F.dropout(out, p=self.dropout, training=self.training)

        if self.residual:
            res = x if self.res_proj is None else self.res_proj(x)
            out = out + res

        if self.malog:
            self.malog_h.append({"attention": out.detach().cpu()})
        return out

    def message(self, x_j: torch.Tensor, alpha: torch.Tensor):
        return x_j * alpha.unsqueeze(-1)
