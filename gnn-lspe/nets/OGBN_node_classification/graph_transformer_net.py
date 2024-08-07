import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer with edge features
    
"""
from layers.graph_transformer_edge_layer_ogbn import GraphTransformerLayer # TODO: adapt to ogbn
from layers.mlp_readout_layer import MLPReadout

from egat import EGATConv
from operator import mul
from functools import reduce

class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        num_classes = net_params.get('num_classes', 112)
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        self.use_bias = net_params['use_bias']
        self.gat = net_params['gat']
        self.hidden_dim = hidden_dim
        self.O_linear = net_params['O_linear']
        max_wl_role_index = 37 # this is maximum graph size in the dataset
        explicit_bias = net_params['explicit_bias']

        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        embed_dim = (hidden_dim,)*2
        self.embedding_h = nn.Embedding(num_atom_type, embed_dim[0])

        if self.edge_feat:
            self.embedding_e = nn.Linear(num_bond_type, embed_dim[1])
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        if not self.gat:
            self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                        self.layer_norm, self.batch_norm, self.residual, use_bias=self.use_bias, explicit_bias=explicit_bias, edge_feat=self.edge_feat, O_linear=self.O_linear) for _ in range(n_layers-1) ]) 
            self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual, use_bias=self.use_bias, explicit_bias=explicit_bias, edge_feat=self.edge_feat, O_linear=self.O_linear))
        else:
            self.layers = nn.ModuleList([
                EGATConv(
                    in_node_feats=hidden_dim,
                    in_edge_feats=hidden_dim,
                    out_node_feats=output_dim//num_heads,
                    out_edge_feats=output_dim//num_heads,
                    num_heads=num_heads,
                    use_bias=self.use_bias,
                    explicit_bias=explicit_bias
                ) for output_dim in ([hidden_dim]*(n_layers-1)+[out_dim])
            ])
        self.MLP_layer = MLPReadout(out_dim, num_classes)   # 1 out dim since regression problem        
        
    def forward(self, blocks, h, p, e, h_lap_pos_enc=None, h_wl_pos_enc=None):

        # input embedding
        h = self.embedding_h(blocks[0].ndata['feat']['_N'])
        h = h.view(-1, self.hidden_dim)
        h = self.in_feat_dropout(h)
        
        # convnets
        for i in range(len(self.layers)):
            e = self.embedding_e(blocks[i].edata['feat'])
            h, e = self.layers[i](blocks[i], h, e)
        g = blocks[-1]
        
        hg = h
            
        hg = self.MLP_layer(hg)
        return hg, g
        
    def loss(self, scores, targets):
        loss = nn.BCEWithLogitsLoss()(scores, targets)
        return loss
    
    def malog(self, malog:bool):
        for layer in self.layers:
            layer.malog = malog
