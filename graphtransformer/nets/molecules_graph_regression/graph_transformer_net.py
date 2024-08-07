import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer with edge features
    
"""
from layers.graph_transformer_edge_layer import GraphTransformerLayer
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
        num_classes = net_params.get('num_classes', 1)
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        self.use_bias = net_params['use_bias']
        self.explicit_bias = net_params['explicit_bias']
        self.gat = net_params['gat']
        self.dataset_name = net_params['dataset_name']
        self.hidden_dim = hidden_dim
        self.O_linear = net_params['O_linear']
        max_wl_role_index = 37 # this is maximum graph size in the dataset
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        embed_dim = (hidden_dim,)*2 if self.dataset_name.lower().startswith('zinc') else (hidden_dim//74, hidden_dim//12) # NOTE: hardcoded for ZINC500k and TOX21
        self.embedding_h = nn.Embedding(num_atom_type, embed_dim[0])

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, embed_dim[1])
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        if not self.gat:
            self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                        self.layer_norm, self.batch_norm, self.residual, use_bias=self.use_bias, explicit_bias=self.explicit_bias, edge_feat=self.edge_feat, O_linear=self.O_linear) for _ in range(n_layers-1) ]) 
            self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual, use_bias=self.use_bias, explicit_bias=self.explicit_bias, edge_feat=self.edge_feat, O_linear=self.O_linear))
        else:
            self.layers = nn.ModuleList([
                EGATConv(
                    in_node_feats=hidden_dim,
                    in_edge_feats=hidden_dim,
                    out_node_feats=output_dim//num_heads,
                    out_edge_feats=output_dim//num_heads,
                    num_heads=num_heads,
                    use_bias=self.use_bias
                ) for output_dim in ([hidden_dim]*(n_layers-1)+[out_dim])
            ])
        self.MLP_layer = MLPReadout(out_dim, num_classes)
        self.embedding_h_noise = None
        self.embedding_e_noise = None
        self.embedding_h_noise_dev = None
        self.embedding_e_noise_dev = None
        self.embedding_h_log = None # used for attacking
        self.embedding_e_log = None # used for attacking
        
    def norm(self, x):
        return x.std()

    def normalize(self, x, dev):
        # NOTE: no EPS, could divide by zero
        return x/self.norm(x)*dev

    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):

        # input embedding
        h = self.embedding_h(h)
        h = h.view(-1, self.hidden_dim)
        self.embedding_h_log = h.cpu()
        if self.embedding_h_noise is not None:
            h = h + self.normalize(self.embedding_h_noise, self.embedding_h_noise_dev)
        h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0),1).to(self.device)
        else: # NOTE: added to ignore edges
            e = self.embedding_e(e)   
        e = e.view(-1, self.hidden_dim)
        self.embedding_e_log = e.cpu()
        if self.embedding_e_noise is not None:
            e = e + self.normalize(self.embedding_e_noise, self.embedding_e_noise_dev)
        
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
            if self.gat:
                h = h.view(-1, reduce(mul, h.shape[-2:], 1))
                e = e.view(-1, reduce(mul, e.shape[-2:], 1))
        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        hg = self.MLP_layer(hg)
        return hg
        
    def loss(self, scores, targets):
        if self.dataset_name == 'TOX21':
            masks = targets[1].bool()
            labels = targets[0]
            loss = nn.BCEWithLogitsLoss()(scores[masks], labels[masks])
        else: # ZINC
            targets = targets.squeeze(-1)
            # loss = nn.MSELoss()(scores,targets)
            loss = nn.L1Loss()(scores, targets)
        return loss
    
    def malog(self, malog:bool):
        for layer in self.layers:
            layer.malog = malog
