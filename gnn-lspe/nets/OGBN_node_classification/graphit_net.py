import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from scipy import sparse as sp
from scipy.sparse.linalg import norm 

# from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

"""
    GraphiT-GT and GraphiT-GT-LSPE
    
"""

from layers.graphit_gt_layer_proteins import GraphiT_GT_Layer
from layers.graphit_gt_lspe_layer import GraphiT_GT_LSPE_Layer
from layers.mlp_readout_layer import MLPReadout

class GraphiTNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        
        full_graph = net_params['full_graph']
        gamma = net_params['gamma']
        self.adaptive_edge_PE = net_params['adaptive_edge_PE']
        
        GT_layers = net_params['L']
        GT_hidden_dim = net_params['hidden_dim']
        GT_out_dim = net_params['out_dim']
        GT_n_heads = net_params['n_heads']
        
        self.residual = net_params['residual']
        self.readout = net_params['readout']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']

        n_classes = net_params['n_classes']
        self.device = net_params['device']
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.pe_init = net_params['pe_init']
        
        self.use_lapeig_loss = net_params['use_lapeig_loss']
        self.lambda_loss = net_params['lambda_loss']
        self.alpha_loss = net_params['alpha_loss']
        
        self.pos_enc_dim = net_params['pos_enc_dim']
        self.use_bias = net_params['use_bias']
        
        if self.pe_init in ['rand_walk']:
            self.embedding_p = nn.Linear(self.pos_enc_dim, GT_hidden_dim)
        
        # self.embedding_h = AtomEncoder(GT_hidden_dim)
        # self.embedding_e = BondEncoder(GT_hidden_dim)
        self.embedding_h = nn.Embedding(net_params['num_atom_type'], GT_hidden_dim)
        self.embedding_e = nn.Linear(net_params['num_bond_type'], GT_hidden_dim)
        
        explicit_bias = net_params['explicit_bias']

        if self.pe_init == 'rand_walk':
            # LSPE
            self.layers = nn.ModuleList([ GraphiT_GT_LSPE_Layer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, full_graph,
                                                                dropout, self.layer_norm, self.batch_norm, self.residual, self.adaptive_edge_PE, use_bias=self.use_bias) for _ in range(GT_layers-1) ])
            self.layers.append(GraphiT_GT_LSPE_Layer(gamma, GT_hidden_dim, GT_out_dim, GT_n_heads, full_graph,
                                                     dropout, self.layer_norm, self.batch_norm, self.residual, self.adaptive_edge_PE, use_bias=self.use_bias))
        else: 
            # NoPE
            self.layers = nn.ModuleList([ GraphiT_GT_Layer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, full_graph,
                                                                dropout, self.layer_norm, self.batch_norm, self.residual, self.adaptive_edge_PE, use_bias=self.use_bias, explicit_bias=explicit_bias) for _ in range(GT_layers-1) ])
            self.layers.append(GraphiT_GT_Layer(gamma, GT_hidden_dim, GT_out_dim, GT_n_heads, full_graph,
                                                     dropout, self.layer_norm, self.batch_norm, self.residual, self.adaptive_edge_PE, use_bias=self.use_bias, explicit_bias=explicit_bias))
        
        self.MLP_layer = MLPReadout(GT_out_dim, n_classes)   
 
        if self.pe_init == 'rand_walk':
            self.p_out = nn.Linear(GT_out_dim, self.pos_enc_dim)
            self.Whp = nn.Linear(GT_out_dim+self.pos_enc_dim, GT_out_dim)
        
        self.g = None              # For util; To be accessed in loss() function
        
    def _update_ndata(self, g, key, val):
        g.ndata.update({key: {'_N': val}})

    def forward(self, blocks, h, p, e, snorm_n=None):
        
        h = self.embedding_h(blocks[0].ndata['feat']['_N'])

        if self.pe_init in ['rand_walk']:
            p = self.embedding_p(p)
        
        for i in range(len(self.layers)):
            e = self.embedding_e(blocks[i].edata['feat'])
            h, p = self.layers[i](blocks[i], h, p, e, snorm_n)
            # NOTE: h is already masked, p is not used

        g = blocks[-1]
        
        if self.pe_init == 'rand_walk':
            p = self.p_out(p)
            self._update_ndata(g, 'p', p)
            # Implementing p_g = p_g - torch.mean(p_g, dim=0)
            means = dgl.mean_nodes(g, 'p')
            batch_wise_p_means = means.repeat_interleave(g.batch_num_nodes(), 0)
            p = p - batch_wise_p_means

            # Implementing p_g = p_g / torch.norm(p_g, p=2, dim=0)
            self._update_ndata(g, 'p', p)
            self._update_ndata(g, 'p2', g.ndata['p']['_N']**2)
            norms = dgl.sum_nodes(g, 'p2')
            norms = torch.sqrt(norms+1e-6)            
            batch_wise_p_l2_norms = norms.repeat_interleave(g.batch_num_nodes(), 0)
            p = p / batch_wise_p_l2_norms
            self._update_ndata(g, 'p', p)
        
            # Concat h and p
            hp = self.Whp(torch.cat((g.ndata['h']['_N'],g.ndata['p']['_N']),dim=-1))
            self._update_ndata(g, 'h', hp)
        
        hg = h
            
        self.g = g # For util; To be accessed in loss() function
        
        return self.MLP_layer(hg), g
        
    def loss(self, pred, labels):
        
        # Loss A: Task loss -------------------------------------------------------------
        loss_a = torch.nn.BCEWithLogitsLoss()(pred, labels)
        
        if self.use_lapeig_loss:
            raise NotImplementedError
        else:
            loss = loss_a
            
        return loss

    def malog(self, malog:bool):
        for layer in self.layers:
            layer.malog = malog
