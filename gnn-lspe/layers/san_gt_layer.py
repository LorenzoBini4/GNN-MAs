import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

"""
    SAN-GT
    
"""

"""
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
    return func

def src_dot_dst_bias(src_field, dst_field, out_field, explicit_bias):
    def func(edges):
        out = (edges.src[src_field] * edges.dst[dst_field])
        out_bias = torch.transpose(edges.dst[dst_field], 0, 1) @ explicit_bias
        out_bias = torch.transpose(out_bias, 0, 1)
        return {
            out_field: out,
            f'{out_field}_bias': out_bias
        }
    return func

def imp_exp_attn_bias(implicit_attn, explicit_edge, explicit_bias=None):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """
    def func(edges):
        attn = edges.data[implicit_attn]
        src_bias = edges.data[f'{implicit_attn}_bias']
        out = attn * edges.data[explicit_edge]
        src_bias = torch.transpose(src_bias, 0, 1) # h*e*1
        bias = src_bias @ explicit_bias # h*e*d
        bias = torch.transpose(bias, 0, 1) # e*h*d
        return {implicit_attn: out + bias}
    return func

def src_mul_edge_bias(src_field, edge_field, out_field, explicit_bias):
    def func(edges):
        out = edges.src[src_field] * edges.data[edge_field]
        src_bias = edges.data[edge_field]
        src_bias = torch.transpose(src_bias, 0, 1) # h*e*1
        bias = src_bias @ explicit_bias # h*e*d
        bias = torch.transpose(bias, 0, 1) # e*h*d
        return {out_field: out + bias}
    return func

def exp_real(field, L):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))/(L+1)}
    return func


def exp_fake(field, L):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': L*torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))/(L+1)}
    return func

def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func


"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph, use_bias, attention_for, explicit_bias=False):
        super().__init__()
        
       
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.full_graph=full_graph
        self.attention_for = attention_for
        self.gamma = gamma
        self.explicit_bias = explicit_bias
        
        if self.attention_for == "h":
            if use_bias:
                self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                self.E = nn.Linear(in_dim, out_dim * num_heads, bias=True)

                if self.full_graph:
                    self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                    self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                    self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)

                self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)

            else:
                self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                self.E = nn.Linear(in_dim, out_dim * num_heads, bias=False)

                if self.full_graph:
                    self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                    self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                    self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)

                self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        if explicit_bias:
            self.k_bias = nn.Parameter(torch.randn((num_heads, out_dim, 1)))
            self.e_bias = nn.Parameter(torch.randn((num_heads, 1, out_dim)))
            self.v_bias = nn.Parameter(torch.randn((num_heads, 1, out_dim)))
        
        
    def propagate_attention(self, g):

        
        if self.full_graph:
            real_ids = torch.nonzero(g.edata['real']).squeeze()
            fake_ids = torch.nonzero(g.edata['real']==0).squeeze()

        else:
            real_ids = g.edges(form='eid')
            
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'), edges=real_ids)
        
        if self.full_graph:
            g.apply_edges(src_dot_dst('K_2h', 'Q_2h', 'score'), edges=fake_ids)
        

        # scale scores by sqrt(d)
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        
        # Use available edge features to modify the scores for edges
        g.apply_edges(imp_exp_attn('score', 'E'), edges=real_ids)
        
        if self.full_graph:
            g.apply_edges(imp_exp_attn('score', 'E_2'), edges=fake_ids)
            
        
        if self.full_graph:
            # softmax and scaling by gamma
            L = torch.clamp(self.gamma, min=0.0, max=1.0)   # Gamma \in [0,1]
            g.apply_edges(exp_real('score', L), edges=real_ids)
            g.apply_edges(exp_fake('score', L), edges=fake_ids)
        
        else:
            g.apply_edges(exp('score'), edges=real_ids)

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score_soft', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score_soft', 'score_soft'), fn.sum('score_soft', 'z'))
        
    def propagate_attention_explicit_bias(self, g):

        
        if self.full_graph:
            real_ids = torch.nonzero(g.edata['real']).squeeze()
            fake_ids = torch.nonzero(g.edata['real']==0).squeeze()

        else:
            real_ids = g.edges(form='eid')
            
        g.apply_edges(src_dot_dst_bias('K_h', 'Q_h', 'score', explicit_bias=self.k_bias), edges=real_ids)
        
        if self.full_graph:
            g.apply_edges(src_dot_dst_bias('K_2h', 'Q_2h', 'score', explicit_bias=self.k_bias_2), edges=fake_ids)
        

        # scale scores by sqrt(d)
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        g.apply_edges(scaling('score_bias', np.sqrt(self.out_dim)))
        
        # Use available edge features to modify the scores for edges
        g.apply_edges(imp_exp_attn_bias('score', 'E', explicit_bias=self.e_bias), edges=real_ids)
        
        if self.full_graph:
            g.apply_edges(imp_exp_attn_bias('score', 'E_2', explicit_bias=self.e_bias_2), edges=fake_ids)
            
        
        if self.full_graph:
            # softmax and scaling by gamma
            L = torch.clamp(self.gamma, min=0.0, max=1.0)   # Gamma \in [0,1]
            g.apply_edges(exp_real('score', L), edges=real_ids)
            g.apply_edges(exp_fake('score', L), edges=fake_ids)
            g.apply_edges(exp_real('score_bias', L), edges=real_ids)
            g.apply_edges(exp_fake('score_bias', L), edges=fake_ids)
        
        else:
            g.apply_edges(exp('score'), edges=real_ids)
            g.apply_edges(exp('score_bias'), edges=real_ids)

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, src_mul_edge_bias('V_h', 'score_soft', 'V_h', explicit_bias=self.v_bias), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score_soft', 'score_soft'), fn.sum('score_soft', 'z'))
        
    
    def forward(self, g, h, e):
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        E = self.E(e)
        
        if self.full_graph:
            Q_2h = self.Q_2(h)
            K_2h = self.K_2(h)
            E_2 = self.E_2(e)
            
        V_h = self.V(h)

        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.edata['E'] = E.view(-1, self.num_heads, self.out_dim)
        
        
        if self.full_graph:
            g.ndata['Q_2h'] = Q_2h.view(-1, self.num_heads, self.out_dim)
            g.ndata['K_2h'] = K_2h.view(-1, self.num_heads, self.out_dim)
            g.edata['E_2'] = E_2.view(-1, self.num_heads, self.out_dim)
        
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        if self.explicit_bias:
            self.propagate_attention_explicit_bias(g)
        else:
            self.propagate_attention(g)
        
        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))
        
        return h_out, g.edata['score']
    

class SAN_GT_Layer(nn.Module):
    """
        Param: 
    """
    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph, dropout=0.0,
                 layer_norm=False, batch_norm=True, residual=True, use_bias=False, explicit_bias=False):
        super().__init__()
        
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        
        self.attention_h = MultiHeadAttentionLayer(gamma, in_dim, out_dim//num_heads, num_heads,
                                                   full_graph, use_bias, attention_for="h", explicit_bias=explicit_bias)
        
        self.O_h = nn.Linear(out_dim, out_dim)
        
        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
        
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
        self.malog_h = []
        self.malog_e = []
        self.malog = False
            
        
    def forward(self, g, h, p, e, snorm_n):
        if self.malog:
            malog_h = {}
            malog_e = {}
            self.malog_h.append(malog_h)
            self.malog_e.append(malog_e)
        
        h_in1 = h # for first residual connection
        
        # [START] For calculation of h -----------------------------------------------------------------
        
        # multi-head attention out
        h_attn_out, e_attn_out = self.attention_h(g, h, e)
        
        #Concat multi-head outputs
        h = h_attn_out.view(-1, self.out_channels)
        e_attn_out = e_attn_out.view(-1, e_attn_out.shape[-2]*e_attn_out.shape[-1])
        if self.malog:
            malog_h['attention'] = (h.to('cpu'))
            malog_e['attention'] = (e_attn_out.to('cpu'))
       
        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h # residual connection

        # GN from benchmarking-gnns-v1
        # h = h * snorm_n
        
        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h # residual connection       

        # GN from benchmarking-gnns-v1
        # h = h * snorm_n
        
        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)         
        
        # [END] For calculation of h -----------------------------------------------------------------
        
        return h, None
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)