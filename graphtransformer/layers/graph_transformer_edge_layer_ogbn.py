import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

"""
    Graph Transformer Layer with edge features
    
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
def imp_exp_attn(implicit_attn, explicit_edge, edge_feat=True):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """
    if edge_feat:    
        def func(edges):
            return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
    else:
        def func(edges):
            return {implicit_attn: torch.ones_like(edges.data[implicit_attn] * edges.data[explicit_edge], device=edges.data[implicit_attn].device)}
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

# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat, use_edge_feat):
    if use_edge_feat:
        def func(edges):
            return {'e_out': edges.data[edge_feat]}
    else:
        def func(edges):
            return {'e_out': torch.ones_like(edges.data[edge_feat], device=edges.data[edge_feat].device)}
    return func


def exp(field, edge_feat=True):
    if edge_feat:
        def func(edges):
            # clamp for softmax numerical stability
            return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    else:
        def func(edges):
            # clamp for softmax numerical stability
            return {field: torch.exp((torch.ones_like(edges.data[field], device=edges.data[field].device).sum(-1, keepdim=True)).clamp(-5, 5))}
    return func




"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias, edge_feat=True, explicit_bias=False):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.edge_feat = edge_feat
        self.explicit_bias = explicit_bias
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        if explicit_bias:
            self.k_bias = nn.Parameter(torch.randn((num_heads, out_dim, 1)))
            self.e_bias = nn.Parameter(torch.randn((num_heads, 1, out_dim)))
            self.v_bias = nn.Parameter(torch.randn((num_heads, 1, out_dim)))
    
    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        
        # scaling
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        
        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn('score', 'proj_e', edge_feat=self.edge_feat))
        
        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features('score', use_edge_feat=self.edge_feat))
        
        # softmax
        g.apply_edges(exp('score', edge_feat=self.edge_feat))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
    
    def propagate_attention_explicit_bias(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst_bias('K_h', 'Q_h', 'score', explicit_bias=self.k_bias)) #, edges)
        
        # scaling
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        g.apply_edges(scaling('score_bias', np.sqrt(self.out_dim)))
        
        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn_bias('score', 'proj_e', explicit_bias=self.e_bias))
        
        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features('score', use_edge_feat=self.edge_feat))
        
        # softmax
        g.apply_edges(exp('score', edge_feat=self.edge_feat))
        g.apply_edges(exp('score_bias', edge_feat=self.edge_feat))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, src_mul_edge_bias('V_h', 'score', 'V_h', explicit_bias=self.v_bias), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
    
    def forward(self, g, h, e):
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        if self.edge_feat:
            proj_e = self.proj_e(e)
        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g._node_frames[0]['Q_h'] = Q_h.view(*(Q_h.shape[:-1]), self.num_heads, self.out_dim)
        g._node_frames[0]['K_h'] = K_h.view(*(K_h.shape[:-1]), self.num_heads, self.out_dim)
        g._node_frames[0]['V_h'] = V_h.view(*(V_h.shape[:-1]), self.num_heads, self.out_dim)
        if self.edge_feat:
            g.edata['proj_e'] = proj_e.view(*(proj_e.shape[:-1]), self.num_heads, self.out_dim)
        else:
            g.edata['proj_e'] = torch.ones((*(e.shape[:-1]), self.num_heads, self.out_dim), device=e.device)
        
        srcdata = g._node_frames[0]
        dstdata = g._node_frames[1]
        srcdstmask = torch.isin(srcdata['_ID'], dstdata['_ID'])
        for key in srcdata.keys():
            if key not in {'feat', '_ID'}:
                dstdata.update({key: srcdata[key][srcdstmask]})
        
        if self.explicit_bias:
            self.propagate_attention_explicit_bias(g)
        else:
            self.propagate_attention(g)
        
        h_out = g._node_frames[1]['wV'] / (g._node_frames[1]['z'] + torch.full_like(g._node_frames[1]['z'], 1e-6)) # adding eps to all values here
        if self.edge_feat:
            e_out = g.edata['e_out']
        else:
            e_out = e
        
        return h_out, e_out
    

class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False, edge_feat=True, O_linear=True, explicit_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        self.edge_feat = edge_feat
        self.O_linear = O_linear
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias, edge_feat=self.edge_feat, explicit_bias=explicit_bias)

        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)
        
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)
        
        # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_e_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)
        self.malog_h = []
        self.malog_e = []
        self.malog = False
        
    def forward(self, g, h, e):
        if self.malog:
            malog_h = {}
            malog_e = {}
        
        h_in1 = h # for first residual connection
        if self.edge_feat:
            e_in1 = e # for first residual connection
        
        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(g, h, e)
        
        h = h_attn_out.view(-1, self.out_channels)
        if self.edge_feat:
            e = e_attn_out.view(-1, self.out_channels)
        if self.malog:
            malog_h['attention'] = (h.to('cpu'))
            malog_e['attention'] = (e.to('cpu'))
        
        h = F.dropout(h, self.dropout, training=self.training)
        if self.edge_feat:
            e = F.dropout(e, self.dropout, training=self.training)

        if self.O_linear:
            h = self.O_h(h)
            if self.edge_feat:
                e = self.O_e(e)

        if self.residual:
            h = h_in1 + h # residual connection
            if self.edge_feat:
                e = e_in1 + e # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            if self.edge_feat:
                e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            if self.edge_feat:
                e = self.batch_norm1_e(e)

        h_in2 = h # for second residual connection
        if self.edge_feat:
            e_in2 = e # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # FFN for e
        if self.edge_feat:
            e = self.FFN_e_layer1(e)
            e = F.relu(e)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h # residual connection       
            if self.edge_feat:
                e = e_in2 + e # residual connection  

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            if self.edge_feat:
                e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            if self.edge_feat:
                e = self.batch_norm2_e(e)             

        if self.malog:
            self.malog_h.append(malog_h)
            self.malog_e.append(malog_e)
        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)