"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math

import dgl

from train.metrics import MAE



def train_epoch_sparse(model, optimizer, device, data_loader, epoch, train_loss_function = lambda m, s, t : m.loss(s, t), zero_grad = lambda i, m, o : o.zero_grad() if i==0 else None, model_train = lambda m : m.train()):
    model_train(model)
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_targets, batch_snorm_n) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_targets = batch_targets.to(device)
        batch_targets = batch_targets.squeeze(-1)
        batch_snorm_n = batch_snorm_n.to(device)
        # optimizer.zero_grad()
        zero_grad(0, model, optimizer)

        try:
            batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
        except KeyError:
            batch_pos_enc = None
        
        if model.pe_init == 'lap_pe':
            sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)

        batch_scores, __ = model.forward(batch_graphs, batch_x, batch_pos_enc, batch_e, batch_snorm_n)
        del __

        loss = train_loss_function(model, batch_scores, batch_targets)
        loss.backward()
        zero_grad(1, model, optimizer)
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    
    return epoch_loss, epoch_train_mae, optimizer

def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    nb_data = 0
    out_graphs_for_lapeig_viz = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets, batch_snorm_n) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            batch_targets = batch_targets.squeeze(-1)
            batch_snorm_n = batch_snorm_n.to(device)

            try:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
            except KeyError:
                batch_pos_enc = None
                
            batch_scores, batch_g = model.forward(batch_graphs, batch_x, batch_pos_enc, batch_e, batch_snorm_n)

            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
            
            out_graphs_for_lapeig_viz += dgl.unbatch(batch_g)
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        
    return epoch_test_loss, epoch_test_mae, out_graphs_for_lapeig_viz

