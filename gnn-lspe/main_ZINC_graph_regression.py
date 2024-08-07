




"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt

import pickle
import copy

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self






"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from nets.ZINC_graph_regression.load_net import gnn_model # import all GNNS
from data.data import LoadData # import dataset




"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device








"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs, train_bool=True, weights=None, epoch=0):
    global model
    global train_loader
    global val_loader
    global test_loader
    t0 = time.time()
    per_epoch_time = []
        
    DATASET_NAME = dataset.name
    
    if net_params['pe_init'] == 'lap_pe':
        tt = time.time()
        print("[!] -LapPE: Initializing graph positional encoding with Laplacian PE.")
        dataset._add_lap_positional_encodings(net_params['pos_enc_dim'])
        print("[!] Time taken: ", time.time()-tt)
    elif net_params['pe_init'] == 'rand_walk':
        tt = time.time()
        print("[!] -LSPE: Initializing graph positional encoding with rand walk features.")
        dataset._init_positional_encodings(net_params['pos_enc_dim'], net_params['pe_init'])
        print("[!] Time taken: ", time.time()-tt)
        
        tt = time.time()
        print("[!] -LSPE (For viz later): Adding lapeigvecs to key 'eigvec' for every graph.")
        dataset._add_eig_vecs(net_params['pos_enc_dim'])
        print("[!] Time taken: ", time.time()-tt)
        
    if MODEL_NAME in ['SAN', 'GraphiT']:
        if net_params['full_graph']:
            st = time.time()
            print("[!] Adding full graph connectivity..")
            dataset._make_full_graph() if MODEL_NAME == 'SAN' else dataset._make_full_graph((net_params['p_steps'], net_params['gamma']))
            print('Time taken to add full graph connectivity: ',time.time()-st)
    
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
        
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file, viz_dir = dirs
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
        torch.cuda.manual_seed_all(params['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))

    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)
    if weights is not None:
        model.load_state_dict(weights)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_MAEs, epoch_val_MAEs = [], [] 
    
    # import train functions for all GNNs
    from train.train_ZINC_graph_regression import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network

    train_loader = DataLoader(trainset, num_workers=4, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
    val_loader = DataLoader(valset, num_workers=4, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, num_workers=4, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    test_log_loader = DataLoader(testset, num_workers=4, batch_size=1, shuffle=False, collate_fn=dataset.collate)
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(epoch, params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss, epoch_train_mae, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
                    
                epoch_val_loss, epoch_val_mae, __ = evaluate_network(model, device, val_loader, epoch)
                epoch_test_loss, epoch_test_mae, __ = evaluate_network(model, device, test_loader, epoch)
                del __
                
                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_MAEs.append(epoch_train_mae)
                epoch_val_MAEs.append(epoch_val_mae)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('train/_mae', epoch_train_mae, epoch)
                writer.add_scalar('val/_mae', epoch_val_mae, epoch)
                writer.add_scalar('test/_mae', epoch_test_mae, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                        
                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_MAE=epoch_train_mae, val_MAE=epoch_val_mae,
                              test_MAE=epoch_test_mae)


                per_epoch_time.append(time.time()-start)

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch-1:
                        os.remove(file)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break
                
                # Stop training after params['max_time'] hours
                if time.time()-t0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
                
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    # model.malog(True)
    if params['attack'] == -1:
        train_loss_lapeig, train_mae, g_outs_train = evaluate_network(model, device, train_loader, epoch)
        val_loss_lapeig, val_mae, g_outs_val  = evaluate_network(model, device, val_loader, epoch)
        model.malog(True)
        test_loss_lapeig, test_mae, g_outs_test = evaluate_network(model, device, test_log_loader, epoch)

        print("Epoch", epoch, end=' - ')
        print("Val LOSS: {:.4f}".format(val_loss_lapeig), end=' - ')
        print("Val ACC: {:.4f}".format(val_mae), end=' - ')
        print("Test LOSS: {:.4f}".format(test_loss_lapeig), end=' - ')
        print("Test ACC: {:.4f}".format(test_mae))

        print("Test MAE: {:.4f}".format(test_mae))
        print("Train MAE: {:.4f}".format(train_mae))
        print("Convergence Time (Epochs): {:.4f}".format(epoch))
        print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
        print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
        
        
        if net_params['pe_init'] == 'rand_walk':
            # Visualize actual and predicted/learned eigenvecs
            from utils.plot_util import plot_graph_eigvec
            if not os.path.exists(viz_dir):
                os.makedirs(viz_dir)

            sample_graph_ids = [15,25,45]

            for f_idx, graph_id in enumerate(sample_graph_ids):

                # Test graphs
                g_dgl = g_outs_test[graph_id]

                f = plt.figure(f_idx, figsize=(12,6))

                plt1 = f.add_subplot(121)
                plot_graph_eigvec(plt1, graph_id, g_dgl, feature_key='eigvec', actual_eigvecs=True)

                plt2 = f.add_subplot(122)
                plot_graph_eigvec(plt2, graph_id, g_dgl, feature_key='p', predicted_eigvecs=True)

                f.savefig(viz_dir+'/test'+str(graph_id)+'.jpg')

                # Train graphs
                g_dgl = g_outs_train[graph_id]

                f = plt.figure(f_idx, figsize=(12,6))

                plt1 = f.add_subplot(121)
                plot_graph_eigvec(plt1, graph_id, g_dgl, feature_key='eigvec', actual_eigvecs=True)

                plt2 = f.add_subplot(122)
                plot_graph_eigvec(plt2, graph_id, g_dgl, feature_key='p', predicted_eigvecs=True)

                f.savefig(viz_dir+'/train'+str(graph_id)+'.jpg')

        writer.close()

        """
            Write the results in out_dir/results folder
        """
        with open(write_file_name + '.txt', 'w') as f:
            f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
        FINAL RESULTS\nTEST MAE: {:.4f}\nTRAIN MAE: {:.4f}\n\n
        Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
            .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                    test_mae, train_mae, epoch, (time.time()-t0)/3600, np.mean(per_epoch_time)))
    else:
        model.malog(False)
        def _freeze_r(p):
            for x in p:
                try:
                    _freeze_r(x.parameters())
                except:
                    pass
                x.requires_grad = False
        _freeze_r(model.parameters())
        noise_loader = test_loader
        sample_loss, sample_mae, _ = evaluate_network(model, device, noise_loader, epoch)
        def noise_loss_function(m, s, t): # model, scores, targets
            return -m.loss(s,t)
        def noise_zero_grad(i, m, o): # index, model, optimizer
            if i==0:
                m.zero_grad()
                o.zero_grad()
        print('EMBEDDING H SHAPE:', model.embedding_h_log.shape)
        print('EMBEDDING E SHAPE:', model.embedding_e_log.shape)
        print()
        print('EMBEDDING H STDDEV:', model.norm(model.embedding_h_log))
        print('EMBEDDING E STDDEV:', model.norm(model.embedding_e_log))
        print('SAMPLE LOSS (true value):', sample_loss)
        print('SAMPLE MAE:', sample_mae)
        print()
        print('='*80)
        print()

        devs = np.logspace(-2, -1, 3)
        for dev in devs:
            dev = np.float64(dev)
            sample_losses_random = []
            sample_maes_random = []
            sample_losses_optimized = []
            sample_maes_optimized = []
            for _ in range(1):
                # inject random noise
                model.embedding_h_noise_dev = dev
                model.embedding_e_noise_dev = dev
                if params['shared_noise']:
                    model.embedding_h_noise = nn.Parameter(torch.randn(model.embedding_h_log.shape[-1], dtype=model.embedding_h_log.dtype, device=model.device))
                    model.embedding_e_noise = nn.Parameter(torch.randn(model.embedding_e_log.shape[-1], dtype=model.embedding_e_log.dtype, device=model.device))
                else:
                    model.embedding_h_noise = nn.Parameter(torch.randn(model.embedding_h_log.shape, dtype=model.embedding_h_log.dtype, device=model.device))
                    model.embedding_e_noise = nn.Parameter(torch.randn(model.embedding_e_log.shape, dtype=model.embedding_e_log.dtype, device=model.device))
                # test with random noise
                sample_loss, sample_mae, _ = evaluate_network(model, device, noise_loader, epoch)
                sample_losses_random.append(sample_loss)
                sample_maes_random.append(sample_mae)
                # attack (optimize noise)
                noise_optimizer = optim.Adam([
                    model.embedding_h_noise,
                    model.embedding_e_noise
                ], lr=params['init_lr'], weight_decay=params['weight_decay'])
                noise_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    noise_optimizer, mode='min',
                    factor=params['lr_reduce_factor'],
                    patience=params['lr_schedule_patience'],
                    verbose=True
                )
                noise_epoch = 0
                for noise_epoch in tqdm(list(range(1000))):
                    epoch_train_loss, epoch_train_mae, noise_optimizer = train_epoch(model, noise_optimizer, device, noise_loader, noise_epoch, train_loss_function=noise_loss_function, zero_grad=noise_zero_grad, model_train=lambda m:m.eval())
                    noise_scheduler.step(epoch_train_loss)
                    if noise_optimizer.param_groups[0]['lr'] < params['min_lr']:
                        break
                sample_loss, sample_mae, _ = evaluate_network(model, device, noise_loader, noise_epoch)
                sample_losses_optimized.append(sample_loss)
                sample_maes_optimized.append(sample_mae)
            
            # PRINT
            print()
            sample_losses_random = np.array(sample_losses_random)
            sample_maes_random = np.array(sample_maes_random)
            print(f'RANDOM NORMAL NOISE STDDEV: {dev}')
            print(f'SAMPLE LOSSES: AVG {sample_losses_random.mean()} - STD {sample_losses_random.std()}')
            print(f'SAMPLE MAES: AVG {sample_maes_random.mean()} - STD {sample_maes_random.std()}')
            print()
            print('-'*80)
            print()

            sample_losses_optimized = np.array(sample_losses_optimized)
            sample_maes_optimized = np.array(sample_maes_optimized)
            print(f'OPTIMIZED NOISE STDDEV: {model.embedding_h_noise_dev} | {model.embedding_e_noise_dev}')
            print(f'SAMPLE LOSSES: AVG {sample_losses_optimized.mean()} - STD {sample_losses_optimized.std()}')
            print(f'SAMPLE MAES: AVG {sample_maes_optimized.mean()} - STD {sample_maes_optimized.std()}')
            print()
            print(f'GAIN: {sample_losses_optimized.mean()/sample_losses_random.mean()-1}')
            print()
    return model
        




def main():    
    global dataset
    """
        USER CONTROLS
    """
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")    
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--pos_enc_dim', help="Please give a value for pos_enc_dim")
    parser.add_argument('--pos_enc', help="Please give a value for pos_enc")
    parser.add_argument('--alpha_loss', help="Please give a value for alpha_loss")
    parser.add_argument('--lambda_loss', help="Please give a value for lambda_loss")
    parser.add_argument('--pe_init', help="Please give a value for pe_init")
    parser.add_argument('--weights', help="pkl for pre-trained model weights")
    parser.add_argument('--use_bias', help="'True' for true")
    parser.add_argument('--gat', help="'True' for GAT, false for GT (default)")
    parser.add_argument('--attack', type=int, help="position of sample to attack", default=-1)
    parser.add_argument('--explicit_bias', help="use explicit attention bias", default='False')
    parser.add_argument('--shared_noise', help='for attack, "True" for shared noise among nodes / edges (allows batching), "False" for independent noise', default='False')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
        
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    if args.shared_noise is not None:
        params['shared_noise'] = True if args.shared_noise=='True' else False
    params['attack'] = int(args.attack)
    params['weights'] = args.weights
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)   
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.pos_enc is not None:
        net_params['pos_enc'] = True if args.pos_enc=='True' else False
    if args.pos_enc_dim is not None:
        net_params['pos_enc_dim'] = int(args.pos_enc_dim)
    if args.alpha_loss is not None:
        net_params['alpha_loss'] = float(args.alpha_loss)
    if args.lambda_loss is not None:
        net_params['lambda_loss'] = float(args.lambda_loss)
    if args.pe_init is not None:
        net_params['pe_init'] = args.pe_init

    if args.explicit_bias is not None:
        net_params['explicit_bias'] = True if args.explicit_bias=='True' else False
    net_params['gat'] = True if (args.gat is not None and args.gat == 'True') else False
    net_params['use_bias'] = True if (args.use_bias is not None and args.use_bias=='True') else False
        
    
    # ZINC
    net_params['num_atom_type'] = dataset.num_atom_type
    net_params['num_bond_type'] = dataset.num_bond_type

    if MODEL_NAME == 'PNA':
        D = torch.cat([torch.sparse.sum(g.adjacency_matrix(transpose=True), dim=-1).to_dense() for g in
                       dataset.train.graph_lists])
        net_params['avg_d'] = dict(lin=torch.mean(D),
                                   exp=torch.mean(torch.exp(torch.div(1, D)) - 1),
                                   log=torch.mean(torch.log(D + 1)))
    
    logname = MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Y_%m_%d_%Hh%Mm%Ss')
    root_log_dir = out_dir + 'logs/' + logname
    root_ckpt_dir = out_dir + 'checkpoints/' + logname
    write_file_name = out_dir + 'results/result_' + logname
    write_config_file = out_dir + 'configs/config_' + logname
    viz_dir = out_dir + 'viz/' + logname
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file, viz_dir

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    if params['weights'] is None:
        train_bool = True
        epoch=0
        weights = None
    else:
        train_bool = False
        epoch = int(params['weights'].rsplit('.',1)[0].rsplit('_',1)[-1])
        weights = torch.load(params['weights'])

    print(params)
    print(net_params)

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    model_out = train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs, train_bool=train_bool, weights=weights, epoch=epoch)

    if params['attack'] == -1:
        data = {
            'layers.malog_h': [layer.malog_h for layer in model_out.layers],
            # 'layers.malog_p': [layer.malog_p for layer in model_out.layers],
            'layers.malog_e': [layer.malog_e for layer in model_out.layers]
        }
        with open(os.path.join(root_log_dir, 'RUN_0', 'malog.pkl'),'wb') as outfile:
            pickle.dump(data, outfile)

    try:
        print(model.layers[0].attention_h)
    except:
        pass
    print('MA logs in', os.path.join(root_log_dir, 'RUN_0', 'malog.pkl'))
    print('checkpoints in', root_ckpt_dir)

    
    
    
    
    
    
    
main()    





