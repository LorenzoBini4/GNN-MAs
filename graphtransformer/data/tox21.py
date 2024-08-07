from torch.utils.data import Dataset
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.data import Tox21
import dgl
import torch
import numpy as np

class Tox21SplitDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = len(data)
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.data[idx][1:] # graph, labels masks

class Tox21Dataset():
    def __init__(self):
        node_featurizer = CanonicalAtomFeaturizer()
        edge_featurizer = CanonicalBondFeaturizer()
        tox21 = Tox21(node_featurizer=node_featurizer, edge_featurizer=edge_featurizer)
        self.name = 'TOX21'
        self.num_atom_type = torch.cat([x.ndata['h'] for x in tox21.graphs]).unique().shape[0]
        self.num_bond_type = torch.cat([x.edata['e'] for x in tox21.graphs if 'e' in x.edata.keys()]).unique().shape[0]
        l = [x for x in tox21 if 'e' in x[1].edata.keys()]
        for x in l:
            x[1].ndata['feat'] = x[1].ndata['h'].int()+2 # + 2 beacuse the minimum of the values in ndata is -2
            del x[1].ndata['h']
            x[1].edata['feat'] = x[1].edata['e'].int()
            del x[1].edata['e']
        self.l = l
        self.len = len(l)
        idx = np.arange(self.len)
        np.random.shuffle(idx)
        train_idx, val_idx, test_idx = tuple(np.split(idx, [int(0.8*self.len), int(0.9*self.len)]))
        self.train = Tox21SplitDataset([l[x] for x in train_idx])
        self.val = Tox21SplitDataset([l[x] for x in val_idx])
        # self.val = Tox21SplitDataset([l[x] for x in val_idx[:len(val_idx)//2]])
        self.test = Tox21SplitDataset([l[x] for x in test_idx])
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.l.idx
    def collate(self, samples): # copied from ZINC
        # The input samples is a list of pairs (graph, label, mask).
        graphs, labels, masks = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))#.unsqueeze(1)
        masks = torch.tensor(np.array(masks))#.unsqueeze(1)
        batched_graph = dgl.batch(graphs)       
        
        return batched_graph, torch.stack((labels, masks))


