import torch
import torch.nn.functional as F
import dgl

from torch_geometric.data import Data
from dgl.data import ChameleonDataset, SquirrelDataset, MUTAGDataset

class DGLPyGNodeDataset:
    """
    Utility loader that keeps the official DGL datasets but exposes them as
    PyG Data objects so we can run PyG models without changing the source
    datasets.
    """

    def __init__(self, name: str, split: int = 0, raw_dir: str = "dataset_cache"):
        self.name = name.lower()
        self.split = split
        self.raw_dir = raw_dir
        self.data = None
        self.num_classes = None
        self._load()

    def _load(self):
        if self.name == "chameleon":
            ds = ChameleonDataset(raw_dir=self.raw_dir)
            self.num_classes = ds.num_classes
            self.data = self._convert_geom_gcn(ds[0])
        elif self.name == "squirrel":
            ds = SquirrelDataset(raw_dir=self.raw_dir)
            self.num_classes = ds.num_classes
            self.data = self._convert_geom_gcn(ds[0])
        elif self.name == "mutag":
            ds = MUTAGDataset(raw_dir=self.raw_dir)
            self.num_classes = ds.num_classes
            self.predict_category = ds.predict_category
            self.data = self._convert_rdf(ds[0])
        else:
            raise ValueError(f"Unsupported dataset {self.name}")

    def _pick_mask_column(self, mask: torch.Tensor):
        if mask is None:
            return None
        if mask.dim() == 1:
            return mask.bool()
        if self.split >= mask.shape[1]:
            raise ValueError(f"Requested split {self.split} but only {mask.shape[1]} splits available")
        return mask[:, self.split].bool()

    def _convert_geom_gcn(self, g: dgl.DGLGraph) -> Data:
        src, dst = g.edges()
        data = Data(
            x=g.ndata["feat"].float(),
            y=g.ndata["label"].long(),
            edge_index=torch.stack([src, dst], dim=0),
            num_nodes=g.num_nodes(),
        )
        for key in ["train_mask", "val_mask", "test_mask"]:
            if key in g.ndata:
                data[key] = self._pick_mask_column(g.ndata[key])
        data.num_classes = self.num_classes
        return data

    def _convert_rdf(self, hg: dgl.DGLHeteroGraph) -> Data:
        homo = dgl.to_homogeneous(hg)
        type_ids = homo.ndata[dgl.NTYPE]
        nid = homo.ndata[dgl.NID]
        pred_type = hg.get_ntype_id(self.predict_category)
        pred_mask = type_ids == pred_type

        # Features: prefer existing features, otherwise fall back to a one-hot of node types.
        feat = homo.ndata.get("feat")
        if feat is None:
            feat = F.one_hot(type_ids, num_classes=len(hg.ntypes)).float()
        else:
            feat = feat.float()

        y = torch.full((homo.num_nodes(),), -1, dtype=torch.long)
        labels_pred = hg.nodes[self.predict_category].data["label"].view(-1)
        y[pred_mask] = labels_pred[nid[pred_mask]]

        def build_mask(name: str):
            src_mask = hg.nodes[self.predict_category].data.get(name)
            mask = torch.zeros_like(y, dtype=torch.bool)
            if src_mask is not None:
                mask[pred_mask] = self._pick_mask_column(src_mask)[nid[pred_mask]]
            return mask

        train_mask = build_mask("train_mask")
        val_mask = build_mask("val_mask")
        test_mask = build_mask("test_mask")

        # If no validation mask is provided, carve out a small split from the training nodes.
        if val_mask.sum() == 0 and train_mask.sum() > 0:
            train_nodes = torch.nonzero(train_mask, as_tuple=False).view(-1)
            val_count = max(1, int(0.1 * train_nodes.numel()))
            val_indices = train_nodes[:val_count]
            train_mask[val_indices] = False
            val_mask[val_indices] = True

        data = Data(
            x=feat,
            y=y,
            edge_index=torch.stack(homo.edges(), dim=0),
            num_nodes=homo.num_nodes(),
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_classes=self.num_classes,
        )
        return data
