# PyG GAT Massive Activations

PyTorch Geometric implementation of GAT for node-level Massive Activations logging on the DGL node-feature datasets Chameleon, Squirrel, and MUTAG. Datasets are loaded through DGL to keep the original splits, then converted to PyG `Data` objects.

## Run

```bash
cd pyg
bash scripts/train/gat_chameleon.sh   # or gat_squirrel.sh / gat_mutag.sh
# extra overrides can be passed as CLI args, e.g. --split 1 --epochs 500
```

Logs and checkpoints follow the same layout as the other repo components:

- MA logs: `pyg/out/<DATASET>/logs/.../RUN_0/malog.pkl`
- Checkpoints: `pyg/out/<DATASET>/checkpoints/...`
- Results/config snapshots: `pyg/out/<DATASET>/results|configs/...`

MA logging stores the attention logits (before softmax) per edge/head under `layers.malog_e`, along with the post-attention node activations under `layers.malog_h`.

## Plot MA histograms

Use the helper to plot node (and, if present, edge) attention magnitudes:

```bash
cd pyg
python plot_malog.py --malog out/Squirrel/logs/GraphTransformer_Squirrel_GPU0_.../RUN_0/malog.pkl --edges
# omit --edges to skip edge plots
``` 
