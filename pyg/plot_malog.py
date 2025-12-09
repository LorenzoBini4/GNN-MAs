import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

def _local_get_values(a, feat="h", stage="attention", use_layers=None, use_abs=True, flatten=True, cat_batches=True):
    # copy of plot/utils.get_values without the logs.json dependency.
    if isinstance(a, dict):
        data = [(t, a.get(f"layers.malog_{t}")) for t in feat]
    else:
        data = [("e", a)]
    if use_layers is None:
        use_layers = range(len(data[0][1]))
    logs = {k: [] for k in use_layers}
    for _, layers in data:
        if layers is None:
            continue
        for i in use_layers:
            ae = layers[i]
            z = []
            for n in range(len(ae)):
                x = ae[n][stage].detach()
                d = int(np.prod(x.shape[1:]))
                x = x.reshape(-1, d)
                if use_abs:
                    x = x.abs()
                z.append(x)
            if cat_batches:
                logs[i] = torch.cat(z)
                if flatten:
                    logs[i] = logs[i].reshape(-1)
            else:
                logs[i] = z
    return logs

def _hist_layer(values, title, out_path, bins=200):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins, color=(235 / 255, 97 / 255, 35 / 255))
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("|activation|")
    ax.set_ylabel("count")
    ax.set_title(title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_malog(malog_path: Path, out_dir: Path, include_edges: bool):
    with open(malog_path, "rb") as f:
        malog = pickle.load(f)

    # Node activations
    node_values = _local_get_values(malog, feat="h", stage="attention", use_abs=True, flatten=True, cat_batches=True)
    for layer_idx, tensor in node_values.items():
        values = tensor.cpu().numpy().reshape(-1)
        _hist_layer(values, f"Layer {layer_idx} node attention", out_dir / f"layer{layer_idx}_node.png")

    # Edge activations (if any)
    if include_edges and malog.get("layers.malog_e") is not None:
        edge_values = _local_get_values(malog, feat="e", stage="attention", use_abs=True, flatten=True, cat_batches=True)
        for layer_idx, tensor in edge_values.items():
            values = tensor.cpu().numpy().reshape(-1)
            _hist_layer(values, f"Layer {layer_idx} edge attention", out_dir / f"layer{layer_idx}_edge.png")

def main():
    parser = argparse.ArgumentParser(description="Plot MA histograms from malog.pkl produced by PyG GAT")
    parser.add_argument("--malog", required=True, help="Path to malog.pkl")
    parser.add_argument("--out_dir", required=False, help="Directory to save plots (default: alongside malog)")
    parser.add_argument("--edges", action="store_true", help="Plot edge logs if available")
    args = parser.parse_args()

    malog_path = Path(args.malog).resolve()
    if not malog_path.exists():
        raise FileNotFoundError(malog_path)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else malog_path.parent / "plots"
    plot_malog(malog_path, out_dir, include_edges=args.edges)
    print(f"Plots saved to {out_dir}")

if __name__ == "__main__":
    main()
