import os
import sys
import glob
import time
import json
import random
import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
os.environ.setdefault("DGLBACKEND", "pytorch")

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from data import DGLPyGNodeDataset
from nets import GATNet
from train import train_epoch, evaluate_network

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cuda_ok = torch.cuda.is_available()
    if use_gpu and not cuda_ok:
        raise RuntimeError("Requested GPU but CUDA is not available. Check drivers/visibility.")
    if cuda_ok and use_gpu:
        print("cuda available with GPU:", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    print("Using CPU.")
    return torch.device("cpu")

def view_model_param(net_params):
    model = GATNet(net_params)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print("MODEL/Total parameters:", total_param)
    return total_param

def train_val_pipeline(params, net_params, dataset, dirs, train_bool=True, weights=None, epoch_start=0):
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params["device"]
    data = dataset.data.to(device)

    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params["seed"])
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed(params["seed"])

    model = GATNet(net_params).to(device)
    if weights is not None:
        model.load_state_dict(weights)

    optimizer = optim.Adam(
        model.parameters(), lr=params["init_lr"], weight_decay=params["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=params["lr_reduce_factor"],
        patience=params["lr_schedule_patience"],
        verbose=True,
        min_lr=params["min_lr"],
    )

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_accs, epoch_val_accs = [], []
    per_epoch_time = []

    start0 = time.time()
    if train_bool:
        try:
            for epoch in range(epoch_start, params["epochs"]):
                start = time.time()

                epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, data, device)
                epoch_val_loss, epoch_val_acc = evaluate_network(model, data, device, "val")

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_accs.append(epoch_train_acc)
                epoch_val_accs.append(epoch_val_acc)

                writer.add_scalar("train/_loss", epoch_train_loss, epoch)
                writer.add_scalar("val/_loss", epoch_val_loss, epoch)
                writer.add_scalar("train/_acc", epoch_train_acc, epoch)
                writer.add_scalar("val/_acc", epoch_val_acc, epoch)
                writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

                scheduler.step(epoch_val_acc)
                per_epoch_time.append(time.time() - start)

                if (epoch + 1) % params["print_epoch_interval"] == 0:
                    print(
                        f"Epoch {epoch}: train_loss {epoch_train_loss:.4f}, val_loss {epoch_val_loss:.4f}, "
                        f"train_acc {epoch_train_acc:.4f}, val_acc {epoch_val_acc:.4f}, lr {optimizer.param_groups[0]['lr']:.6f}"
                    )

                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(), f"{ckpt_dir}/epoch_{epoch}.pkl")

                files = glob.glob(ckpt_dir + "/*.pkl")
                for file in files:
                    epoch_nb = int(file.split("_")[-1].split(".")[0])
                    if epoch_nb < epoch - 1:
                        os.remove(file)

                if optimizer.param_groups[0]["lr"] <= params["min_lr"]:
                    print("Reached minimum learning rate threshold; stopping training.")
                    break

                if time.time() - start0 > params["max_time"] * 3600:
                    print("Max training time reached; stopping.")
                    break

        except KeyboardInterrupt:
            print("Exiting early because of KeyboardInterrupt")

    train_loss, train_acc = evaluate_network(model, data, device, "train")
    val_loss, val_acc = evaluate_network(model, data, device, "val")
    test_loss, test_acc = evaluate_network(model, data, device, "test")
    print(f"Final Train Acc: {train_acc:.4f}")
    print(f"Final Val Acc: {val_acc:.4f}")
    print(f"Final Test Acc: {test_acc:.4f}")
    if per_epoch_time:
        print(f"AVG TIME PER EPOCH: {np.mean(per_epoch_time):.4f}s")

    writer.close()

    with open(write_file_name + ".txt", "w") as f:
        f.write(
            f"""Dataset: {dataset.name},
Params={params}
NetParams={net_params}

FINAL RESULTS
TRAIN LOSS: {train_loss:.4f}
VAL LOSS: {val_loss:.4f}
TEST LOSS: {test_loss:.4f}
TRAIN ACCURACY: {train_acc:.4f}
VAL ACCURACY: {val_acc:.4f}
TEST ACCURACY: {test_acc:.4f}
"""
        )

    # MA logging on the evaluation forward pass (before any softmax in attention).
    model.malog(True)
    model.eval()
    with torch.no_grad():
        _ = model(data)
    log_payload = {
        "layers.malog_h": [layer.malog_h for layer in model.layers],
    }
    os.makedirs(os.path.join(root_log_dir, "RUN_0"), exist_ok=True)
    with open(os.path.join(root_log_dir, "RUN_0", "malog.pkl"), "wb") as outfile:
        pickle.dump(log_payload, outfile)
    print("MA logs in", os.path.join(root_log_dir, "RUN_0", "malog.pkl"))

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="config.json with training/model/data details")
    parser.add_argument("--gpu_id", help="gpu id to use")
    parser.add_argument("--dataset", help="dataset name (chameleon, squirrel, mutag)")
    parser.add_argument("--out_dir", help="output directory")
    parser.add_argument("--seed", help="seed")
    parser.add_argument("--epochs", help="epochs")
    parser.add_argument("--init_lr", help="initial learning rate")
    parser.add_argument("--lr_reduce_factor", help="lr reduce factor")
    parser.add_argument("--lr_schedule_patience", help="lr schedule patience")
    parser.add_argument("--min_lr", help="min lr")
    parser.add_argument("--weight_decay", help="weight decay")
    parser.add_argument("--print_epoch_interval", help="print interval")
    parser.add_argument("--max_time", help="max time (hours)")
    parser.add_argument("--hidden_dim", help="hidden dim")
    parser.add_argument("--n_layers", help="number of GAT layers")
    parser.add_argument("--n_heads", help="number of heads")
    parser.add_argument("--dropout", help="dropout")
    parser.add_argument("--attn_dropout", help="attention dropout")
    parser.add_argument("--residual", help="use residual")
    parser.add_argument("--split", help="split index (for Chameleon/Squirrel 0-9)")
    parser.add_argument("--weights", help="pkl for pre-trained model weights")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    if args.gpu_id is not None:
        config["gpu"]["id"] = int(args.gpu_id)
        config["gpu"]["use"] = True
    device = gpu_setup(config["gpu"]["use"], config["gpu"]["id"])

    DATASET_NAME = args.dataset if args.dataset is not None else config["dataset"]
    out_dir = args.out_dir if args.out_dir is not None else config["out_dir"]
    params = config["params"]
    if args.seed is not None:
        params["seed"] = int(args.seed)
    if args.epochs is not None:
        params["epochs"] = int(args.epochs)
    if args.init_lr is not None:
        params["init_lr"] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params["lr_reduce_factor"] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params["lr_schedule_patience"] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params["min_lr"] = float(args.min_lr)
    if args.weight_decay is not None:
        params["weight_decay"] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params["print_epoch_interval"] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params["max_time"] = float(args.max_time)

    params["weights"] = args.weights

    net_params = config["net_params"]
    net_params["device"] = device
    if args.hidden_dim is not None:
        net_params["hidden_dim"] = int(args.hidden_dim)
    if args.n_layers is not None:
        net_params["n_layers"] = int(args.n_layers)
    if args.n_heads is not None:
        net_params["n_heads"] = int(args.n_heads)
    if args.dropout is not None:
        net_params["dropout"] = float(args.dropout)
    if args.attn_dropout is not None:
        net_params["attn_dropout"] = float(args.attn_dropout)
    if args.residual is not None:
        net_params["residual"] = True if args.residual == "True" else False
    if args.split is not None:
        net_params["split"] = int(args.split)

    dataset = DGLPyGNodeDataset(DATASET_NAME, split=net_params.get("split", 0), raw_dir=config.get("raw_dir", "dataset_cache"))
    net_params["in_dim"] = dataset.data.x.shape[1]
    net_params["n_classes"] = dataset.num_classes
    net_params["total_param"] = view_model_param(net_params)
    MODEL_NAME = "GAT"

    logname = MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config["gpu"]["id"]) + "_" + time.strftime("%Y_%b_%d_%Hh%Mm%Ss")
    root_log_dir = os.path.join(out_dir, "logs", logname)
    root_ckpt_dir = os.path.join(out_dir, "checkpoints", logname)
    write_file_name = os.path.join(out_dir, "results", "result_" + logname)
    write_config_file = os.path.join(out_dir, "configs", "config_" + logname)
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    for path in [os.path.join(out_dir, "results"), os.path.join(out_dir, "configs")]:
        os.makedirs(path, exist_ok=True)

    with open(write_config_file + ".txt", "w") as f:
        f.write(
            f"""Dataset: {DATASET_NAME},
Model: {MODEL_NAME}

params={params}

net_params={net_params}

Total Parameters: {net_params['total_param']}
"""
        )

    if params["weights"] is None:
        train_bool = True
        epoch = 0
        weights = None
    else:
        train_bool = False
        epoch = int(params["weights"].rsplit(".", 1)[0].rsplit("_", 1)[-1])
        weights = torch.load(params["weights"])

    train_val_pipeline(params, net_params, dataset, dirs, train_bool=train_bool, weights=weights, epoch_start=epoch)

if __name__ == "__main__":
    main()
