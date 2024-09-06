# Description

Official repository implementation for "[Characterizing Massive Activations of Attention Mechanism in Graph Neural Networks](https://arxiv.org/abs/2409.03463)".

This project analyzes behaviors of models taken from the following three repositories:
- [graphdeeplearning/graphtransformer](https://github.com/graphdeeplearning/graphtransformer)
- [DevinKreuzer/SAN](https://github.com/DevinKreuzer/SAN)
- [vijaydwivedi75/gnn-lspe](https://github.com/vijaydwivedi75/gnn-lspe)

which can be found in the directories [graphtransformer](./graphtransformer), [SAN](./SAN), and [gnn-lspe](./gnn-lspe), with due modifications.\
Some code is integrated from [labstructbioinf/EdgeGat](https://github.com/labstructbioinf/EdgeGat).


<br>

# Setup

The three repositories are based on [graphdeeplearning/benchmarking-gnns](https://github.com/graphdeeplearning/benchmarking-gnns), please refer to their guide to setup cuda, without installing their environment: [01_benchmark_installation.md](https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/docs/01_benchmark_installation.md).\
A conda environment is provided in [gnnma_gpu.yml](./gnnma_gpu.yml) to provide dependencies for this project.

Please refer to the three repositories to download the datasets (except for TOX21 in GraphTransformer, which is loaded using `dgllife` at runtime).


<br>

# Run experiments

The model/dataset configurations are assigned to the three repositories in the following way:
- graphtransformer: GT with ZINC and TOX21
- SAN: SAN with ZINC
- gnn-lspe: all the others (GT with PROTEINS, SAN with TOX21 and PROTEINS, GraphiT)

When running an experiments, two paths will be printed by the program just before exiting:
- MA logs: logfile containing activation values, to be used for [plots](./plot), it is suggested to store the paths in [plot/malogs.py](./plot/malogs.py) to use them for generating plots
- checkpoints: directory containing the trained model's weights, to be used for further testing or attacks.

Commands to run experiments can be found in [graphtransformer/scripts](./graphtransformer/scripts), [SAN/scripts](./SAN/scripts), and [gnn-lspe/scripts](./gnn-lspe/scripts).\
Such commands should be run from the repository's main directory (i.e., [graphtransformer](./graphtransformer), [SAN](./SAN), or [gnn-lspe](./gnn-lspe))


<br>

# Plots

Plots can be made using notebooks in [plot](./plot).\
Logs must be previously generated while running the experiments, and log paths stored in [plot/malogs.py](./plot/malogs.py).


<br>

# Attack

The attack results on GT with TOX21 can be reproduced using the commands in [graphtransformer/scripts/attack](./graphtransformer/scripts/attack) (optionally using models checkpoints produced in experiments previously ran, adding command line parameters `--epochs 0 --weights <checkpoint>`)

<br><br><br>

