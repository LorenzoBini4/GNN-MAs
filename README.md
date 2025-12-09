# Description

Official repository implementation for "[Massive Activations in Graph Neural Networks: Decoding Attention for Domain-Dependent Interpretability](https://ceur-ws.org/Vol-4059/paper2.pdf)" 
published in [Proceedings of the Second Workshop on Explainable Artificial Intelligence for the Medical Domain](https://ceur-ws.org/Vol-4059/), 
co-located with the 28th European Conference on Artificial Intelligence - ECAI 2025, volume 4059 of CEUR Workshop Proceedings, CEUR-WS.org, 
Bologna, Italy, 25-30 October 2025.

This project analyzes behaviors of models taken from the following three repositories:
- [graphdeeplearning/graphtransformer](https://github.com/graphdeeplearning/graphtransformer)
- [DevinKreuzer/SAN](https://github.com/DevinKreuzer/SAN)
- [vijaydwivedi75/gnn-lspe](https://github.com/vijaydwivedi75/gnn-lspe)

which can be found in the directories [graphtransformer](./graphtransformer), [SAN](./SAN), and [gnn-lspe](./gnn-lspe), with due modifications.\
Some code is integrated from [labstructbioinf/EdgeGat](https://github.com/labstructbioinf/EdgeGat).

Additionally, a PyTorch Geometric (PyG) GAT pipeline is provided in [pyg](./pyg) to mirror the Massive Activations logging on DGL node-feature datasets (Chameleon, Squirrel, MUTAG).


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
- pyg: PyG GAT with Chameleon, Squirrel, MUTAG (node classification)

When running an experiments, two paths will be printed by the program just before exiting:
- MA logs: logfile containing activation values, to be used for [plots](./plot), it is suggested to store the paths in [plot/logs.json](./plot/logs.json) to use them for generating plots
- checkpoints: directory containing the trained model's weights, to be used for further testing or attacks.

Commands to run experiments can be found in [graphtransformer/scripts](./graphtransformer/scripts), [SAN/scripts](./SAN/scripts), and [gnn-lspe/scripts](./gnn-lspe/scripts).\
PyG GAT scripts are under [pyg/scripts/train](./pyg/scripts/train) (e.g., `bash scripts/train/gat_chameleon.sh` from inside `pyg`).
Such commands should be run from the repository's main directory (i.e., [graphtransformer](./graphtransformer), [SAN](./SAN), or [gnn-lspe](./gnn-lspe))


<br>

# Plots

Plots can be made using notebooks in [plot](./plot).\
Logs must be previously generated while running the experiments, and log paths stored in [plot/logs.json](./plot/logs.json).


<br>

# Attack

The attack results on GT with TOX21 can be reproduced using the commands in [graphtransformer/scripts/attack](./graphtransformer/scripts/attack) (optionally using models checkpoints produced in experiments previously ran, adding command line parameters `--epochs 0 --weights <checkpoint>`)

<br>

# Citation
Please cite our [paper](https://ceur-ws.org/Vol-4059/paper2.pdf) if you use this work, thank you!
```bibtex
@inproceedings{BiniSorbiMassiveActivations2025,
	title = {Massive Activations in Graph Neural Networks: Decoding Attention for Domain-Dependent Interpretability},
	author = {Bini, Lorenzo and Sorbi, Marco and Marchand-Maillet, St{\'e}phane},
	booktitle = {Proceedings of the Second Workshop on Explainable Artificial Intelligence for the Medical Domain, co-located with the 28th European Conference on Artificial Intelligence - ECAI 2025},
    editor = {Casalino, Gabriella and Castellano, Giovanna and Kaczmarek-Majer, Katarzyna and Scaringi, Raffaele and Zaza, Gianluca},
	year = {2025},
	eventdate = {2025-10-25},
	address = {Bologna, Italy},
	volume = {4059},
	series = {CEUR Workshop Proceedings},
	publisher = {CEUR-WS.org},
	issn = {1613-0073},
	url = {https://ceur-ws.org/Vol-4059/paper2.pdf}
}
```
