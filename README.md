# Data-driven Mixed Integer Optimization through Probabilistic Multi-variable Branching

This is the code for the paper [Data-driven Mixed Integer Optimization through Probabilistic Multi-variable Branching](https://arxiv.org/abs/2305.12352). The GNN models used to predict solutions are from [gnn4co repo](https://github.com/furkancanturk/gnn4co).

## To reproduce the experiments

### Install the environment
```bash
conda deactivate
conda env create -f environment.yaml
conda activate gnn_mvb
```
**Note:** 
1. You must obtain valid licenses for both COPT and Gurobi to successfully execute this code.
2. Ensure that numpy version 1.x is installed to prevent potential runtime issues.

### Prepare the data

1. Data for experiments based on GNN models:
As shown by [gnn4co repo](https://github.com/furkancanturk/gnn4co), download the files [here](https://drive.google.com/drive/folders/1zunn3_KcgXmiuvN3-y6Jihcr6QDKK1JC) and save them in the `MIPGNN/data` folder.

Next, generate the graph data using the following commands:

```bash
python MIPGNN/data_generation.py --prob_name indset --dt_types val
python MIPGNN/data_generation.py --prob_name indset --dt_types target
python MIPGNN/data_generation.py --prob_name setcover --dt_types val
python MIPGNN/data_generation.py --prob_name cauctions --dt_types val
```

2. Data for experiments based on GNN models:
Download data [here](https://drive.google.com/drive/folders/1JuCc1TpbGkERCjfAlzBltGBzFyrYYDku?usp=sharing) and save them in the `logistics_experiments/data` folder.

### Run the scripts
1. For experiments based on logistic regression models:

```bash
python gnn_experiments.py
```

If the script runs successfully, the results will be saved in the `results/` folder.

The script for reproducing all the experiments is `run.sh`.

1. For experiments based on logistic regression models:
```bash
cd logistics_experiments
python run_logistic_scuc.py
```
