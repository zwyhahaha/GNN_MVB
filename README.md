# Data-driven Mixed Integer Optimization through Probabilistic Multi-variable Branching

This is the code for the paper [Data-driven Mixed Integer Optimization through Probabilistic Multi-variable Branching](https://arxiv.org/abs/2305.12352). The GNN models used to predict solutions are from [gnn4co repo](https://github.com/furkancanturk/gnn4co).

## To reproduce the experiments

### Install the environment
```bash
conda deactivate
conda env create -f environment.yaml
conda activate gnn_mvb
```
**Note:** use numpy=1.x to avoid potential runtime issues.

### Prepare the data
As shown by [gnn4co repo](https://github.com/furkancanturk/gnn4co), download the files [here](https://drive.google.com/drive/folders/1zunn3_KcgXmiuvN3-y6Jihcr6QDKK1JC) and save them in the `MIPGNN/data` folder.

Then further generate the graph data by

```bash
python MIPGNN/data_generation.py --prob_name indset --dt_types val
python MIPGNN/data_generation.py --prob_name indset --dt_types target
python MIPGNN/data_generation.py --prob_name setcover --dt_types val
python MIPGNN/data_generation.py --prob_name cauctions --dt_types val
```

### Run the scripts
```bash
python gnn_experiments.py
```
If runs successfully, you can see the results in the `results/` folder.

The scripts for reproducing all the experiments is in `run.sh`.

## Contributors
- Wenzhi Gao, gwz@stanford.edu
- Wanyu Zhang, wanyuzhang@stu.sufe.edu.cn

## Citation

If you use this repository in your research, please cite the following papers:

```
@article{chen2023pre,
  title={Pre-trained mixed integer optimization through multi-variable cardinality branching},
  author={Chen, Yanguang and Gao, Wenzhi and Ge, Dongdong and Ye, Yinyu},
  journal={arXiv preprint arXiv:2305.12352},
  year={2023}
}
```

The GNN model used in this repository is from [gnn4co](https://github.com/furkancanturk/gnn4co):
```
@article{canturk2024,
    author = {Cantürk, Furkan and Varol, Taha and Aydoğan, Reyhan, and Özener, Okan Örsan},
    year = {2024},
    month = {6},
    pages = {327-376},
    journal = {Journal of Artificial Intelligence Research},
    volume = {80},
    title = {Scalable Primal Heuristics Using Graph Neural Networks for Combinatorial Optimization},
    doi = {10.1613/jair.1.14972}
}
```