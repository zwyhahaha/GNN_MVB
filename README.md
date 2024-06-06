# Scalable Primal Heuristics Using Graph Neural Networks for Combinatorial Optimization 

Furkan Cantürk, Taha Varol, Reyhan Aydoğan, and Okan Örsan Özener

[Publication Page](https://www.jair.org/index.php/jair/article/view/14972)

## Reproducibility
Download the files [here](https://drive.google.com/drive/folders/1zunn3_KcgXmiuvN3-y6Jihcr6QDKK1JC) and extract them to `data` folder under the project folder.

Run the following commands to create the project environment.

```
conda deactivate
conda env create -f environment.yaml
conda activate gnn4co
```

You can reproduce the experiments in the paper using `main.py`.

```
python main.py --prob_name indset --config_id 0 --solve_t 1800 --reduction_t 60
```


## Environment
Tested with Python 3.10.14, PyTorch 2.2.2, PyTorch Geometric 2.5.2, and CPLEX 22.1 on Windows and Linux platforms.

You can use `environment.yaml` to generate a conda environment for Windows platforms.

Alternatively, you can create a conda environment with the following commands.

```
conda deactivate
conda create -n gnn4co python=3.10 pytorch=2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda activate gnn4co
conda install pyg -c pyg
conda install cplex docplex pandas ipykernel torchmetrics wandb plotly openpyxl  -c conda-forge -c ibmdecisionoptimization -c plotly
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.2+cu118.html  
```

A commercial or academic CPLEX licence is needed to run the .lp / .mps instances in the project. Please refer to [CPLEX's page](https://www.ibm.com/docs/en/icos/22.1.1?topic=cplex-setting-up-python-api) to setup the Python API of CPLEX.


## Citation

If you find this repository helpful in your publications, please consider citing our paper.

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

This project partially includes coding from the [MIPGNN](https://github.com/lyeskhalil/mipGNN).
If you use our repository, please cite the following study as well.

```
 @article{Khalil2022,
author = {Khalil, Elias and Morris, Christopher and Lodi, Andrea},
year = {2022},
pages = {10219-10227},
title = {{MIP-GNN}: A Data-Driven Framework for Guiding Combinatorial Solvers},
volume = {36},
journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
doi = {10.1609/aaai.v36i9.21262}
}
```
