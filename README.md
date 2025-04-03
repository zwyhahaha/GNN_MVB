1. obtain GNN models
We use the trained GNN models from: https://github.com/furkancanturk/gnn4co
```bash
git clone git@github.com:furkancanturk/gnn4co.git MIPGNN
```

1. prepare the data
Download the files (here)[https://drive.google.com/drive/folders/1zunn3_KcgXmiuvN3-y6Jihcr6QDKK1JC] and extract them to data folder under the project folder.
Generate the graph data by
```python
python MIPGNN/data_generation.py --prob_name indset --dt_types val
python MIPGNN/data_generation.py --prob_name indset --dt_types target
python MIPGNN/data_generation.py --prob_name setcover --dt_types val
python MIPGNN/data_generation.py --prob_name cauctions --dt_types val
```

3. install the environment
```bash
conda deactivate
conda env create -f environment.yaml
conda activate gnn4co
```

4. run an example mvb experiments
```bash
python gnn_experiments.py
```
you can see the results in the `results/` folder.
the scripts for reproducing all the experiments is in `run.sh`.

Note: use numpy=1.x to avoid certain possible issues.