import random
import glob
import time
import pickle
import os
import numpy as np
import pandas as pd

from pathlib import Path
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple

import torch
from torch.optim import lr_scheduler
import cplex
from docplex.mp.model_reader import ModelReader
from docplex.mp.conflict_refiner import ConflictRefiner

from models.gnn import SimpleMIPGNN, MIPGNN
from graph_preprocessing import AbcNorm, GraphDataset
from loss import LossHandler
from nn_utils import NoamLR
from global_vars import *


def to_numpy(tensor_obj):
    return tensor_obj.cpu().detach().numpy()

def set_random_state(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed()
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def free_gpu_memory():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

def get_cplex_instance(instance_path, instance_name, interface_type = 'cplex', instance_file_type=None):
    
    if instance_file_type is None:
        mps_instance_path = str(instance_path.joinpath(instance_name)) + ".mps"
        lp_instance_path = str(instance_path.joinpath(instance_name)) + ".lp"
        
        if Path(mps_instance_path).exists():
            instance_file_path = mps_instance_path
        elif Path(lp_instance_path).exists():
            instance_file_path = lp_instance_path
        else:
            raise Exception(f"No {instance_name} in {instance_path}")
    else:
        instance_file_path = str(instance_path.joinpath(instance_name + instance_file_type)) 

    if interface_type == 'cplex':
        instance_cpx = cplex.Cplex(instance_file_path)
    else: # docplex
        instance_cpx = ModelReader.read_model(instance_file_path)
    
    return instance_cpx

def get_trained_model_config(
        prob_name: str,
        config_id: int
        ) -> pd.DataFrame:
    
    config = pd.read_excel(MODEL_DIR / f"{prob_name}_model_configs.xlsx", index_col=0).loc[config_id].T.to_dict()
    
    return config

def get_model(model_dir, var_feature_size, con_feature_size, n_batches,
              network_name,
              hidden,
              num_layers,
              dropout, 
              aggr,
              activation,
              norm,
              binary_pred,
              pred_loss_type,
              batch_size, num_epochs, lr, weight_decay, bias_threshold, edl_lambda, evidence_func,
               scheduler_step_size, gamma, scheduler_type, prenorm, abc_norm, random_state,
                **kwargs):
    
    model_params = OrderedDict(
        gnn_type = network_name,
        num_layers = num_layers,
        var_feature_size = var_feature_size, 
        con_feature_size = con_feature_size,
        hidden = hidden,
        dropout = dropout,
        aggr = aggr,
        activation = activation,
        norm = norm,
        binary_pred = binary_pred)

    train_params = OrderedDict(
        batch_size = batch_size,
        num_epochs = num_epochs,
        lr = lr,
        # weight_decay = weight_decay, # include it if it is not fixed
        # bias_threshold = bias_threshold, # include it if it is not fixed
        pred_loss_type = pred_loss_type,
        edl_lambda = edl_lambda,
        evidence_func = evidence_func,
        scheduler_type = scheduler_type,
        scheduler_step_size = scheduler_step_size,
        gamma = gamma,
        prenorm = prenorm,
        abc_norm = abc_norm,
        random_state = random_state)
    
    criterion = LossHandler(edl_lambda, evidence_func, pred_loss_type, n_steps=n_batches*num_epochs).to(DEVICE)

    if "+" in network_name:
        model = MIPGNN(**model_params)
    else:
        model = SimpleMIPGNN(**model_params)
    
    model_name = name_model(model_params, train_params)    
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  
    
    if scheduler_type == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=EPS)
    elif scheduler_type == 'noam':
        scheduler = NoamLR(optimizer, init_lr=lr, max_lr=lr*5, final_lr=lr, warmup_epochs=num_epochs//4, total_epochs=num_epochs, steps_per_epoch=n_batches)
    elif scheduler_type == 'cycle':
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr*10, pct_start=0.25, steps_per_epoch=n_batches, epochs=num_epochs)
    elif scheduler_type == 'plateau':    
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=10, min_lr=1e-6)
    else:
        raise Exception(f"No {scheduler_type} in scheduler_type options.")

    return model_name, model, criterion, optimizer, scheduler

def name_model(model_params, train_params):

    # Remove the below pop operations if these params are used or not fixed !!!
    if 'var_feature_size' in model_params:
        model_params.pop('var_feature_size') 
    if 'con_feature_size' in model_params:
        model_params.pop('con_feature_size')
    
    if train_params["scheduler_type"] != 'step':
        train_params.pop("scheduler_step_size")
        train_params.pop("gamma")

    lst = []
    for key, value in model_params.items():

        if value in [False, None]:
            continue
        if value is True:
            lst.append(str(key))
        else:
            lst.append(str(value))

    for key, value in train_params.items():

        if value in [False, None, 'nonorm']:
            continue
        if value is True:
            lst.append(str(key))
        else:
            lst.append(str(value))

    model_name =  "_".join(lst)
    
    return model_name 

# parsing a text file storing some stats about CPLEX results
def get_solver_results(solution_path):
    cols = ["file_name", "phase1_status", "phase1_gap", "phase2_status", "phase2_gap", "phase2_bestobj", "num_solutions", "total_time", "phase1_time", "phase2_time"]
    with open(solution_path.joinpath("results.log"), "r") as f:
        results = f.read().splitlines()

    logs = [line.replace(', ', "-").split(',') for line in results]
    logs = pd.DataFrame(logs, columns=cols)
    logs.index = logs.file_name.str.replace(".lp", "", regex=True).str.replace(".mps.gz", "", regex=True).str.replace(".mps", "", regex=True)

    for col in logs.columns:
        try:
            logs[col] = logs[col].astype(float)
        except:
            pass

    return logs


def get_co_datasets(prob_name: str, dataset_names: List[str], dataset_sizes: List[int], abc_norm: bool):
    
    dt_lst = []
    dtypes = np.array(['train', 'val', 'test', 'transfer'])
    
    for i, dt_name in  enumerate(dataset_names):

        dt_type = dtypes[['train' in dt_name, 'val' in dt_name, 'test' in dt_name, 'transfer' in dt_name]][0]
        graph_path = DATA_DIR.joinpath('graphs', prob_name, dt_name)
        instance_path = DATA_DIR.joinpath('instances', prob_name, dt_name)
        solution_path = DATA_DIR.joinpath('solution_pools', prob_name, dt_name)
        
        target_instances = sorted(list(set([name.replace("_data.pt", '') 
                                        for name in os.listdir(graph_path.joinpath('processed')) 
                                        if name.endswith('_data.pt')])))
                                        
        # Checking training and validation instances of setcover problem since they could be (rarely) solved suboptimally.
        if prob_name in ["setcover"] and dt_type in ['train', 'val']:
            # print(f"# {dt_name} instances:", len(target_instances))
            
            logs = get_solver_results(solution_path).loc[target_instances]
            logs = logs[logs['phase2_gap'] != -1] # masking out infeasible instances
            target_instances = sorted(list(set(logs[logs['phase2_gap'] < 0.0001].index) & set(target_instances)))
            # print(f"# {dt_name} instances solved optimally:", len(target_instances))

        size = dataset_sizes[i]
        
        target_instances = target_instances[:size]
        
        #target_instances = OrderedDict(zip(range(len(target_instances)),sorted(target_instances)))

        transform_func = AbcNorm if abc_norm else None
        
        dataset = GraphDataset(prob_name, dt_type, dt_name, instance_path, graph_path, \
                               target_instances, transform=transform_func)[:size]
        
        dt_lst.append(dataset)

        if dt_type in ['train', 'val']:
            bias = np.mean([data.y_incumbent.mean().item() for i, data in dataset])
            print(prob_name, dt_name, "| Size:", len(dataset), "Bias:", bias)

    return dt_lst