import sys
sys.path.append(".")
import os
import argparse
import pandas as pd
from pathlib import Path
from multiprocessing import cpu_count, Pool
from utils import get_co_datasets, get_model


import cplex_run
import data_generation
import trainer
import ml_augmented_opt
from utils import *
from global_vars import *

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob_name", type=str, default='indset', choices=PROB_NAMES)
    parser.add_argument("--config_id", type=int, default=0, help="Index of model configuration at the xlsx files under trained_models folder.")
    parser.add_argument("--solve_t", type=float, default=1800, help="Total time limit in seconds for ml_augmented_opt per instance.")
    parser.add_argument("--reduction_t", type=float, default=60, help="Total time limit in seconds for UPR per instance.")
    parser.add_argument("--strategy", type=str, default='UPR+NS', choices=['UPR+NS', 'NS'])
    parser.add_argument("--n_threads", type=int, default=N_THREADS//2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_log", type=bool, default=False)
    args = parser.parse_args()

    prob_name = args.prob_name 
    config_id = args.config_id
    train_dt_name = TRAIN_DT_NAMES[prob_name]
    val_dt_name = VAL_DT_NAMES[prob_name]
    target_dt_names_lst = TARGET_DT_NAMES[prob_name]
    dt_names = [train_dt_name, val_dt_name] + target_dt_names_lst
    solve_timelimit = args.solve_t
    reduction_timelimit = args.reduction_t
    strategy = args.strategy 
    n_threads = args.n_threads 
    seed = args.seed
    WANDB_LOG = args.wandb_log 

    config = get_trained_model_config(prob_name, config_id=config_id)
    dt_sizes = {'train':config["train_size"], 'val':config["val_size"], 'test':config["test_size"], 'transfer':config["transfer_size"]}
    config['random_state'] = seed


    # STEP 1: Collecting optimal solutions with CPLEX for the training and validation instances 
    for dt_name in [train_dt_name, val_dt_name]: 
        n_instances = dt_sizes['train'] if 'train' in dt_name else dt_sizes['val']
        
        # Some setcover instances cannot be solved optimally in a short time, 
        # so, n_instances is increased to get enough number of optimally solved instances for the training and validation datasets.
        if prob_name == 'setcover': 
            n_instances = int(n_instances * 1.5)
        
        collection_time_limit = solve_timelimit//2 # It can be also reliably set to 5 mins (for the training and validation datasets in the project)

        cplex_run.main(prob_name, dt_name, n_instances, collection_time_limit, n_threads)

    # STEP 2: Generating Graph Data objects from .mps / .lp files under instances/{prob_name}/{dt_name} folder and the solution vectors 
    data_generation.main(prob_name, dt_names, dt_sizes, False, n_threads)


    # STEP 3: Training a GNN model if the model configuration (config) is not available in trained_models folder
    val_dt = get_co_datasets(prob_name, [val_dt_name], [config['train_size']], config['abc_norm'])[0]
    data = val_dt[0][1]
    var_feature_size = data.var_node_features.size(-1) # number of features used for variable nodes in the bipartite graph representing MIP
    con_feature_size = data.con_node_features.size(-1) # number of features used for constraint nodes in the bipartite graph representing MIP

    model_dir = MODEL_DIR.joinpath(config["prob_name"], train_dt_name)
    model_name, _, _, _, _ = get_model(model_dir, var_feature_size, con_feature_size, n_batches=1, **config)

    if not model_dir.joinpath(model_name+".pt").exists():
        train_dt = get_co_datasets(prob_name, [train_dt_name], [config['val_size']], config['abc_norm'])[0]
        model = trainer.main(train_dt, val_dt, config, WANDB_LOG)
        del train_dt
        
    # STEP 4: Solving the all instances of datasets in target_dt_names_lst using the trained GNN and CPLEX
    ml_augmented_opt.experiment(
        prob_name, 
        train_dt_name,
        val_dt_name,
        target_dt_names_lst,
        config["val_size"], 
        config["test_size"],
        config["transfer_size"],
        solve_timelimit,
        reduction_timelimit,
        [config],
        [strategy],
        n_threads
    )

    # ALTERNATIVE: You can use the following code to solve only one instance

    # _, model, _ = ml_augmented_opt.load_model(config, model_dir)
    
    # for target_dt_name in target_dt_names_lst:
    #     instance_path = DATA_DIR.joinpath('instances', prob_name, target_dt_name)
    #     data_path = DATA_DIR.joinpath('graphs', prob_name, target_dt_name, 'processed')  
    #     inference_dir = PROJECT_DIR.joinpath('trained_models', config["prob_name"], train_dt_name, "inference")
    #     log_path = inference_dir.joinpath(f"{target_dt_name}_{strategy}_{model_name}")
        
    #     data_file_paths = sorted(glob.glob(str(data_path.joinpath("*_data.pt"))))
    #     target_instances = [path.split("\\")[-1].removesuffix("_data.pt") for path in data_file_paths]
    #     uncertainty_threshold, _, _, _ = ml_augmented_opt.get_uncertainty_params(config, model, val_dt, 'median')
        
    #     for instance_name in target_instances:
            
    #         instance_cpx = get_cplex_instance(instance_path, instance_name, interface_type='cplex')

    #         solution, result_df = ml_augmented_opt.opt_with_gnn(
    #             model,
    #             instance_cpx,
    #             config,
    #             data,
    #             instance_name,
    #             strategy,
    #             solve_timelimit,
    #             reduction_timelimit,
    #             uncertainty_threshold,
    #             instance_path,
    #             log_path
    #             )