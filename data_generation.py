import os
import glob
from pathlib import Path
import cplex
import networkx as nx
import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count
import time
import argparse
from typing import List, Dict
import pickle

from global_vars import *
from utils import *
from graph_preprocessing import *


def get_instance_names(instance_path, instance_file_type):

    problem_names = sorted([
        name.replace(instance_file_type, '')
        for name in os.listdir(instance_path) 
        if name.endswith(instance_file_type)
        ])
    
    print("#Available problems:", len(problem_names))

    return problem_names

def get_solved_instance_names(solution_path):

    solved_probs = sorted([
        name.replace('_pool.npz', '') 
        for name in os.listdir(solution_path) 
        if name.endswith('_pool.npz')
        ])
    
    print("#Solved problems:", len(solved_probs))

    return solved_probs

def presolve(cpx_instance, instance_name, instance_path):
    cpx_instance.set_log_stream(None)
    cpx_instance.set_results_stream(None)

    presolved_model_name = str(instance_path.joinpath(instance_name + ".pre"))
    
    presolved = False
    ptime = time.time()

    for method_id in [1,2,4]: #1:primal, 2:dual, 4:barrier
        
        cpx_instance.presolve.presolve(method_id)
        
        try: #error handling: cpx_instance.presolve.get_status() produces error if cpx_instance could not be presolved
            if cpx_instance.presolve.get_status() == cpx_instance.presolve.status.has_problem:
                cpx_instance.presolve.write(presolved_model_name)
                pre_cpx_instance = cplex.Cplex(presolved_model_name)
                presolved = method_id
                break
        except Exception as e:
            print("Cplex Exception:", e)
    
    if presolved:

        varnames_diff = len(set(pre_cpx_instance.variables.get_names()) - set(cpx_instance.variables.get_names()))
        
        if varnames_diff > 0:
            with open(instance_path.joinpath('presolve_varnames_diff.txt'), 'a+') as f:
                f.write(",".join([instance_name, str(presolved), str(varnames_diff)]) +"\n")

            print("Varnames difference in:", instance_name, str(presolved), str(varnames_diff))
            
            result_cpx_instance = cpx_instance
        else:
            result_cpx_instance = pre_cpx_instance
    
    else:
        result_cpx_instance = cpx_instance
    
    num_vars, num_cons = cpx_instance.variables.get_num(), cpx_instance.linear_constraints.get_num()
    r_num_vars, r_num_cons = result_cpx_instance.variables.get_num(), result_cpx_instance.linear_constraints.get_num()
    
    presolve_stats = ",".join([instance_name, str(num_vars), str(num_cons), str(r_num_vars), str(r_num_cons), str(presolved)])
    with open(instance_path.joinpath('presolve_stats.txt'), 'a+') as f:
        f.write(presolve_stats +"\n")
        print(instance_name, "num var reduction:", r_num_vars/num_vars, "num con reduction:", r_num_cons/num_cons, "presolve type:", presolved)
    
    print("Presolve time:", time.time()-ptime)

    return result_cpx_instance

def get_presolve_stats(instance_path):
    
    # Reduction rate in number of variables and constraints after presolve
    with open(instance_path.joinpath('presolve_stats.txt'), 'r') as f:
        presolve_stats = f.read().splitlines()
    
    presolve_stats = pd.DataFrame(presolve_stats)[0].str.split(',', expand=True).replace('None', np.nan)
    presolve_stats.dropna(axis=0, inplace=True)
    presolve_stats.drop_duplicates(inplace=True)
    presolve_stats.columns = ["instance_name", "num_vars", "num_cons", "reduced_num_vars", "reduced_num_cons", "presolve_method"]
    presolve_stats.set_index('instance_name', inplace=True)
    presolve_stats = presolve_stats.astype(float)
    presolve_stats['num_vars_reduction'] = presolve_stats["reduced_num_vars"]/presolve_stats["num_vars"]
    presolve_stats['num_cons_reduction'] = presolve_stats["reduced_num_cons"]/presolve_stats["num_cons"]
    print(presolve_stats[['num_vars_reduction', 'num_cons_reduction']].mean())
    
    return presolve_stats

def cpx_to_graph_data(instance_name, instance_file_type, instance_path, get_presolved=True):

    model = cplex.Cplex(str(instance_path.joinpath(instance_name + instance_file_type))) 
    process_time = time.time()
    
    if get_presolved:
        model = presolve(model, instance_name, instance_path)
 
    graph, _ = get_bipartite_graph(model)
    data = create_data_object(instance_name, model, graph, is_labeled=False, save_dir='', process_time=process_time)

    print(f"{instance_name} preprocessing completed in {time.time()-process_time} secs.")

    return model, graph, data

def create_graph_and_data_object(args):
 
    prob_name, dt_type, instance_name, instance_file_type, instance_path, solution_path, graph_path, write_graph = args

    graph_file_name = instance_name + "_labeled_graph.gpickle"
    data_file_name = instance_name + "_data.pt"
    solution_file_path = solution_path.joinpath(instance_name + "_pool.npz")
    data_path = graph_path.joinpath('processed')
    
    if data_file_name in os.listdir(data_path):
        # print(data_file_name, 'available')
        return

    model = cplex.Cplex(str(instance_path.joinpath(instance_name + instance_file_type))) 
            
    # check setcover instance has been modified
    if prob_name == 'setcover': 
        assert min(model.objective.get_linear()) >= 5

    preprocess_start_time = time.time()

    is_labeled = False
    graph, _ = get_bipartite_graph(model)

    if dt_type in ['train', 'val']:

        if Path(solution_file_path).exists():
            
            graph = get_labeled_graph(graph, instance_name, model, solution_path)
            is_labeled = True

        else:
            raise Exception(">> Solution pool for",  prob_name, dt_type, instance_name, "is not available.")

    _ = create_data_object(instance_name, model, graph, is_labeled, str(data_path), preprocess_start_time)

    if write_graph:
        with open(graph_path.joinpath(graph_file_name),  'wb') as f:
            pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)

def main(prob_name: str, 
         dt_names: List[str], 
         dt_sizes: Dict[str, int], 
         write_graph: bool = False,
         n_threads: int = cpu_count()
         ) -> None:
    
    instance_file_type = INSTANCE_FILE_TYPES[prob_name]
    dtypes = np.array(['train', 'val', 'test', 'transfer'])
    task_args = []

    for i, dt_name in enumerate(dt_names):
        
        dt_type = dtypes[['train' in dt_name, 'val' in dt_name, 'test' in dt_name, 'transfer' in dt_name]][0]
        instance_path = DATA_DIR.joinpath('instances', prob_name, dt_name)
        solution_path = DATA_DIR.joinpath('solution_pools', prob_name, dt_name)
        graph_path = DATA_DIR.joinpath('graphs', prob_name, dt_name)
        data_path = graph_path.joinpath('processed')
        
        Path(graph_path).mkdir(parents=True, exist_ok=True)
        Path(data_path).mkdir(parents=True, exist_ok=True)

        # if "train" in dt_name or 'val' in dt_name:
        #     instance_names = get_solved_instance_names(solution_path)[:dt_sizes[dt_type]]
          
        # else:
        instance_names = get_instance_names(instance_path, instance_file_type)

        already_created_data_obj_names = list(set([path_name.split("\\")[-1].replace("_data.pt", '') for path_name in glob.glob(str(data_path.joinpath("*_data.pt")))]))
        to_create = list(set(instance_names) - set(already_created_data_obj_names))
   
        for instance_name in to_create:    
            task_args.append([prob_name, dt_type, instance_name, instance_file_type, instance_path, solution_path, graph_path, write_graph])
        
    if len(task_args) > 0:
        
        print(f">> {len(task_args)} data objects in total to be created for {prob_name} {dt_names}...")

        with Pool(n_threads) as pool:
            pool.map(create_graph_and_data_object, task_args)
            pool.close()

            
if __name__ == '__main__':
    
    PROB_NAMES = list(TRAIN_DT_NAMES.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("--prob_name", type=str, default = 'cauctions', choices=PROB_NAMES)
    parser.add_argument("--dt_types", type=lambda arg: arg.split("+"), nargs='?', default = 'val', help='Values separated by "+"')
    parser.add_argument("--n_threads", type=int, default=cpu_count())
    args = parser.parse_args()

    prob_name = args.prob_name
    n_threads = args.n_threads 
    dt_sizes = {'train':1000, 'val':200, 'test':50, 'target':50}
    dt_names = []
    
    if 'train' in args.dt_types:
        dt_names.append(TRAIN_DT_NAMES[prob_name])
    if 'val' in args.dt_types:
        dt_names.append(VAL_DT_NAMES[prob_name])
    if 'target' in args.dt_types:
        dt_names = TARGET_DT_NAMES[prob_name]

    main(prob_name, dt_names, dt_sizes, write_graph=False, n_threads=cpu_count())
