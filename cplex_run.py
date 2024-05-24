import os
import argparse
import time
import random
import glob
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd

import cplex
from cplex.callbacks import MIPCallback

from global_vars import *
from utils import *

class LoggingCB(MIPCallback):
    def __init__(self, env):
        self.start_t = time.time()
        self.t = []
        self.gap = []
        self.obj = []
        self.depth = []
        super().__init__(env)

    def __call__(self):
        self.t.append(time.time())
        self.gap.append(self.get_MIP_relative_gap())
        self.obj.append(self.get_incumbent_objective_value())
        self.depth.append(self.get_current_node_depth())

    def to_df(self, solve_time, gap, obj):
        if len(self.t) > 0:
            self.t = np.array(self.t) - self.start_t 
            end_time = max(solve_time, self.t[-1] + 0.01)
            self.t = np.append(self.t, end_time)
            self.gap.append(gap)
            self.obj.append(obj)
            self.depth.append(self.depth[-1])
        
            df = pd.DataFrame(np.array([self.t, self.gap, self.obj, self.depth]).T, columns=["t", "gap", "bestobj", "depth"])
            df["t"] = df["t"].apply(np.ceil)
            df = df.groupby("t")[['gap', 'bestobj', 'depth']].mean().reset_index()
              
        else:
            df = pd.DataFrame([[np.ceil(solve_time), np.nan, np.nan, np.nan]], columns=["t", "gap", "bestobj", "depth"])

        return df

    def save_as_npz(self, npz_path, solve_time, gap, obj, skip_t=0):
        df = self.to_df(solve_time, gap, obj)

        vals = df.values.astype(np.float32)
        with open(npz_path, 'wb') as f:
            np.savez_compressed(f, log=vals)

def disable_output_cpx(instance_cpx):
    instance_cpx.set_log_stream(None)
    instance_cpx.set_warning_stream(None)
    instance_cpx.set_results_stream(None)


def solveIP(cpx_instance, timelimit, mipgap, relgap_pool, maxsols, threads, memlimit, treememlimit, cpx_tmp, do_phase1, do_phase2):

    cpx_instance.parameters.timelimit.set(timelimit)
    cpx_instance.parameters.threads.set(threads)
    cpx_instance.parameters.mip.strategy.file.set(2)
    cpx_instance.parameters.workdir.set(cpx_tmp)
    cpx_instance.parameters.mip.tolerances.mipgap.set(mipgap)
    
    phase1_gap, phase2_gap, phase2_bestobj = -1, -1, -1

    if do_phase1:
        print("Starting Phase I.")

        phase1_time = cpx_instance.get_time()
        cpx_instance.solve()
        phase1_time = round(cpx_instance.get_time() - phase1_time, 2)
        phase1_status = cpx_instance.solution.get_status_string()
        num_solutions = cpx_instance.solution.pool.get_num()
        if cpx_instance.solution.is_primal_feasible():
            phase1_gap = cpx_instance.solution.MIP.get_mip_relative_gap() 
            phase2_bestobj = cpx_instance.solution.get_objective_value()

        print("Finished Phase I.", phase1_time, phase1_status, num_solutions)

        cpx_instance.parameters.mip.tolerances.mipgap.set(min([1.0, max([phase1_gap, mipgap])]))
    else:
        cpx_instance.parameters.mip.tolerances.mipgap.set(mipgap)
        phase1_time = 0
        phase1_status, phase1_gap = None, None
        
    cpx_instance.parameters.timelimit.set(timelimit)
    # 2 = Moderate: generate a larger number of solutions
    cpx_instance.parameters.mip.pool.intensity.set(2)
    # Replace the solution which has the worst objective
    cpx_instance.parameters.mip.pool.replace.set(1)
    # Maximum number of solutions generated for the solution pool by populate
    cpx_instance.parameters.mip.limits.populate.set(maxsols)
    # Maximum pool size
    cpx_instance.parameters.mip.pool.capacity.set(maxsols)
    # Relative gap for the solution pool
    cpx_instance.parameters.mip.pool.relgap.set(relgap_pool)

    print("Starting Phase II.")
    
    phase2_time = cpx_instance.get_time()
    if do_phase2:
        cpx_instance.parameters.emphasis.mip.set(1)
        cpx_instance.parameters.workmem.set(memlimit)
    
        cpx_instance.parameters.mip.limits.treememory.set(treememlimit)
        cpx_instance.populate_solution_pool()
    phase2_time = round(cpx_instance.get_time() - phase2_time, 2)
    phase2_status = cpx_instance.solution.get_status_string()
    num_solutions = cpx_instance.solution.pool.get_num()
    if num_solutions:
        phase2_gap = cpx_instance.solution.MIP.get_mip_relative_gap()
        phase2_bestobj = cpx_instance.solution.get_objective_value()
    
    print("Finished Phase II in", phase2_time, phase2_status, num_solutions)
    
    total_time = phase1_time + phase2_time
    
    return phase1_status, phase1_gap, phase2_status, phase2_gap, phase2_bestobj, num_solutions, total_time, phase1_time, phase2_time


def search(
    cpx_instance,
    instance_name,
    out_path,
    file_name,
    timelimit=1800.0,
    threads=1,
    memlimit=2000,
    treememlimit=20000,
    mipgap=0.1,
    relgap_pool=0.1,
    maxsols=1000,
    cpx_output=False,
    cpx_tmp="cpx_tmp/",
    do_phase1=True,
    do_phase2=True,
):

    results_path = out_path.joinpath(f"{instance_name}_result.log")
    npz_path = out_path.joinpath(instance_name + "_pool.npz")
    
    # disable all cplex output
    if not cpx_output:
        disable_output_cpx(cpx_instance)
        
    result = solveIP(
                    cpx_instance,
                    timelimit, 
                    mipgap, 
                    relgap_pool, 
                    maxsols, 
                    threads, 
                    memlimit, 
                    treememlimit,
                    cpx_tmp,
                    do_phase1,
                    do_phase2)
            
    result = (file_name,) + result
    str_result = ",".join([str(x) for x in result])
    print(str_result)

    with open(results_path, "a") as results_file:
        results_file.write(str_result + "\n")      

    if result:
        num_solutions = result[-4]
        if num_solutions >= 1:
            solutions_matrix = np.zeros((num_solutions, len(cpx_instance.solution.pool.get_values(0))))
            objval_arr = np.zeros(num_solutions)
            for sol_idx in range(num_solutions):
                sol_objval = cpx_instance.solution.pool.get_objective_value(sol_idx)
                objval_arr[sol_idx] = sol_objval 
                solutions_matrix[sol_idx] = cpx_instance.solution.pool.get_values(sol_idx)
            solutions_obj_matrix = np.concatenate((np.expand_dims(objval_arr, axis=0).T, solutions_matrix), axis=1)

            with open(npz_path, 'wb') as f:
                np.savez_compressed(f, solutions=solutions_obj_matrix)
                print("Wrote npz file.")

    return result

def cplex_run(args):
    
    timelimit, file_name, prob_name, dt_name, instance_file_type, instance_path, solution_path, log_path = args

    instance_name = file_name.replace(instance_file_type, "")
    cpx_instance = cplex.Cplex(str(instance_path.joinpath(file_name)))
   

    print(">>>", file_name)
    if prob_name == 'setcover':
        print("updating obj coefs")
        var_names = cpx_instance.variables.get_names()
        seed = int(file_name.split("_")[-1].split(".")[0])
        np.random.seed(seed)
        random.seed(seed)
        cpx_instance.objective.set_linear(list(zip(var_names, np.random.randint(5, 100, len(var_names)).astype(float))))
        cpx_instance.write(str(instance_path.joinpath(file_name)))
    
    my_cb = cpx_instance.register_callback(LoggingCB)
    
    result = search(
        cpx_instance,
        instance_name,
        out_path = solution_path,
        file_name = file_name,
        timelimit=timelimit,
        threads=1,
        memlimit=20000,
        treememlimit=200000,
        mipgap=0.0,
        relgap_pool=0.001,
        maxsols=1000,
        cpx_output=False,
        cpx_tmp="cpx_tmp",
        do_phase1 = True,
        do_phase2 = False, 
    )
    
    _, phase1_status, phase1_gap, phase2_status, phase2_gap, phase2_bestobj, num_solutions, total_time, phase1_time, phase2_time = result
    my_cb.save_as_npz(log_path.joinpath(instance_name+ "_log.npz"), total_time, phase2_gap, phase2_bestobj)

def main(prob_name: str, dt_name: str, n_instances: int, timelimit: float, n_threads:int):

    instance_file_type = INSTANCE_FILE_TYPES[prob_name]

    instance_path = DATA_DIR.joinpath('instances', prob_name, dt_name)
    solution_path = DATA_DIR.joinpath('solution_pools', prob_name, dt_name)
    solution_path.mkdir(parents=True, exist_ok=True)
    log_path = solution_path.joinpath("logs")
    log_path.mkdir(parents=True, exist_ok=True)
    Path('cpx_tmp').mkdir(parents=True, exist_ok=True)

    problems = sorted(list(filter(lambda name: name.endswith(instance_file_type), os.listdir(instance_path))))[:n_instances]
    solved_probs = [name.replace('_pool.npz', instance_file_type) for name in os.listdir(solution_path) if name.endswith('_pool.npz')]
    to_solve_problems = sorted(list(set(problems) - set(solved_probs)))
    print("#all:", len(problems), "#solved:", len(solved_probs), "#to solve:", len(to_solve_problems))
    
    args = []

    for i in range(len(to_solve_problems)):
        args.append([timelimit, to_solve_problems[i], prob_name, dt_name, instance_file_type, instance_path, solution_path, log_path])

    with Pool(n_threads) as p:
        p.map(cplex_run, args)
        p.close()
    
    for log_file in glob.glob(str(solution_path.joinpath("*.log"))):
        
        with open(log_file, "r") as f:
            r = f.read()

        with open(solution_path.joinpath("results.log"), "a") as results_file:
            results_file.write(r)  

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prob_name", type=str, default='indset', choices=PROB_NAMES)
    parser.add_argument("-d", "--dt_name", type=str, default='train_1000_4')
    parser.add_argument("-n", "--n_instances", type=int, default=1000)
    parser.add_argument("-t", "--timelimit", type=float, default=60.0, help='CPLEX time limit in seconds per instance.')
    parser.add_argument("-nt", "--n_threads", type=int, default=cpu_count() // 2)
    args = parser.parse_args()

    prob_name = args.prob_name 
    dt_name = args.dt_name 
    n_instances = args.n_instances 
    
    timelimit = args.timelimit
    n_threads = args.n_threads 
 
    main(prob_name, dt_name, n_instances, timelimit, n_threads)