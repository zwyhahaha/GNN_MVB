import os
import glob
import heapq
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from copy import deepcopy
import numpy as np
from sklearn.model_selection import ParameterGrid

import torch
import cplex
from cplex.callbacks import NodeCallback, BranchCallback
from cplex.exceptions import CplexSolverError
import sys
import os
sys.path.append("./MIPGNN")
from global_vars import *
from utils import *
from loss import *
from graph_preprocessing import *
from models.gnn import BaseModel
from cplex_run import LoggingCB

def load_model(config, model_dir, data=None):

    if not data is None:
        if type(data) is GraphDataset:
            data = data[0][1]

        var_feature_size = data.var_node_features.size(-1)
        con_feature_size = data.con_node_features.size(-1)

    else:
        var_feature_size, con_feature_size = 5, 5

    model_name, model, criterion, _, _ = get_model(model_dir, var_feature_size, con_feature_size, n_batches=1, **config)
    model.to(DEVICE)
    state = torch.load(str(model_dir.joinpath(model_name+".pt")), map_location=DEVICE)
    model.load_state_dict(state)

    return model_name, model, criterion

def scoring(loss, loss_tuples, bias_tuples, pred, y, bias_threshold, step_type):
    
    pred = pred.round().long()
    #y = y.round().long()
  
    bias_tuples = np.mean(np.array(bias_tuples), 0)
    true_bias, pred_bias, bias_error = bias_tuples
    loss_tuples = np.mean(np.array(loss_tuples), 0)

    metric_names = [ "loss", "acc", "f1", "precision", "recall", "pred_loss", "true_bias", 'pred_bias', 'bias_error']
    metrics = loss.item(),  acc(pred, y).item(), f1(pred, y).item(), pr(pred, y).item(), re(pred, y).item(), true_bias, pred_bias, bias_error
    scores = dict(zip(metric_names, metrics))

    return scores

def get_uncertainty(outputs, num_classes=2, evidence_func_name='softplus'):

    if evidence_func_name in evidence_funcs:
        evidence_func = evidence_funcs[evidence_func_name]
        evidence = evidence_func(outputs)
        alpha = evidence + 1
        uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)

    else:
        probs = torch.softmax(outputs, dim=1)
        uncertainty = 1 - torch.max(probs, dim=1)

    return uncertainty, evidence

def get_prediction(config, model, data):

    model.eval()

    with torch.no_grad():
        output = model(data) # .to(DEVICE)

    binary_mask = to_numpy(data.is_binary).squeeze()
    binary_idx = np.arange(binary_mask.shape[0])[binary_mask]

    uncertainty, evidence = get_uncertainty(output, 2, config['evidence_func'])
    
    probs = torch.softmax(output, axis=1)
    prediction = probs[:,1].clone()
    #prediction = (data.ub - data.lb) * prediction.view(-1) + data.lb # open it for predictions of integer decision values 
    probs = to_numpy(probs)
    prediction = to_numpy(prediction).squeeze()
    prediction[binary_mask] = prediction[binary_mask].round() 
    uncertainty = to_numpy(uncertainty).squeeze()
    evidence = to_numpy(evidence).squeeze()

    incumbent = None 

    if 'y_incumbent' in data:
        incumbent = (data.ub - data.lb) * data.y_incumbent + data.lb
        incumbent = to_numpy(incumbent).squeeze()
        assert prediction.shape[0] == incumbent.shape[0]

    if binary_mask.sum() != len(binary_mask): # i.e., not all vars are binary
        prediction = prediction[binary_mask]
        if 'y_incumbent' in data:
            incumbent = incumbent[binary_mask]

    assert uncertainty.shape[0] == prediction.shape[0] == sum(binary_mask)

    return probs, prediction, uncertainty, evidence, incumbent, binary_idx

def get_uncertainty_params(config, model, dt, threshold_type = 'median'):
    
    val_u_mean_lst = []
    val_confident_acc_lst = []
    val_bound_err_lst = []
    val_confident_ratio = []
        
    for graph_idx, data in tqdm(dt):
        
        if config['abc_norm']:
            data = AbcNorm(data)

        probs, pred, uncertainty, evidence, target, binary_idx = get_prediction(config, model, data)
      
        correct_pred_uncertainty = np.median(uncertainty[target==pred]) if threshold_type == 'median' else np.mean(uncertainty[target==pred])
        val_u_mean_lst.append(correct_pred_uncertainty)
        is_confident = (uncertainty <= correct_pred_uncertainty).ravel()
        val_confident_ratio.append(is_confident.mean())
        confident_acc = (pred[is_confident] == target[is_confident]).mean()
        val_confident_acc_lst.append(confident_acc)
        pred_sum = pred.sum()
        incumbent_sum = target.sum()
        val_bound_err_lst.append(incumbent_sum - pred_sum)

    val_u_mean = np.mean(val_u_mean_lst)
    val_confident_ratio_mean = np.mean(val_confident_ratio)
    val_confident_acc_mean = np.mean(val_confident_acc_lst)
    val_bound_error_stats = pd.Series(val_bound_err_lst)
   
    print("val_u_mean:", val_u_mean, "val_confident_ratio_mean:", val_confident_ratio_mean, "val_confident_acc_mean:", val_confident_acc_mean)
    print("val_bound_error_stats:")
    print(val_bound_error_stats.describe())
    val_bound_error_stats = val_bound_error_stats.describe()[['min', 'mean', 'max', 'std']]
   
    return val_u_mean, val_confident_ratio_mean, val_confident_acc_mean, val_bound_error_stats

def get_uncertainty_metrics(data, pred, uncertainty, target, threshold_type='median'):
    
    correct_pred_uncertainty = np.median(uncertainty[target==pred]) if threshold_type == 'median' else np.mean(uncertainty[target==pred])
    incorrect_pred_uncertainty = np.median(uncertainty[target!=pred]) if threshold_type == 'median' else np.mean(uncertainty[target!=pred])
 
    is_confident = (uncertainty <= correct_pred_uncertainty).ravel()
    confident_ratio = is_confident.mean()
    confident_acc = (pred[is_confident] == target[is_confident]).mean()

    return correct_pred_uncertainty, incorrect_pred_uncertainty, confident_ratio, confident_acc

def cplex_solve(instance_cpx, timelimit, n_threads):
    
    instance_cpx.set_log_stream(None)
    instance_cpx.set_warning_stream(None)
    instance_cpx.set_results_stream(None)
    instance_cpx.set_error_stream(None)
    instance_cpx.parameters.mip.display.set(0)
    instance_cpx.parameters.timelimit.set(timelimit)
    instance_cpx.parameters.threads.set(1)
    #instance_cpx.parameters.workmem.set(2000)
    instance_cpx.parameters.mip.strategy.file.set(2)
    instance_cpx.parameters.randomseed.set(0)

    st = time.time()
    instance_cpx.solve()
    solve_time = time.time() - st

    status = instance_cpx.solution.get_status_string()
    gap, bestobj, sol = np.nan, np.nan, None

    if instance_cpx.solution.is_primal_feasible():
        gap = instance_cpx.solution.MIP.get_mip_relative_gap()
        bestobj = instance_cpx.solution.get_objective_value()
        sol = np.array(instance_cpx.solution.get_values())

    return sol, solve_time, status, gap, bestobj

def reduction_then_solve(instance_cpx, timelimit, prediction, confident_idx, binary_idx):
    
    if len(confident_idx) > 0:
        assignments = list(map(lambda ind: cplex.SparsePair([int(ind)], [1]), binary_idx[confident_idx]))
        instance_cpx.linear_constraints.add(
            assignments,
            senses = ['E'] * len(confident_idx),
            rhs = prediction[confident_idx].tolist(),
            names = [str(i) + "_fixed_val" for i in confident_idx]
        )
    
    log_cb = instance_cpx.register_callback(LoggingCB)
    
    sol, solve_time, status, gap, bestobj = cplex_solve(instance_cpx, timelimit, n_threads=1)
    
    result = dict(reduction_solve_time=solve_time, reduction_status=status, reduction_gap=gap, reduction_bestobj=bestobj)
    
    log_df = log_cb.to_df(solve_time, gap, bestobj)
    del log_cb

    return sol, result, log_df

def uncertainty_based_reduction(instance_cpx, solve_timelimit, prediction, binary_idx, uncertainty, min_unc_threshold, n_iter):
    
    # numerical precision of min_unc_threshold might be problematic for some of problems (especially setcover) and models,
    #  so, r is clipped to be in (0.4, 0.55) 
    i = 0
    unc_t = min_unc_threshold 
    while True:
        confident_idx_ = get_confident_pred_idx(uncertainty, prediction, unc_t, binary_idx)
        r = len(confident_idx_) / len(binary_idx)
        if r > 0.55:
            unc_t -= 0.00001
        elif r < 0.4:
            unc_t += 0.00001
        else:
            break
        i += 1

    min_q = sum(uncertainty <= unc_t)/len(uncertainty)
    max_q = 1.0
    dq = (max_q - min_q) / (n_iter - 1)

    iter_results = []
    previous_cuts = []
    log_df_lst = []

    for i in range(n_iter):
      
        q = max_q - dq*i
        unc_t = np.quantile(uncertainty, q)

        confident_idx = get_confident_pred_idx(uncertainty, prediction, unc_t, binary_idx)
        current_cuts = [str(i) + "_fixed_val" for i in confident_idx]
        to_delete_cut = list(set(previous_cuts) - set(current_cuts))
        to_add_idx = confident_idx if i == 0 else []

        instance_cpx.linear_constraints.delete(to_delete_cut)
        reduction_time_limit = solve_timelimit if i == n_iter - 1 else solve_timelimit/2 
        
        sol, result, log_df = reduction_then_solve(instance_cpx, reduction_time_limit, prediction, to_add_idx, binary_idx)
        log_df_lst.append(log_df)
     
        iter_result = { f"{i+1}_" + key: val  for key, val in result.items()}
        iter_result[f"{i+1}_reduction_rate"] = len(confident_idx) / len(binary_idx)
        iter_results.append(iter_result)

        previous_cuts = current_cuts.copy()
    
    pprint(iter_results)

    return sol, result, iter_results, log_df_lst

def save_reduction_log_df(log_df_lst, log_path, instance_name):
    
    for i, df in enumerate(log_df_lst):
        if i > 0:
            df["t"] += log_df_lst[i-1].iloc[-1]["t"]
    
    log_df = pd.concat(log_df_lst)
    log_df.to_csv(log_path.joinpath(instance_name+ "_reduction_log.csv"), index=False)

def node_selection(instance_cpx, sol, timelimit, prediction, binary_idx, var_scores):

    if not sol is None:
        binary_sol = sol[binary_idx]
        binary_idx, binary_sol = binary_idx.astype(int).tolist(), binary_sol.astype(int).tolist()
    
        var_names = instance_cpx.variables.get_names()
        assignment = [var_names, binary_sol]
        instance_cpx.MIP_starts.add(assignment, 2)

    branch_cb = instance_cpx.register_callback(AttachDataCB)
    node_cb = instance_cpx.register_callback(NodeSelectionCB)

    branch_cb.scores = var_scores
    
    node_cb.last_best = 0
    node_cb.freq_best = 100

    node_priority = []
    branch_cb.node_priority = node_priority
    node_cb.node_priority = node_priority            

    branch_cb.time = 0
    node_cb.time = 0

    sol, solve_time, status, gap, bestobj = cplex_solve(instance_cpx, timelimit, n_threads=1)
    result = dict(ns_solve_time=solve_time, ns_status=status, ns_gap=gap, ns_bestobj=bestobj)

    return sol, result

class NodeSelectionCB(NodeCallback):

    def __call__(self):
        if self.get_num_nodes() == 0:
            return

        self.last_best += 1
        if self.freq_best > 0 and self.last_best % self.freq_best == 0:
            return

        while True:
            try:
                best_score, best_node = heapq.heappop(self.node_priority)
                node_idx = self.get_node_number((best_node,))
            except:# CplexSolverError as cse:
                continue
            break

        self.select_node(node_idx)

class AttachDataCB(BranchCallback):

    def __call__(self):
        time_start = time.time()
 
        nodesel_score = 0.0

        if self.get_num_branches() > 0:
            _, var_info = self.get_branch(0)
            branching_var_idx = var_info[0][0]

        if self.get_num_nodes() > 0:
            nodesel_score = self.get_node_data()


        for branch_idx in range(self.get_num_branches()):
            node_estimate, var_info = self.get_branch(branch_idx)
     
            var_idx = var_info[0][0]
            var_val = var_info[0][2]

            var_score = self.scores[var_idx, int(var_val)]
            nodesel_score_child = nodesel_score + var_score

            node_seqnum = self.make_cplex_branch(branch_idx, node_data=nodesel_score_child)
            
            heapq.heappush(self.node_priority, (-nodesel_score_child, node_seqnum))

        self.time += time.time() - time_start

def repair(instance_path, instance_name, prediction, binary_idx, confident_idx, repair_timelimit):
             
    print("REPAIR...")
    feasible_confident_idx = binary_idx[confident_idx]

    docpx_instance =  get_cplex_instance(instance_path, instance_name, interface_type='docplex')
    docpx_instance.add_constraints_([docpx_instance.get_var_by_index(i) == prediction[i] for i in feasible_confident_idx], [str(i) + "_fixed_val" for i in feasible_confident_idx])
    
    repair_time = 0
    repair_iter = 0
    is_repaired = False
    print(">>> # Cons with fixed val cuts:", docpx_instance.number_of_constraints)

    while repair_timelimit > 0 and len(feasible_confident_idx) > 0 :

        rt = time.time()
        cr_result = ConflictRefiner().refine_conflict(docpx_instance, display=True, time_limit=repair_timelimit)
        rt = time.time() - rt
        repair_timelimit -= rt
        repair_time += rt

        if cr_result.number_of_conflicts == 0:
            is_repaired = True
            break

        conflicting_con_names = cr_result.as_output_table()['Name'].values
        conflicting_var_idx = set()
        for cname in conflicting_con_names:
            constraint = docpx_instance.get_constraint_by_name(cname)
            conflicting_var_idx.update(set([var.index for var in  constraint.iter_variables()]))

        infeasible_fixed_var_idx = set(feasible_confident_idx) & conflicting_var_idx
        #print(cr_result.as_output_table())
        #infeasible_fixed_var_idx = [int(c.replace("_fixed_val", '')) for c in conflicting_con_names if 'fixed_val' in c]
        
        print(f"Repair iter-{repair_iter}: {len(infeasible_fixed_var_idx)} infeasible fixed vars...")
        print(infeasible_fixed_var_idx)
 
        docpx_instance.remove_constraints(
            [
                docpx_instance.get_constraint_by_name(str(i) + "_fixed_val")
                for i in infeasible_fixed_var_idx
            ]
        )

        print(f"# Cons after removing {len(infeasible_fixed_var_idx)} fixed val cuts:", docpx_instance.number_of_constraints)
        feasible_confident_idx = list(set(feasible_confident_idx) - set(infeasible_fixed_var_idx))

        repair_iter += 1

    if len(feasible_confident_idx)  == 0:
        is_repaired = True
     
    feasible_confident_idx = [list(binary_idx).index(i) for i in feasible_confident_idx]

    return is_repaired, feasible_confident_idx, repair_time, repair_iter

def solve_with_warmstart(instance_cpx, timelimit, binary_idx, sol):

    binary_sol = sol[binary_idx]
    binary_idx, binary_sol = binary_idx.astype(int).tolist(), binary_sol.astype(int).tolist()
   
    var_names = instance_cpx.variables.get_names()
    assignment = [var_names, binary_sol]
    instance_cpx.MIP_starts.add(assignment, 2)

    sol, solve_time, status, gap, bestobj = cplex_solve(instance_cpx, timelimit, n_threads=1)
    result = dict(iter3_solve_time=solve_time, iter3_status=status, iter3_gap=gap, iter3_bestobj=bestobj)
    
    return sol, result

def get_confident_pred_idx(uncertainty, prediction, uncertainty_threshold, binary_idx):
    confident_mask = uncertainty <= uncertainty_threshold
    confident_idx = list(np.arange(len(prediction), dtype=int)[confident_mask])
    return sorted(confident_idx) #binary_idx[confident_idx]


def get_confident_pred_acc_coverage(decisions, actuals, uncertainties, thresholds, dataset):
    
    confident_acc = []
    confident_coverage = []
    
    for t in thresholds:
    
        acc = []
        coverage = []

        for graph_idx, data in dataset:
            instance_name = data.instance_name
            is_binary = data.is_binary.numpy()

            prediction = decisions[instance_name][is_binary]
            u = uncertainties[instance_name][is_binary]
            target = actuals[instance_name][is_binary]
    
            is_confident = (u <= t).ravel()
            acc_ = (prediction[is_confident] == target[is_confident]).mean()
            acc.append(acc_)
            coverage.append(is_confident.mean())

        confident_acc.append(np.mean(acc))
        confident_coverage.append(np.mean(coverage))
    
    return confident_acc, confident_coverage


def opt_with_gnn(
        model,
        instance_cpx,
        config,
        data,
        instance_name,
        strategy,
        timelimit,
        reduction_timelimit,
        uncertainty_threshold,
        instance_path,
        log_path,
        ):

    pred_time = time.time()
    
    if config['abc_norm']:
        data = AbcNorm(data)

    probs, prediction, uncertainty, evidence, _, binary_idx = get_prediction(config, model, data)

    pred_time = time.time() - pred_time

    preprocess_time = data.process_time
    solve_timelimit = timelimit - preprocess_time - pred_time
    total_solve_time = time.time()
    repair_time = 0
    repair_iter = 0

    result = {}    
    result['preprocess_time'] = preprocess_time
    result['pred_time'] = pred_time

    confident_idx = get_confident_pred_idx(uncertainty, prediction, uncertainty_threshold, binary_idx)
    solution = None
    
    if 'R' in strategy:

        rst = time.time()
        log_df_lst = []

        if 'UPR' in strategy:
            print(f">>> Starting UPR...")
            solution, reduction_result, iter_results, log_df_lst = uncertainty_based_reduction(instance_cpx, reduction_timelimit, prediction, binary_idx, uncertainty, uncertainty_threshold, n_iter=5)
            print(f">>> {instance_name}: UPR completed.")
            for iter_result in iter_results:
                result.update(iter_result)
        else:                  
            solution, reduction_result, log_df = reduction_then_solve(instance_cpx, reduction_timelimit, prediction, confident_idx, binary_idx)
            log_df_lst.append(log_df)
         
        is_infeasible_reduction = 'infeasible' in reduction_result['reduction_status']
        feasible_confident_idx = confident_idx.copy()

        if is_infeasible_reduction:
            repair_timelimit = reduction_timelimit
            print(f">>> {instance_name}: Starting repair...")
            is_repaired, feasible_confident_idx, repair_time, repair_iter = repair(instance_path, instance_name, prediction, binary_idx, confident_idx, repair_timelimit)

            if is_repaired:
                instance_cpx = get_cplex_instance(instance_path, instance_name, interface_type='cplex')
                solution, reduction_result, log_df = reduction_then_solve(instance_cpx, reduction_timelimit, prediction, feasible_confident_idx, binary_idx)               
                log_df_lst.append(log_df)
            else:
                print(f">>> {instance_name}: The reduced problem could not be repaired!!!")

        rst = time.time() - rst
        solve_timelimit -= rst

        save_reduction_log_df(log_df_lst, log_path, instance_name)
        
        result.update(reduction_result)
        result['repair_time'] = repair_time
        result['repair_iter'] = repair_iter
        result['is_infeasible_reduction'] = is_infeasible_reduction
        result["reduction_rate"] = len(confident_idx) / result["num_variables"]
        result["reduction_rate_after_repair"] = len(feasible_confident_idx)/result["num_variables"]
        
        #print("Reduction result:", result)
 
    if 'NS' in strategy:

        if 'UPR' in strategy:

            # Deleting the cuts added by UPR so that the original problem instance is restored for global search with NS.
            con_names = instance_cpx.linear_constraints.get_names()
            reduction_cuts = [c for c in con_names if 'fixed_val' in c]
            instance_cpx.linear_constraints.delete(reduction_cuts)
        
        var_scores = probs if config['pred_loss_type'] == 'bce' else 1 - (1 / (1+evidence))

        cb = instance_cpx.register_callback(LoggingCB)
        print(f">>> {instance_name}: Starting NS...")
        solution, ns_result = node_selection(instance_cpx, solution, solve_timelimit, prediction, binary_idx, var_scores)    

        result.update(ns_result)    

        cb.save_as_npz(str(log_path.joinpath(instance_name+ "_log.npz")), ns_result['ns_solve_time'], ns_result['ns_gap'], ns_result['ns_bestobj'])

    print(result)
    
    total_solve_time = time.time() - total_solve_time
    result['total_solve_time'] = total_solve_time
    result_df = pd.DataFrame.from_dict([result])

    return solution, result_df

def experiment_opt_with_gnn(args):

    instance_name, model, data_path, instance_path, log_path, config, strategy, timelimit, reduction_timelimit, train_dt_name, \
    val_u_mean, val_confident_ratio_mean, val_confident_acc_mean, val_bound_error_stats = args
   
    val_bound_err_min, val_bound_err_mean, val_bound_err_max, val_bound_err_std = val_bound_error_stats

    result_df_file_path = log_path.joinpath(f"{instance_path}.csv")

    if Path(result_df_file_path).exists():
        result_df = pd.read_csv(str(result_df_file_path))
        return result_df
    
    data = torch.load(str(data_path.joinpath(instance_name + "_data.pt")), map_location=DEVICE)
    instance_name = data.instance_name

    print(">>> Reading:", instance_name)
    instance_cpx = get_cplex_instance(instance_path, instance_name, interface_type='cplex')

    num_vars = instance_cpx.variables.get_num()
    num_cons = instance_cpx.linear_constraints.get_num()
    assert num_vars == data.num_var_nodes.item()
    assert num_cons == data.num_con_nodes.item()

    if model is None:
        model_dir = PROJECT_DIR.joinpath('trained_models', config["prob_name"], train_dt_name)
        _, model, _ = load_model(config, model_dir, data)

    pred_time = time.time()
    
    if config['abc_norm']:
        data = AbcNorm(data)

    probs, prediction, uncertainty, evidence, incumbent, binary_idx = get_prediction(config, model, data)

    pred_time = time.time() - pred_time
  
    preprocess_time = data.process_time
    solve_timelimit = timelimit - preprocess_time - pred_time
    total_solve_time = time.time()
    repair_time = 0
    repair_iter = 0

    result = {}    
    result['instance_name'] = instance_name
    result["num_variables"] = num_vars
    result["num_constraints"] = num_cons
    result['preprocess_time'] = preprocess_time
    result['pred_time'] = pred_time

    uncertainty_threshold = val_u_mean
    confident_idx = get_confident_pred_idx(uncertainty, prediction, uncertainty_threshold, binary_idx)
    sol = None
    
    # if 'R' in strategy:

    #     rst = time.time()
    #     log_df_lst = []

    #     if 'UPR' in strategy:
    #         print(f">>> {instance_name}: Starting UPR...")
    #         sol, reduction_result, iter_results, log_df_lst = uncertainty_based_reduction(instance_cpx, reduction_timelimit, prediction, binary_idx, uncertainty, uncertainty_threshold, n_iter=5)
    #         print(f">>> {instance_name}: UPR completed.")
    #         for iter_result in iter_results:
    #             result.update(iter_result)
    #     else:                  
    #         sol, reduction_result, log_df = reduction_then_solve(instance_cpx, reduction_timelimit, prediction, confident_idx, binary_idx)
    #         log_df_lst.append(log_df)
         
    #     is_infeasible_reduction = 'infeasible' in reduction_result['reduction_status']
    #     feasible_confident_idx = confident_idx.copy()

    #     if is_infeasible_reduction:
    #         repair_timelimit = reduction_timelimit
    #         print(f">>> {instance_name}: Starting repair...")
    #         is_repaired, feasible_confident_idx, repair_time, repair_iter = repair(instance_path, instance_name, prediction, binary_idx, confident_idx, repair_timelimit)

    #         if is_repaired:
    #             instance_cpx = get_cplex_instance(instance_path, instance_name, interface_type='cplex')
    #             sol, reduction_result, log_df = reduction_then_solve(instance_cpx, reduction_timelimit, prediction, feasible_confident_idx, binary_idx)               
    #             log_df_lst.append(log_df)
    #         else:
    #             print(f">>> {instance_name}: The reduced problem could not be repaired!!!")

    #     rst = time.time() - rst
    #     solve_timelimit -= rst

    #     save_reduction_log_df(log_df_lst, log_path, instance_name)
        
    #     result.update(reduction_result)
    #     result['repair_time'] = repair_time
    #     result['repair_iter'] = repair_iter
    #     result['is_infeasible_reduction'] = is_infeasible_reduction
    #     result["reduction_rate"] = len(confident_idx) / result["num_variables"]
    #     result["reduction_rate_after_repair"] = len(feasible_confident_idx)/result["num_variables"]
        
    #     #print("Reduction result:", result)
 
    # if 'NS' in strategy:

    #     if 'UPR' in strategy:

    #         # Deleting the cuts added by UPR so that the original problem instance is restored for global search with NS.
    #         con_names = instance_cpx.linear_constraints.get_names()
    #         reduction_cuts = [c for c in con_names if 'fixed_val' in c]
    #         instance_cpx.linear_constraints.delete(reduction_cuts)
        
    #     var_scores = probs if config['pred_loss_type'] == 'bce' else 1 - (1 / (1+evidence))

    #     cb = instance_cpx.register_callback(LoggingCB)
    #     print(f">>> {instance_name}: Starting NS...")
    #     sol, ns_result = node_selection(instance_cpx, sol, solve_timelimit, prediction, binary_idx, var_scores)    

    #     result.update(ns_result)    

    #     cb.save_as_npz(str(log_path.joinpath(instance_name+ "_log.npz")), ns_result['ns_solve_time'], ns_result['ns_gap'], ns_result['ns_bestobj'])

    print(result)
    
    # total_solve_time = time.time() - total_solve_time

    # result['timelimit'] = timelimit
    # result['total_solve_time'] = total_solve_time
    # result['soft_pred_bias'] = probs[:,1].mean()
    # result["val_u_mean"] = val_u_mean
    # result["val_confident_ratio_mean"] = val_confident_ratio_mean
    # result["val_bound_min"] = val_bound_err_min
    # result["val_bound_mean"] = val_bound_err_mean
    # result["val_bound_max"] = val_bound_err_max
    # result["val_bound_std"] = val_bound_err_std
    
    if not incumbent is None: # i.e., data is labeled

        mask = np.ones(len(prediction), dtype=bool)
        mask[confident_idx] = False
        uncertain_sum = prediction[mask].sum()
        true_sum = incumbent[mask].sum()
        sol_diff = np.nan if sol is None else 1 - (incumbent == sol[binary_idx]).mean()  
        correct_u_mean, incorrect_u_mean, confident_ratio_mean, confident_acc_mean = get_uncertainty_metrics(data, prediction, uncertainty, incumbent)

        result["pred_bias"] = prediction.mean()
        result['true_bias'] = incumbent.mean()
        result["confident_true_bias"] = incumbent[confident_idx].mean()
        result["confident_pred_bias"] = prediction[confident_idx].mean()
        result["confident_acc"] = (prediction[confident_idx] == incumbent[confident_idx]).mean()
        result["acc"] = (incumbent == prediction).mean()
        result["avg_pr"] = avg_pr(torch.tensor(probs[:,1]).view(-1), torch.tensor(incumbent.round()).long()).item()
        result["uncertain_sum_err"] = uncertain_sum - true_sum
        result["sol_diff"] = sol_diff
        result["val_confident_acc_mean"] = val_confident_acc_mean
        result["correct_u_mean"] = correct_u_mean
        result["incorrect_u_mean"] = incorrect_u_mean
        result["confident_ratio_mean"] = confident_ratio_mean
        result["confident_acc_mean"] = confident_acc_mean
    
    result_df = pd.DataFrame.from_dict([result])
    result_df.to_csv(str(result_df_file_path), index=False)

    return result_df   

def experiment(
        prob_name: str, 
        train_dt_name: str,
        val_dt_name: str,
        target_dt_names: list[str],
        val_size: int, 
        test_size: int,
        target_size: int,
        timelimit: float,
        reduction_timelimit: float,
        model_configs: list[dict],
        models_dict: Optional[Dict[str, BaseModel]],
        strategies: list[str],
        n_threads: int = N_THREADS,
        ):

  val_dt = get_co_datasets(prob_name, [val_dt_name], [val_size], False)[0]
  models = []
  strategies = ['NS', 'UPR+NS']
  
  for i, config in enumerate(model_configs):

    model_dir = PROJECT_DIR.joinpath('trained_models', config["prob_name"], train_dt_name)
    inference_dir = model_dir.joinpath("inference")
    Path(inference_dir).mkdir(parents=True, exist_ok=True)
    
    if models_dict is not None:
        model_name, model, _ = load_model(config, model_dir, val_dt)

    print(">>> Getting validation metrics...") 
    print(f">>> Model {i} config:", config)

    # This line takes some time bcs metrics calculated per instance individually
    val_unc_threshold, val_confident_ratio_mean, val_confident_acc_mean, val_bound_error_stats = get_uncertainty_params(config, model, val_dt, 'median')
    
    models.append([model_name, model, val_unc_threshold, val_confident_ratio_mean, val_confident_acc_mean, val_bound_error_stats])

    target_dataset_probs = []

    for target_dt_name in target_dt_names:
        size = test_size if 'test' in target_dt_name else target_size
        data_path = DATA_DIR.joinpath('graphs', prob_name, target_dt_name, 'processed')        
        data_file_paths = sorted(glob.glob(str(data_path.joinpath("*_data.pt"))))[:size]
        target_instances = [path.split("\\")[-1].removesuffix("_data.pt") for path in data_file_paths]
        target_dataset_probs.append(target_instances)

    for model_idx in range(len(models)):

        #reading model config and params derived from validation results
        config = model_configs[model_idx]
        print(">>>>", model_idx, config)
        
        model_name, model, val_unc_threshold, val_confident_ratio_mean, val_confident_acc_mean, val_bound_error_stats = models[model_idx]

        inference_dir = PROJECT_DIR.joinpath('trained_models', config["prob_name"], train_dt_name, "inference")

        # for target_dt_idx, target_dt_name in enumerate(target_dt_names):
        #     print(">>>>>>", target_dt_name)

        #     instance_path = DATA_DIR.joinpath('instances', prob_name, target_dt_name)
            
        #     target_instances = target_dataset_probs[target_dt_idx]
        
        #     for strategy in strategies:
                    
        #         data_path = DATA_DIR.joinpath('graphs', prob_name, target_dt_name, 'processed')  
        #         log_path = inference_dir.joinpath(f"{target_dt_name}_{strategy}_{model_name}")
        #         Path(log_path).mkdir(parents=True, exist_ok=True)
                
        #         args = pd.DataFrame(target_instances, columns=['instance_name'], index=np.arange(len(target_instances)))
        #         args_lst = [deepcopy(model), data_path, instance_path, log_path, config, strategy, timelimit, reduction_timelimit, train_dt_name, \
        #                     val_unc_threshold, val_confident_ratio_mean, val_confident_acc_mean, val_bound_error_stats]
                
        #         for i, arg in enumerate(args_lst):
        #             args[i] = [arg] * len(args)

        #         args = args.values.tolist()

        #         experiment_opt_with_gnn(args[0])

        #         # with Pool(n_threads) as p:
        #         #     p.map(experiment_opt_with_gnn, args)

        #         log_files = [ name  for name in glob.glob(str(log_path.joinpath("*.csv"))) if not 'reduction' in name]
                
        #         if len(log_files) > 0:           
        #             results_df = pd.concat([pd.read_csv(file_path) for file_path in log_files])
        #             results_df.to_csv(str(log_path)+".csv", index=False)
        break

if __name__ == '__main__':

    prob_name = PROB_NAMES[-3]
    train_dt_name = TRAIN_DT_NAMES[prob_name]
    val_dt_name = VAL_DT_NAMES[prob_name]
    target_dt_names_lst = TARGET_DT_NAMES[prob_name]
    val_size = 100
    test_size = 50
    target_size = 50
    timelimit = 60*30
    reduction_timelimit = 60
    n_threads = N_THREADS // 2
    strategies = ['NS', 'UPR+NS']

    data_params = dict(
        random_state = [0],
        train_size = [1000],
        val_size = [200],
        test_size = [100],
        transfer_size = [100],
        prob_name = [prob_name]
    )

    model_hparams = dict(
        network_name = ['EC+V', 'EC', 'EC+E'],
        hidden = [32],
        num_layers = [8],
        dropout = [0.1],
        aggr = ['comb'],
        activation=['relu'],
        norm = ['graph'],
        binary_pred = [False],
        prenorm = [True],
        abc_norm = [True]
    )

    train_hparams = dict(
        batch_size = [8],
        num_epochs = [24],
        lr = [1e-4],
        weight_decay = [0.0],
        bias_threshold = [0.5],
        pred_loss_type = ['edl_digamma', 'bce'],
        edl_lambda = [6.0, None],
        evidence_func = ['softplus'],
        scheduler_step_size = [None],
        gamma = [None],
        scheduler_type = ['cycle']
    )

    #factors = ['network_name', 'strategy', 'abc_norm', 'prenorm']

    param_grid = ParameterGrid({**model_hparams, **train_hparams, **data_params})
    model_configs = []

    for config in param_grid:

        if config['pred_loss_type'] == 'bce' and not config['edl_lambda'] is None:
            continue 
        if not config['pred_loss_type'] == 'bce' and config['edl_lambda'] is None:
            continue 
        if config['num_layers'] == 8 and config['hidden'] == 64:
            continue
        if config['num_layers'] == 4 and config['hidden'] == 32:
            continue

        model_configs.append(config)
    
    experiment(
        prob_name, 
        train_dt_name,
        val_dt_name,
        target_dt_names_lst,
        val_size, 
        test_size,
        target_size,
        timelimit,
        reduction_timelimit,
        model_configs,
        strategies,
        n_threads
    )
  
