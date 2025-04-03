import numpy as np
from MIPGNN.ml_augmented_opt import *
from MIPGNN.utils import get_trained_model_config
from main_utils import *
from MVB import *
from gurobipy import *
from pandas import DataFrame
import csv

def get_probs_(config, model, data):
    probs, _, _, _, y_true, _ = get_prediction(config, model, data)
    return probs[:,1], y_true

def get_accuracy(y_true, y_pred, tau):
    """
    Compute accuracy for threshold tau (between 0 and 0.5)
    """
    
    assert tau >= 0 and tau <= 0.5
    
    tau_up = 1 - tau
    tau_low = tau
    
    up_id = np.where(y_pred >= tau_up)[0]
    low_id = np.where(y_pred <= tau_low)[0]
    
    true_up_id = np.where((y_pred >= tau_up) & (y_true == 1))[0]
    acc_up = (np.sum(y_true[up_id] == 1) - np.sum(y_pred[up_id])) / len(up_id)
    # acc_low = np.sum(y_true[up_id] == 1) / len(up_id)
    acc_low = np.sum(y_true[low_id] == 0) / len(low_id)
    
    if len(up_id) == 0:
        acc_up = 0
    
    if len(low_id) == 0:
        acc_low = 0
    
    return acc_up, acc_low, len(up_id), len(low_id)


def get_accuracy_mean_var(y_trues, y_preds, tau):
    """
    Compute sample mean and variance of accuracy for threshold tau across instances
    """
    
    acc_ups = []
    acc_lows = []
    
    for y_true, y_pred in zip(y_trues, y_preds):
        acc_up, acc_low, len_up_id, len_low_id = get_accuracy(y_true, y_pred, tau)
        acc_ups.append(acc_up)
        acc_lows.append(acc_low)
        
    return np.mean(acc_ups), np.var(acc_ups), np.mean(acc_lows), np.var(acc_lows), len_up_id, len_low_id


if __name__ == "__main__":
    import argparse
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob_name", type=str, default='indset')
    args = parser.parse_args()
    prob_name = args.prob_name
    config = get_config(prob_name)
    target_dt_names_lst = [VAL_DT_NAMES[prob_name]]

    y_trues = []
    y_preds = []
    for target_dt_name in target_dt_names_lst:
        instance_names = get_instance_names(prob_name, target_dt_name)
        for instance_name in instance_names:
            try: 
                data = get_data(prob_name, target_dt_name, instance_name)
                model, model_name = get_model(prob_name, config, data)
                instance_path = get_instance_path(prob_name, target_dt_name, instance_name)
                y_pred, y_true = get_probs_(config, model, data)
                y_trues.append(y_true)
                y_preds.append(y_pred)
            except Exception as e:
                print("Error: ", e)
                continue
    
    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)
    
    with open(f'results/tau_{prob_name}.csv', mode='a', newline='') as file:
        file.truncate(0)
        writer = csv.writer(file)
        writer.writerow(["tau", "acc_up_mean", "acc_low_mean", "acc_up_var", "acc_low_var", "len_up_id", "len_low_id"])
        for tau in np.arange(0.010, 0.50, 0.0051):
            tau = round(tau, 3)
            acc_up_mean, acc_up_var, acc_low_mean, acc_low_var, len_up_id, len_low_id = get_accuracy_mean_var(y_trues, y_preds, tau)
            print("Threshold: ", tau)
            writer.writerow([tau, acc_up_mean, acc_low_mean, acc_up_var, acc_low_var, len_up_id, len_low_id])