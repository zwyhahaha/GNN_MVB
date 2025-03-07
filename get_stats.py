import numpy as np
from ml_augmented_opt import *
from utils import get_trained_model_config
from mvb_experiments.MVB import *
from mvb_experiments.mkp.result.mkpUtils import *
from gurobipy import *
from pandas import DataFrame
import csv

def get_config(prob_name):
    config = get_trained_model_config(prob_name, 0)
    return config

def get_instance_names(prob_name, target_dt_name, sample):
    instances_dir = DATA_DIR.joinpath('graphs', prob_name, target_dt_name, 'processed')
    # instance_type = INSTANCE_FILE_TYPES[prob_name]
    instance_type = '.pt'
    instance_names = [f.stem.replace('_data', '') for f in instances_dir.glob(f'*{instance_type}')]
    if sample:
        random.seed(42)
        instance_names = random.sample(instance_names, 20)
    return instance_names

def get_instance_path(prob_name, target_dt_name, instance_name):
    instance_type = INSTANCE_FILE_TYPES[prob_name]
    instance_path = f'data/instances/{prob_name}/{target_dt_name}/{instance_name}{instance_type}'
    return instance_path 

def get_data(prob_name, target_dt_name, instance_name):
    data_path = DATA_DIR.joinpath('graphs', prob_name, target_dt_name, 'processed')
    data = torch.load(str(data_path.joinpath(instance_name + "_data.pt")),map_location=DEVICE)
    instance_name = data.instance_name
    print(">>> Reading:", instance_name)
    return data

def get_model(config, data):
    model_dir = PROJECT_DIR.joinpath('trained_models', prob_name, TRAIN_DT_NAMES[prob_name])
    model_name, model, _ = load_model(config, model_dir, data)
    return model, model_name

def get_probs(config, model, data):
    probs, prediction, _, _, y_true, _ = get_prediction(config, model, data)
    return probs[:,1], prediction, y_true

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
        instance_names = get_instance_names(prob_name, target_dt_name, sample=0)
        for instance_name in instance_names:
            try: 
                data = get_data(prob_name, target_dt_name, instance_name)
                model, model_name = get_model(config, data)
                instance_path = get_instance_path(prob_name, target_dt_name, instance_name)
                y_pred, _, y_true = get_probs(config, model, data)
                y_trues.append(y_true)
                y_preds.append(y_pred)
            except Exception as e:
                print("Error: ", e)
                continue
    
    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)
    
    with open(f'results/tilde_tau_{prob_name}.csv', mode='a', newline='') as file:
        file.truncate(0)
        writer = csv.writer(file)
        writer.writerow(["tau", "acc_up_mean", "acc_low_mean", "acc_up_var", "acc_low_var", "len_up_id", "len_low_id"])
        for tau in np.arange(0.010, 0.50, 0.0051):
            tau = round(tau, 3)
            acc_up_mean, acc_up_var, acc_low_mean, acc_low_var, len_up_id, len_low_id = get_accuracy_mean_var(y_trues, y_preds, tau)
            print("Threshold: ", tau)
            writer.writerow([tau, acc_up_mean, acc_low_mean, acc_up_var, acc_low_var, len_up_id, len_low_id])