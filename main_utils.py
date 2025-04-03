from MIPGNN.ml_augmented_opt import load_model, get_prediction
from MIPGNN.global_vars import *
from MIPGNN.utils import get_trained_model_config
from gurobipy import *
import numpy as np
import pandas as pd

def get_config(prob_name):
    config = get_trained_model_config(prob_name, 0)
    return config

def get_instance_names(prob_name, target_dt_name):
    instances_dir = DATA_DIR.joinpath('graphs', prob_name, target_dt_name, 'processed')
    instance_type = '.pt'
    instance_names = [f.stem.replace('_data', '') for f in instances_dir.glob(f'*{instance_type}')]
    return instance_names

def get_instance_path(prob_name, target_dt_name, instance_name):
    instance_type = INSTANCE_FILE_TYPES[prob_name]
    instance_path = DATA_DIR.joinpath(f'instances/{prob_name}/{target_dt_name}/{instance_name}{instance_type}')
    instance_path = str(instance_path)
    return instance_path 

def get_data(prob_name, target_dt_name, instance_name):
    data_path = DATA_DIR.joinpath('graphs', prob_name, target_dt_name, 'processed')
    data = torch.load(str(data_path.joinpath(instance_name + "_data.pt")),map_location=DEVICE, weights_only=False)
    instance_name = data.instance_name
    print(">>> Reading:", instance_name)
    return data

def get_model(prob_name, config, data):
    model_dir = PROJECT_DIR.joinpath('trained_models', prob_name, TRAIN_DT_NAMES[prob_name])
    model_name, model, _ = load_model(config, model_dir, data)
    return model, model_name

def get_probs(config, model, data):
    """
    Get the probabilities and predictions from GNN.
    """
    probs, prediction, _, _, _, _ = get_prediction(config, model, data)
    return probs[:,1], prediction

def get_probs_from_lp(instance_path, solver):
    """
    Get the probabilities and predictions from LP relaxation.
    """
    initgrbmodel = read(instance_path)
    relaxgrbmodel = initgrbmodel.relax()
    relaxgrbmodel.setParam("CrossOver", 0)
    relaxgrbmodel.setParam("Method", 2)
    relaxgrbmodel.optimize()
    sol = relaxgrbmodel.getVars()
    probs = np.array([v.x for v in sol])
    prediction = np.zeros(len(probs))
    prediction[probs >= 0.5] = 1
    return probs, prediction

def computeObjLoss(mvbObj, originalObj, ModelSense = -1):
    if ModelSense == -1:
        return (originalObj - mvbObj) / originalObj * 100
    elif ModelSense == 1:
        return (mvbObj - originalObj) / originalObj * 100
    
def get_results(instance_name,m,n,ori_time,ori_warm_time,mvb_time,ori_best_time,ori_warm_best_time,TimeDominance,objLoss,objLoss_warm):

    print(f"{'='*20} Primal Heuristic Time {instance_name} {'='*20}")
    print(f"{'Original Time':<25}{'Original (Warm) Time':<30}{'MVB Time':<20}")
    print(f"{ori_best_time:<25.3f}{ori_warm_best_time:<30.3f}{TimeDominance:<20.3f}")
    print(f"{'='*63}")

    results = {
            'instance_name': instance_name,
            'rows': m,
            'cols': n,
            'ori_time': ori_time,
            'ori_warm_time': ori_warm_time,
            'mvb_time': mvb_time,
            'ori_best_time': ori_best_time,
            'ori_warm_best_time': ori_warm_best_time,
            'TimeDominance': TimeDominance,
            'objLoss': objLoss,
            'objLoss_warm': objLoss_warm
        }
    return results
    
def log_results(prob_name, target_dt_name, experiment_name, results):

    results_path = PROJECT_DIR.joinpath('../results',prob_name,target_dt_name, f'{experiment_name}.csv')
    print(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if not results_path.exists():
        results_df = pd.DataFrame([results])
        results_df.to_csv(results_path, index=False)
    else:
        results_df = pd.read_csv(results_path)
        results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)
        results_df.to_csv(results_path, index=False)