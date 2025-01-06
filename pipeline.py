from ml_augmented_opt import *
from utils import get_trained_model_config
from mvb_experiments.MVB import *
from mvb_experiments.mkp.result.mkpUtils import *
from gurobipy import *
import coptpy as cp 
from coptpy import COPT

TMVBarray = [0, 0]
TMVBWarmarray = [0, 0]
TOriginalArray = [0, 0, 0, 0]

def get_config(prob_name):
    config = get_trained_model_config(prob_name, 0)
    return config

def get_instance_names(prob_name, target_dt_name):
    instances_dir = DATA_DIR.joinpath('instances', prob_name, target_dt_name)
    instance_type = INSTANCE_FILE_TYPES[prob_name]
    instance_names = [f.stem for f in instances_dir.glob(f'*{instance_type}')]
    return instance_names

def get_data(prob_name, target_dt_name, instance_name):
    data_path = DATA_DIR.joinpath('graphs', prob_name, target_dt_name, 'processed')  
    data = torch.load(str(data_path.joinpath(instance_name + "_data.pt")))
    instance_name = data.instance_name
    print(">>> Reading:", instance_name)
    return data

def get_model(config, data):
    model_dir = PROJECT_DIR.joinpath('trained_models', prob_name, TRAIN_DT_NAMES[prob_name])
    model_name, model, _ = load_model(config, model_dir, data)
    return model, model_name

def get_probs(config, model, data):
    probs, prediction, _, _, _, _ = get_prediction(config, model, data)
    return probs, prediction

def get_instance_path(prob_name, target_dt_name, instance_name):
    instance_path = f'data/instances/{prob_name}/{target_dt_name}/{instance_name}.lp'
    return instance_path 

def whenIsBestObjFound(model, where, time=TMVBarray):
    if where == GRB.Callback.MIPSOL:
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        # objnow = model.cbGet(GRB.Callback.MIPSOL_OBJ)

        if objbst > time[1]:
            bestTime = model.cbGet(GRB.Callback.RUNTIME)
            # print("Solution found at %3g" % bestTime)
            time[0] = bestTime
            time[1] = objbst

def whenIsBestWarmObjFound(model, where, time=TMVBWarmarray):
    if where == GRB.Callback.MIPSOL:
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        # objnow = model.cbGet(GRB.Callback.MIPSOL_OBJ)

        if objbst > time[1]:
            bestTime = model.cbGet(GRB.Callback.RUNTIME)
            # print("Solution found at %3g" % bestTime)
            time[0] = bestTime
            time[1] = objbst

def whenIsMVBObjFound(model, where, time=TOriginalArray):
    if where == GRB.Callback.MIPSOL:
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        objnow = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        objbst = max(objbst, objnow)

        if objbst >= time[2]:
            time[1] = 1
            time[3] = model.cbGet(GRB.Callback.RUNTIME)
            model.terminate()

def computeObjLoss(mvbObj, originalObj):
    return (originalObj - mvbObj) / originalObj * 100

def mvb_experiment(instance_path, solver, probs, prediction, args):
    instance_name = instance_path.split('/')[-1].split('.')[0]
    if solver == 'copt':
        env = cp.Envr()
        cp_model = env.createModel("lp")
        cp_model.read(instance_path)
        initcpmodel = cp_model.clone()

        print(">>> COPT solve")
        cp_model.solve()
        ori_time = cp_model.getAttr("SolvingTime")

        print(">>> COPT warm solve")
        cp_model.setMipStart(cp_model.getVars(), prediction)
        cp_model.solve()
        ori_warm_time = cp_model.getAttr("SolvingTime")

        print(">>> MVB solve")
        m=cp_model.getAttr("Rows")
        n=cp_model.getAttr("Cols")
        mvbsolver = MVB(m, n)
        mvbsolver.registerModel(initcpmodel, solver=solver)
        mvbsolver.registerVars(list(range(n)))
        mvb_model = mvbsolver.getMultiVarBranch(Xpred=probs[:,1])
        mvb_model.solve()
        mvb_time = mvb_model.getAttr("SolvingTime")

        print(ori_time, ori_warm_time, mvb_time)
        return ori_time, ori_warm_time, mvb_time
    elif solver == 'gurobi':
        initgrbmodel = read(instance_path)
        initgrbmodel.setParam("TimeLimit", args.maxtime)

        grbmodel = initgrbmodel.copy()
        grbmodel.setParam("MIPGap", args.gap)
        TMVBarray[0] = 0
        TMVBarray[1] = 0
        grbmodel.optimize(whenIsBestObjFound)
        originalgap = grbmodel.getAttr("MIPGap")
        ori_time = grbmodel.getAttr("RunTime")
        originalObjVal = grbmodel.getAttr("ObjVal")
        ori_best_time = TMVBarray[0]

        grbmodel = initgrbmodel.copy()
        grbmodel.setParam("MIPGap", args.gap)
        grbmodel.setAttr(GRB.Attr.Start, grbmodel.getVars(), prediction)
        TMVBWarmarray[0] = 0
        TMVBWarmarray[1] = 0
        grbmodel.optimize(whenIsBestWarmObjFound)
        originalgap_warm = grbmodel.getAttr("MIPGap")
        ori_warm_time = grbmodel.getAttr("RunTime")
        originalObjVal_warm = grbmodel.getAttr("ObjVal")
        ori_warm_best_time = TMVBWarmarray[0]

        m=grbmodel.getAttr("NumConstrs")
        n=grbmodel.getAttr("NumVars")
        mvbsolver = MVB(m, n)
        
        mvbsolver.registerModel(initgrbmodel, solver=solver)
        mvbsolver.registerVars(list(range(n)))
        mvbsolver.setParam(threshold=args.fixthresh,tmvb=[args.fixthresh, args.psucceed], pSuccess = [args.psucceed])
        mvb_model = mvbsolver.getMultiVarBranch(Xpred=probs[:,1])
        mvb_model.setParam("MIPGap", 0.0)
        TOriginalArray[0] = 0
        TOriginalArray[1] = 0
        TOriginalArray[2] = min(originalObjVal, originalObjVal_warm)
        TOriginalArray[3] = 0
        mvb_model.optimize(whenIsMVBObjFound)
        mvb_time = mvb_model.getAttr("RunTime")
        mvbObjVal = mvb_model.getAttr("ObjVal")
        if TOriginalArray[1]:
            TimeDominance = TOriginalArray[3]
        else:
            TimeDominance = args.maxtime

        objLoss = computeObjLoss(mvbObjVal, originalObjVal)
        objLoss_warm = computeObjLoss(mvbObjVal, originalObjVal_warm)

        print(ori_best_time, ori_warm_best_time, TimeDominance, objLoss, objLoss_warm)

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
    else:
        ori_time, ori_warm_time, mvb_time = None, None, None
        print(">>> Solver not supported")
    

def log_results(prob_name, target_dt_name, instance_name, experiment_name, results):

    results_path = PROJECT_DIR.joinpath('results',prob_name,target_dt_name, f'{experiment_name}.csv')
    print(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if not results_path.exists():
        results_df = pd.DataFrame([results])
        results_df.to_csv(results_path, index=False)
    else:
        results_df = pd.read_csv(results_path)
        results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)
        results_df.to_csv(results_path, index=False)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--maxtime", type=float, default=3600.0)
parser.add_argument("--fixthresh", type=float, default=1.1)
parser.add_argument("--psucceed", type=float, default=0.9)
parser.add_argument("--gap", type=float, default=0.01)
parser.add_argument("--solver", type=str, default='gurobi')
parser.add_argument("--prob_name", type=str, default='indset')

args = parser.parse_args()
solver = args.solver
prob_name = args.prob_name
config = get_config(prob_name)
# target_dt_names_lst = TARGET_DT_NAMES[prob_name]
target_dt_names_lst = [VAL_DT_NAMES[prob_name]]
experiment_name = f"{solver}_fixthresh_{args.fixthresh}_psucceed_{args.psucceed}_gap_{args.gap}_maxtime_{args.maxtime}"

for target_dt_name in target_dt_names_lst:
    instance_names = get_instance_names(prob_name, target_dt_name)
    for instance_name in instance_names:
        data = get_data(prob_name, target_dt_name, instance_name)
        model, model_name = get_model(config, data)
        # experiment_name = model_name
        probs, prediction = get_probs(config, model, data)
        instance_path = get_instance_path(prob_name, target_dt_name, instance_name)
        results = mvb_experiment(instance_path, solver, probs, prediction, args)
        log_results(prob_name, target_dt_name, instance_name, experiment_name, results)