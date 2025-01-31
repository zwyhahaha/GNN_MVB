from ml_augmented_opt import *
from utils import get_trained_model_config
from mvb_experiments.MVB import *
from mvb_experiments.mkp.result.mkpUtils import *
from gurobipy import *
import coptpy as cp 
from coptpy import COPT
import numpy as np

TMVBarray = [0, 0]
TMVBWarmarray = [0, 0]
TOriginalArray = [0, 0, 0, 0]

def get_config(prob_name):
    config = get_trained_model_config(prob_name, 2)
    return config

def get_instance_names(prob_name, target_dt_name):
    instances_dir = DATA_DIR.joinpath('graphs', prob_name, target_dt_name, 'processed')
    instance_type = INSTANCE_FILE_TYPES[prob_name]
    instance_type = '.pt'
    instance_names = [f.stem.replace('_data', '') for f in instances_dir.glob(f'*{instance_type}')]
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

def normalize_probs(probs):
    a_normalized = np.log(probs + 1e-30)
    a_normalized -= np.min(a_normalized)
    a_normalized /= np.max(a_normalized)
    return a_normalized

def get_instance_path(prob_name, target_dt_name, instance_name):
    instance_type = INSTANCE_FILE_TYPES[prob_name]
    instance_path = f'data/instances/{prob_name}/{target_dt_name}/{instance_name}{instance_type}'
    return instance_path 

def whenIsBestMinObjFound(model, where, time=TMVBarray):
    if where == GRB.Callback.MIPSOL:
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        # objnow = model.cbGet(GRB.Callback.MIPSOL_OBJ)

        if objbst < time[1]:
            bestTime = model.cbGet(GRB.Callback.RUNTIME)
            # print("Solution found at %3g" % bestTime)
            time[0] = bestTime
            time[1] = objbst

def whenIsBestMaxObjFound(model, where, time=TMVBarray):
    if where == GRB.Callback.MIPSOL:
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        # objnow = model.cbGet(GRB.Callback.MIPSOL_OBJ)

        if objbst > time[1]:
            bestTime = model.cbGet(GRB.Callback.RUNTIME)
            # print("Solution found at %3g" % bestTime)
            time[0] = bestTime
            time[1] = objbst

def whenIsBestMinWarmObjFound(model, where, time=TMVBWarmarray):
    if where == GRB.Callback.MIPSOL:
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        # objnow = model.cbGet(GRB.Callback.MIPSOL_OBJ)

        if objbst < time[1]:
            bestTime = model.cbGet(GRB.Callback.RUNTIME)
            # print("Solution found at %3g" % bestTime)
            time[0] = bestTime
            time[1] = objbst

def whenIsBestMaxWarmObjFound(model, where, time=TMVBWarmarray):
    if where == GRB.Callback.MIPSOL:
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        # objnow = model.cbGet(GRB.Callback.MIPSOL_OBJ)

        if objbst > time[1]:
            bestTime = model.cbGet(GRB.Callback.RUNTIME)
            # print("Solution found at %3g" % bestTime)
            time[0] = bestTime
            time[1] = objbst

def whenIsMVBMinObjFound(model, where, time=TOriginalArray):
    if where == GRB.Callback.MIPSOL:
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        objnow = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        objbst = min(objbst, objnow)

        if objbst <= time[2] and time[1] == 0:
            time[1] = 1
            time[3] = model.cbGet(GRB.Callback.RUNTIME)
            # model.terminate()

def whenIsMVBMaxObjFound(model, where, time=TOriginalArray):
    if where == GRB.Callback.MIPSOL:
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        objnow = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        objbst = max(objbst, objnow)

        if objbst >= time[2] and time[1] == 0:
            time[1] = 1
            time[3] = model.cbGet(GRB.Callback.RUNTIME)
            # model.terminate()

def computeObjLoss(mvbObj, originalObj, ModelSense = -1):
    if ModelSense == -1:
        return (originalObj - mvbObj) / originalObj * 100
    elif ModelSense == 1:
        return (mvbObj - originalObj) / originalObj * 100

def mvb_experiment(instance_path, instance_name, solver, probs, prediction, args):
    # instance_name = instance_path.split('/')[-1].split('.')[0]
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
        initgrbmodel.setParam("Heuristics", args.heuristics)
        ModelSense = initgrbmodel.getAttr("ModelSense") # 1: min, -1: max

        grbmodel = initgrbmodel.copy()
        grbmodel.setParam("MIPGap", args.gap)
        TMVBarray[0] = 0
        if ModelSense == 1:
            TMVBarray[1] = np.Inf # obj
            grbmodel.optimize(whenIsBestMinObjFound)
        elif ModelSense == -1:
            TMVBarray[1] = -np.Inf # obj
            grbmodel.optimize(whenIsBestMaxObjFound)
        originalgap = grbmodel.getAttr("MIPGap")
        ori_time = grbmodel.getAttr("RunTime")
        originalObjVal = grbmodel.getAttr("ObjVal")
        ori_best_time = TMVBarray[0]

        ### test prediction accuracy
        sol = grbmodel.getVars()
        sol = np.array([v.x for v in sol])
        acc = (sol==prediction).sum() / len(sol)
        print(">>> Prediction accuracy:", acc)
        idx = (prediction == 1)
        acc1 = (sol[idx]==prediction[idx]).sum() / (len(sol[idx])+1e-8)
        print(">>> Prediction accuracy (1):", acc1)
        acc0 = (sol[~idx]==prediction[~idx]).sum() / (len(sol[~idx])+1e-8)
        print(">>> Prediction accuracy (0):", acc0)

        grbmodel = initgrbmodel.copy()
        grbmodel.setParam("MIPGap", args.gap)
        grbmodel.setAttr(GRB.Attr.Start, grbmodel.getVars(), prediction)
        TMVBWarmarray[0] = 0
        if ModelSense == 1:
            TMVBWarmarray[1] = np.Inf # obj
            grbmodel.optimize(whenIsBestMinWarmObjFound)
        elif ModelSense == -1:
            TMVBWarmarray[1] = -np.Inf # obj
            grbmodel.optimize(whenIsBestMaxWarmObjFound)
        originalgap_warm = grbmodel.getAttr("MIPGap")
        ori_warm_time = grbmodel.getAttr("RunTime")
        originalObjVal_warm = grbmodel.getAttr("ObjVal")
        ori_warm_best_time = TMVBWarmarray[0]

        m=grbmodel.getAttr("NumConstrs")
        n=grbmodel.getAttr("NumVars")
        mvbsolver = MVB(m, n)
        
        mvbsolver.registerModel(initgrbmodel, solver=solver)
        mvbsolver.registerVars(list(range(n)))
        mvbsolver.setParam(threshold=args.fixthresh,tmvb=[args.fixthresh, 0.9999999999999],pSuccessLow = [args.psucceed_low],pSuccessUp = [args.psucceed_up])
        mvb_model = mvbsolver.getMultiVarBranch(Xpred=probs[:,1],upCut=args.upCut,lowCut=args.lowCut,normalize=args.normalize)
        mvb_model.setParam("MIPGap", args.gap/2)
        TOriginalArray[0] = 0
        TOriginalArray[1] = 0
        TOriginalArray[3] = 0
        if ModelSense == 1:
            TOriginalArray[2] = max(originalObjVal, originalObjVal_warm)
            mvb_model.optimize(whenIsMVBMinObjFound)
        elif ModelSense == -1:
            TOriginalArray[2] = min(originalObjVal, originalObjVal_warm)
            mvb_model.optimize(whenIsMVBMaxObjFound)
        mvb_time = mvb_model.getAttr("RunTime")
        try:
            mvbObjVal = mvb_model.getAttr("ObjVal")
        except:
            mvbObjVal = np.nan
        
        if ModelSense == 1:
            isopt = (mvbObjVal <= originalObjVal)
        elif ModelSense == -1:
            isopt = (mvbObjVal >= originalObjVal)

        if TOriginalArray[1]:
            TimeDominance = TOriginalArray[3]
        elif isopt:
            TimeDominance = mvb_time
        else:
            TimeDominance = args.maxtime

        objLoss = computeObjLoss(mvbObjVal, originalObjVal, ModelSense)
        objLoss_warm = computeObjLoss(mvbObjVal, originalObjVal_warm, ModelSense)
        print(ori_best_time, ori_warm_best_time, TimeDominance, objLoss, objLoss_warm)

        results = {
            'instance_name': instance_name,
            'rows': m,
            'cols': n,
            "acc": acc,
            "acc1": acc1,
            "acc0": acc0,
            'ori_time': ori_time,
            'ori_warm_time': ori_warm_time,
            'mvb_time': mvb_time,
            'ori_best_time': ori_best_time,
            'ori_warm_best_time': ori_warm_best_time,
            'TimeDominance': TimeDominance,
            'objLoss': objLoss,
            'objLoss_warm': objLoss_warm
        }

        if args.robust:
            isGeq = 1 if ModelSense == -1 else 0
            model_list = mvbsolver.get_model_list(Xpred=probs[:,1],obj=mvbObjVal,isGeq=isGeq,
                                                  upCut=args.upCut,lowCut=args.lowCut,gap=0)
            model_names = ["cl", "uc", "cc"]
            times = []
            best_obj = mvbObjVal
            for i, model in enumerate(model_list):
                model.setParam("MIPGap", args.gap/2)
                # model.setParam("Cuts", 3)
                # model.setParam("Heuristics", 0)
                model.setParam("MIPFocus", 3)
                model.optimize()
                time = model.getAttr("RunTime")
                times.append(time)
                try:
                    obj = model.getAttr("ObjVal")
                except:
                    obj = np.nan
                if ModelSense == 1:
                    best_obj = min(best_obj, obj)
                elif ModelSense == -1:
                    best_obj = max(best_obj, obj)
                print(f"{model_names[i]}: {time}, {obj}")
            
            objLoss_all = computeObjLoss(best_obj, originalObjVal, ModelSense)
            objLoss_warm_all = computeObjLoss(best_obj, originalObjVal_warm, ModelSense)
            
            results['mvb_time_all'] = mvb_time + sum(times)
            for i, model in enumerate(model_list):
                results[f"{model_names[i]}_time"] = times[i]
            results['objLoss_all'] = objLoss_all
            results['objLoss_warm_all'] = objLoss_warm_all
        
            print(ori_best_time, ori_warm_best_time, TimeDominance, mvb_time + sum(times), times, objLoss, objLoss_warm, objLoss_all, objLoss_warm_all)
        
        return results
    else:
        ori_time, ori_warm_time, mvb_time = None, None, None
        print(">>> Solver not supported")
    

def log_results(prob_name, target_dt_name, experiment_name, results):

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
parser.add_argument("--psucceed_low", type=float, default=0.99)
parser.add_argument("--psucceed_up", type=float, default=0.999)
parser.add_argument("--gap", type=float, default=0.01)
parser.add_argument("--heuristics", type=float, default=0.05)
parser.add_argument("--solver", type=str, default='gurobi')
parser.add_argument("--prob_name", type=str, default='fcmnf')
parser.add_argument("--robust", type=int, default=0)
parser.add_argument("--upCut", type=int, default=0)
parser.add_argument("--lowCut", type=int, default=1)
parser.add_argument("--normalize", type=int, default=0)

args = parser.parse_args()
solver = args.solver
prob_name = args.prob_name
config = get_config(prob_name)
# target_dt_names_lst = TARGET_DT_NAMES[prob_name]
target_dt_names_lst = [VAL_DT_NAMES[prob_name]]
# target_dt_names_lst = [TRAIN_DT_NAMES[prob_name]]
# target_dt_names_lst = ['transfer']
experiment_name = f"{solver}_robust_{args.robust}_plow_{args.psucceed_low * args.lowCut}_pup_{args.psucceed_up * args.upCut}_gap_{args.gap}_normalize_{args.normalize}_heuristics_{args.heuristics}"

for target_dt_name in target_dt_names_lst:
    instance_names = get_instance_names(prob_name, target_dt_name)
    for instance_name in instance_names:
        # instance_name = 'instance_183'
        try:
            data = get_data(prob_name, target_dt_name, instance_name)
            model, model_name = get_model(config, data)
            probs, prediction = get_probs(config, model, data)
            instance_path = get_instance_path(prob_name, target_dt_name, instance_name)
            results = mvb_experiment(instance_path, instance_name, solver, probs, prediction, args)
            log_results(prob_name, target_dt_name, experiment_name, results)
        except Exception as e:
            print(f"Error in {instance_name}: {e}")
            continue