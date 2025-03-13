from ml_augmented_opt import *
from utils import get_trained_model_config
from mvb_experiments.MVB import *
from mvb_experiments.mkp.result.mkpUtils import *
from gurobipy import *
import coptpy as cp
from coptpy import COPT
import numpy as np
import time

TMVBarray = [0, 0]
TMVBWarmarray = [0, 0]
TOriginalArray = [0, 0, 0, 0]
sol = [None, 0, 10, 0.0]

CPOriginalArray = [0]
CPWarmArray = [0]
CPMVBArray = [0]

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
    probs, prediction, _, _, _, _ = get_prediction(config, model, data)
    return probs[:,1], prediction

def get_probs_from_lp(instance_path, solver):
    if solver == 'gurobi':
        initgrbmodel = read(instance_path)
        relaxgrbmodel = initgrbmodel.relax()
        relaxgrbmodel.setParam("CrossOver", 0)
        relaxgrbmodel.setParam("Method", 2)
        relaxgrbmodel.optimize()
        sol = relaxgrbmodel.getVars()
        probs = np.array([v.x for v in sol])
        # sol = get_grb_root_relaxation(initgrbmodel)
        # probs = np.array(sol)
        prediction = np.zeros(len(probs))
        prediction[probs >= 0.5] = 1
        return probs, prediction
    else:
        print(">>> Solver not supported")

def grb_callback(model: Model, where, x=sol):
    if where != GRB.Callback.MIPNODE:
        return

    if model.cbGet(GRB.Callback.MIPNODE_NODCNT) > 0:
        print("Root node completed, terminate now")
        model.terminate()
        return

    if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.Status.OPTIMAL:
        x[0] = model.cbGetNodeRel(model.getVars())
        sol[1] = sol[1] + 1
        print("Root solution count %d" % sol[1])

        if sol[1] >= sol[2]:
            model.terminate()
            sol[3] = model.getAttr(GRB.Attr.Runtime)
            return
    return

def get_grb_root_relaxation(model: Model):
    model.setParam(GRB.Param.Heuristics, 0)
    model.setParam(GRB.Param.Cuts, 3)
    model.optimize(grb_callback)
    root_sol = sol[0]

    return root_sol

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

class cbGetBestTime(cp.CallbackBase):
    
    def __init__(self, vars, ModelSense = -1):
        super().__init__()
        self._vars = vars
        self._ModelSense = ModelSense
        self._bestTime = 3600
        self._bestObj = np.Inf if ModelSense == 1 else -np.Inf

    def callback(self):
        if self.where() == cp.COPT.CBCONTEXT_MIPSOL:
            objbst = self.getInfo("BestObj")
            if self._ModelSense == 1:
                if objbst < self._bestObj:
                    self._bestTime = time.time() - CPOriginalArray[0]
                    self._bestObj = objbst
            elif self._ModelSense == -1:
                if objbst > self._bestObj:
                    self._bestTime = time.time() - CPOriginalArray[0]
                    self._bestObj = objbst

    def get_best_time(self):
        return self._bestTime

    def get_best_obj(self):
        return self._bestObj
    
class cbGetWarmBestTime(cp.CallbackBase):
    
    def __init__(self, vars, ModelSense = -1):
        super().__init__()
        self._vars = vars
        self._ModelSense = ModelSense
        self._bestTime = 3600
        self._bestObj = np.Inf if ModelSense == 1 else -np.Inf

    def callback(self):
        if self.where() == cp.COPT.CBCONTEXT_MIPSOL:
            objbst = self.getInfo("BestObj")
            if self._ModelSense == 1:
                if objbst < self._bestObj:
                    self._bestTime = time.time() - CPWarmArray[0]
                    self._bestObj = objbst
            elif self._ModelSense == -1:
                if objbst > self._bestObj:
                    self._bestTime = time.time() - CPWarmArray[0]
                    self._bestObj = objbst

    def get_best_time(self):
        return self._bestTime

    def get_best_obj(self):
        return self._bestObj

class cbGetMVBBestTime(cp.CallbackBase):
    
    def __init__(self, vars, ModelSense = -1):
        super().__init__()
        self._vars = vars
        self._ModelSense = ModelSense
        self._bestTime = 3600
        self._bestObj = np.Inf if ModelSense == 1 else -np.Inf

    def callback(self):
        if self.where() == cp.COPT.CBCONTEXT_MIPSOL:
            objbst = self.getInfo("BestObj")
            if self._ModelSense == 1:
                if objbst <= self._bestObj:
                    self._bestTime = time.time() - CPMVBArray[0]
                    self._bestObj = objbst
            elif self._ModelSense == -1:
                if objbst >= self._bestObj:
                    self._bestTime = time.time() - CPMVBArray[0]
                    self._bestObj = objbst

    def get_best_time(self):
        return self._bestTime

    def get_best_obj(self):
        return self._bestObj

def mvb_experiment(instance_path, instance_name, solver, probs, prediction, args):
    if solver == 'copt':
        env = cp.Envr()
        cp_model = env.createModel("lp")
        cp_model.read(instance_path)
        cp_model.setParam(COPT.Param.TimeLimit, args.maxtime)
        cp_model.setParam(COPT.Param.Presolve, 2)
        heu_level = {0.0: 0, 0.05: -1, 1.0: 3}
        cp_model.setParam(COPT.Param.HeurLevel, heu_level[args.heuristics])
        initcpmodel = cp_model.clone()
        ModelSense = initcpmodel.getAttr("ObjSense") # 1: min, -1: max

        ori_callback = cbGetBestTime(cp_model.getVars(), ModelSense)
        cpmodel = initcpmodel.clone()
        cpmodel.setParam(COPT.Param.RelGap, args.gap)
        cpmodel.setCallback(ori_callback, cp.COPT.CBCONTEXT_MIPSOL)
        CPOriginalArray[0] = time.time()
        cpmodel.solve()
        ori_time = cpmodel.getAttr("SolvingTime")
        ori_best_time = ori_callback.get_best_time()
        originalObjVal = cpmodel.getAttr("BestObj")

        if args.data_free:
            ori_warm_time = 0
            originalObjVal_warm = -ModelSense * np.Inf
            ori_warm_best_time = 0
        else:
            warm_callback = cbGetWarmBestTime(cp_model.getVars(), ModelSense)
            cpmodel = initcpmodel.clone()
            cpmodel.setParam(COPT.Param.RelGap, args.gap)
            cpmodel.setMipStart(cpmodel.getVars(), prediction)
            cpmodel.setCallback(warm_callback, cp.COPT.CBCONTEXT_MIPSOL)
            CPWarmArray[0] = time.time()
            cpmodel.solve()
            ori_warm_time = cpmodel.getAttr("SolvingTime")
            originalObjVal_warm = cpmodel.getAttr("BestObj")
            ori_warm_best_time = warm_callback.get_best_time()

        m=cp_model.getAttr("Rows")
        n=cp_model.getAttr("Cols")
        mvbsolver = MVB(m, n)
        mvbsolver.registerModel(initcpmodel, solver=solver)
        mvbsolver.registerVars(list(range(n)))
        mvbsolver.setParam(fixratio=args.fixratio, threshold=args.fixthresh,tmvb=[args.fixthresh, args.tmvb],pSuccessLow = [args.psucceed_low],pSuccessUp = [args.psucceed_up])
        mvb_model = mvbsolver.getMultiVarBranch(Xpred=probs,upCut=args.upCut,lowCut=args.lowCut,ratio_involve=args.ratio_involve,ratio=[args.ratio_low,args.ratio_up])
        mvb_model.setParam(COPT.Param.RelGap, args.gap/2)
        bestObjVal = max(originalObjVal, originalObjVal_warm) if ModelSense == 1 else min(originalObjVal, originalObjVal_warm)
        mvb_callback = cbGetMVBBestTime(mvb_model.getVars(), ModelSense=ModelSense)
        mvb_model.setCallback(mvb_callback, cp.COPT.CBCONTEXT_MIPSOL)
        CPMVBArray[0] = time.time()
        mvb_model.solve()
        mvb_time = mvb_model.getAttr("SolvingTime")
        mvbObjVal = mvb_model.getAttr("BestObj")
        TimeDominance = mvb_callback.get_best_time()
        objLoss = computeObjLoss(mvbObjVal, originalObjVal, ModelSense)
        objLoss_warm = computeObjLoss(mvbObjVal, originalObjVal_warm, ModelSense) if not args.data_free else np.nan
        if objLoss >= 1e-06 and objLoss_warm >= 1e-06:
            TimeDominance = args.maxtime

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
        print(ori_time, ori_warm_time, TimeDominance)
        return results
    
    elif solver == 'gurobi':
        initgrbmodel = read(instance_path)
        initgrbmodel.setParam("TimeLimit", args.maxtime)
        initgrbmodel.setParam("Heuristics", args.heuristics)
        initgrbmodel.setParam("PreSolve", 2)
        # initgrbmodel.setParam("Threads", cpu_count()/2)
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

        if args.data_free:
            ori_warm_time = 0
            originalObjVal_warm = -ModelSense * np.Inf
            ori_warm_best_time = 0
        else:
            grbmodel = initgrbmodel.copy()
            grbmodel.setParam("MIPGap", args.gap)
            grbmodel.setAttr(GRB.Attr.Start, grbmodel.getVars(), prediction)
            TMVBWarmarray[0] = 0
            if ModelSense == 1:
                TMVBWarmarray[1] = np.Inf
                grbmodel.optimize(whenIsBestMinWarmObjFound)
            elif ModelSense == -1:
                TMVBWarmarray[1] = -np.Inf
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
        mvbsolver.setParam(fixratio=args.fixratio, threshold=args.fixthresh,tmvb=[args.fixthresh, args.tmvb],pSuccessLow = [args.psucceed_low],pSuccessUp = [args.psucceed_up])
        mvb_model = mvbsolver.getMultiVarBranch(Xpred=probs,upCut=args.upCut,lowCut=args.lowCut,ratio_involve=args.ratio_involve,ratio=[args.ratio_low,args.ratio_up])
        mvb_model.setParam("MIPGap", args.gap)
        mvb_model.setParam("MIPFocus", 2)
        mvb_model.setParam("Cuts", 1)
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
        objLoss_warm = computeObjLoss(mvbObjVal, originalObjVal_warm, ModelSense) if not args.data_free else np.nan
        print(ori_best_time, ori_warm_best_time, TimeDominance, objLoss, objLoss_warm)

        results = {
            'instance_name': instance_name,
            'rows': m,
            'cols': n,
            'acc': acc,
            'acc1': acc1,
            'acc0': acc0,
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
            model_list = mvbsolver.get_model_list(Xpred=probs,obj=mvbObjVal,isGeq=isGeq,
                                                  upCut=args.upCut,lowCut=args.lowCut,gap=0)
            model_names = ["cl", "uc", "cc"]
            times = []
            best_obj = mvbObjVal
            for i, model in enumerate(model_list):
                model.setParam("MIPGap", args.gap)
                # model.setParam("Cuts", 3)
                # model.setParam("Heuristics", 0)
                # model.setParam("MIPFocus", 3)
                model.optimize()
                runtime = model.getAttr("RunTime")
                times.append(runtime)
                try:
                    obj = model.getAttr("ObjVal")
                except:
                    obj = np.nan
                if ModelSense == 1:
                    best_obj = min(best_obj, obj)
                elif ModelSense == -1:
                    best_obj = max(best_obj, obj)
                print(f"{model_names[i]}: {runtime}, {obj}")
            
            objLoss_all = computeObjLoss(best_obj, originalObjVal, ModelSense)
            objLoss_warm_all = computeObjLoss(best_obj, originalObjVal_warm, ModelSense)
            
            results['mvb_time_all'] = mvb_time + sum(times)
            for i, model in enumerate(model_list):
                results[f"{model_names[i]}_time"] = times[i]
            results['objLoss_all'] = objLoss_all
            results['objLoss_warm_all'] = objLoss_warm_all
        
            print(ori_time, ori_warm_time, mvb_time + sum(times), times, objLoss_all, objLoss_warm_all)
        
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
import random
parser = argparse.ArgumentParser()
parser.add_argument("--maxtime", type=float, default=3600.0)
parser.add_argument("--fixthresh", type=float, default=1.1)
parser.add_argument("--fixratio", type=float, default=0.0)
parser.add_argument("--tmvb", type=float, default=0.9)
parser.add_argument("--psucceed_low", type=float, default=0.999)
parser.add_argument("--psucceed_up", type=float, default=0.9)
parser.add_argument("--ratio_low", type=float, default=0.6)
parser.add_argument("--ratio_up", type=float, default=0.0)
parser.add_argument("--gap", type=float, default=0.001)
parser.add_argument("--heuristics", type=float, default=0.05)
parser.add_argument("--solver", type=str, default='copt')
parser.add_argument("--prob_name", type=str, default='indset')
parser.add_argument("--robust", type=int, default=0)
parser.add_argument("--upCut", type=int, default=1)
parser.add_argument("--lowCut", type=int, default=1)
parser.add_argument("--ratio_involve", type=int, default=0)
parser.add_argument("--data_free", type=int, default=0)
parser.add_argument("--sample", type=int, default=0)

args = parser.parse_args()
solver = args.solver
prob_name = args.prob_name
config = get_config(prob_name)
# target_dt_names_lst = TARGET_DT_NAMES[prob_name]
target_dt_names_lst = [VAL_DT_NAMES[prob_name]]
# target_dt_names_lst = [TRAIN_DT_NAMES[prob_name]]
# target_dt_names_lst = ['transfer_2000_4']

if args.ratio_involve:
    experiment_name = f"{solver}_{args.sample}_robust_{args.robust}_df_{args.data_free}_ratio_{args.ratio_low}_{args.ratio_up}_plow_{args.psucceed_low * args.lowCut}_pup_{args.psucceed_up * args.upCut}_gap_{args.gap}_heuristics_{args.heuristics}"
else:
    experiment_name = f"{solver}_{args.sample}_robust_{args.robust}_df_{args.data_free}_tmvb_{args.tmvb}_plow_{args.psucceed_low * args.lowCut}_pup_{args.psucceed_up * args.upCut}_gap_{args.gap}_heuristics_{args.heuristics}"

for target_dt_name in target_dt_names_lst:
    instance_names = get_instance_names(prob_name, target_dt_name, args.sample)
    for instance_name in instance_names:
        try:
            data = get_data(prob_name, target_dt_name, instance_name)
            model, model_name = get_model(config, data)
            instance_path = get_instance_path(prob_name, target_dt_name, instance_name)
            if args.data_free:
                probs, prediction = get_probs_from_lp(instance_path, solver)
            else:
                probs, prediction = get_probs(config, model, data)
            results = mvb_experiment(instance_path, instance_name, solver, probs, prediction, args)
            log_results(prob_name, target_dt_name, experiment_name, results)
        except Exception as e:
            print(f"Error in {instance_name}: {e}")
            continue
