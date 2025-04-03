from MIPGNN.global_vars import *
from main_utils import *
from MVB import *
from gurobipy import *
import coptpy as cp
from coptpy import COPT
import numpy as np
import time

TMVBarray = [0, 0]
TMVBWarmarray = [0, 0]
TOriginalArray = [0, 0, 0, 0]

CPOriginalArray = [0]
CPWarmArray = [0]
CPMVBArray = [0,0]

def whenIsBestMinObjFound(model, where, time=TMVBarray):
    if where == GRB.Callback.MIPSOL:
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)

        if objbst < time[1]:
            bestTime = model.cbGet(GRB.Callback.RUNTIME)
            time[0] = bestTime
            time[1] = objbst

def whenIsBestMaxObjFound(model, where, time=TMVBarray):
    if where == GRB.Callback.MIPSOL:
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)

        if objbst > time[1]:
            bestTime = model.cbGet(GRB.Callback.RUNTIME)
            time[0] = bestTime
            time[1] = objbst

def whenIsBestMinWarmObjFound(model, where, time=TMVBWarmarray):
    if where == GRB.Callback.MIPSOL:
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)

        if objbst < time[1]:
            bestTime = model.cbGet(GRB.Callback.RUNTIME)
            time[0] = bestTime
            time[1] = objbst

def whenIsBestMaxWarmObjFound(model, where, time=TMVBWarmarray):
    if where == GRB.Callback.MIPSOL:
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)

        if objbst > time[1]:
            bestTime = model.cbGet(GRB.Callback.RUNTIME)
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

class cbGetBestTime(cp.CallbackBase):
    
    def __init__(self, vars, ModelSense = -1):
        super().__init__()
        self._vars = vars
        self._ModelSense = ModelSense
        self._bestTime = 3600
        self._bestObj = np.Inf if ModelSense == 1 else -np.Inf

    def callback(self):
        try:
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
        except Exception as e:
            print(f"Exception in callback: {e}")

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
        try:
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
        except Exception as e:
            print(f"Exception in callback: {e}")

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
        self._bestObj = CPMVBArray[1]

    def callback(self):
        if self.where() == cp.COPT.CBCONTEXT_MIPSOL:
            objbst = self.getInfo("BestObj")
            objnow = self.getInfo("MipCandObj")
            if self._ModelSense == 1:
                objbst = min(objbst, objnow)
                if (objbst - self._bestObj)/self._bestObj <= 1e-06:
                    self._bestTime = time.time() - CPMVBArray[0]
                    self._bestObj = objbst
                    self.interrupt()
            elif self._ModelSense == -1:
                objbst = max(objbst, objnow)
                if (self._bestObj - objbst)/self._bestObj <= 1e-06:
                    self._bestTime = time.time() - CPMVBArray[0]
                    self._bestObj = objbst
                    self.interrupt()

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
        CPMVBArray[1] = max(originalObjVal, originalObjVal_warm) if ModelSense == 1 else min(originalObjVal, originalObjVal_warm)
        mvb_callback = cbGetMVBBestTime(cp_model.getVars(), ModelSense)
        mvb_model.setCallback(mvb_callback, cp.COPT.CBCONTEXT_MIPSOL)
        CPMVBArray[0] = time.time()
        mvb_model.solve()
        mvb_time = mvb_model.getAttr("SolvingTime")
        mvbObjVal = mvb_model.getAttr("BestObj")
        TimeDominance = mvb_callback.get_best_time()
        objLoss = computeObjLoss(mvbObjVal, originalObjVal, ModelSense)
        objLoss_warm = computeObjLoss(mvbObjVal, originalObjVal_warm, ModelSense) if not args.data_free else np.nan
        if TimeDominance == args.maxtime and objLoss <= 1e-06 and objLoss_warm <= 1e-06:
            TimeDominance = mvb_time

        results = get_results(instance_name, m, n, ori_time, ori_warm_time, mvb_time, ori_best_time, ori_warm_best_time, TimeDominance, objLoss, objLoss_warm)
        return results
    elif solver == 'gurobi':
        initgrbmodel = read(instance_path)
        initgrbmodel.setParam("TimeLimit", args.maxtime)
        initgrbmodel.setParam("Heuristics", args.heuristics)
        initgrbmodel.setParam("PreSolve", 2)
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
        mvb_model.setParam("MIPGap", args.gap) if args.robust else mvb_model.setParam("MIPGap", args.gap/2)
        if args.robust:
            mvb_model.setParam("Cuts", 1)
            mvb_model.setParam("MIPFocus", 2)
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
        mvbObjVal = mvb_model.getAttr("ObjVal")
        
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
        
        results = get_results(instance_name, m, n, ori_time, ori_warm_time, mvb_time, ori_best_time, ori_warm_best_time, TimeDominance, objLoss, objLoss_warm)

        if args.robust:
            isGeq = 1 if ModelSense == -1 else 0
            model_list = mvbsolver.get_model_list(Xpred=probs,obj=mvbObjVal,isGeq=isGeq,
                                                  upCut=args.upCut,lowCut=args.lowCut,gap=0)
            model_names = ["cl", "uc", "cc"]
            times = []
            best_obj = mvbObjVal
            for i, model in enumerate(model_list):
                model.setParam("MIPGap", args.gap)
                model.optimize()
                runtime = model.getAttr("RunTime")
                times.append(runtime)
                obj = model.getAttr("ObjVal")
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

            print(f"{'='*20} Solving Time {instance_name} {'='*20}")
            print(f"{'Original Time':<25}{'Original (Warm) Time':<30}{'MVB Time':<20}")
            print(f"{ori_time:<25.3f}{ori_warm_time:<30.3f}{mvb_time + sum(times):<20.3f}")
            print(f"{'='*63}")
        
        return results
    else:
        raise ValueError("Solver not supported. Please use 'gurobi' or 'copt'.")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--prob_name", type=str, default='indset', choices=PROB_NAMES)
parser.add_argument("--dt_name", type=str, default='valid', help="dataset name")
parser.add_argument("--robust", type=int, default=0, help="0: primal heuristic experiments, 1: braching rule experiments")
parser.add_argument("--data_free", type=int, default=0, help="0: use GNN prediction, 1: use LP relaxation")
parser.add_argument("--solver", type=str, default='gurobi', choices=['gurobi', 'copt'])
parser.add_argument("--maxtime", type=float, default=3600.0)
parser.add_argument("--gap", type=float, default=0.001)
parser.add_argument("--heuristics", type=float, default=0.05)
parser.add_argument("--fixthresh", type=float, default=1.1)
parser.add_argument("--fixratio", type=float, default=0.0)
parser.add_argument("--tmvb", type=float, default=0.9)
parser.add_argument("--psucceed_low", type=float, default=0.9)
parser.add_argument("--psucceed_up", type=float, default=0.9)
parser.add_argument("--upCut", type=int, default=1)
parser.add_argument("--lowCut", type=int, default=1)
parser.add_argument("--ratio_involve", type=int, default=0)
parser.add_argument("--ratio_low", type=float, default=0.8)
parser.add_argument("--ratio_up", type=float, default=0.0)

args = parser.parse_args()
solver = args.solver
prob_name = args.prob_name
config = get_config(prob_name)
dt_name = args.dt_name

if dt_name == 'target':
    target_dt_names_lst = TARGET_DT_NAMES[prob_name]
elif dt_name == 'train':
    target_dt_names_lst = [TRAIN_DT_NAMES[prob_name]]
elif dt_name == 'valid':
    target_dt_names_lst = [VAL_DT_NAMES[prob_name]]
else:
    target_dt_names_lst = [dt_name]

if args.ratio_involve:
    experiment_name = f"{solver}_robust_{args.robust}_df_{args.data_free}_ratio_{args.ratio_low}_{args.ratio_up}_plow_{args.psucceed_low * args.lowCut}_pup_{args.psucceed_up * args.upCut}_gap_{args.gap}_heuristics_{args.heuristics}"
else:
    experiment_name = f"{solver}_robust_{args.robust}_df_{args.data_free}_tmvb_{args.tmvb}_plow_{args.psucceed_low * args.lowCut}_pup_{args.psucceed_up * args.upCut}_gap_{args.gap}_heuristics_{args.heuristics}"

for target_dt_name in target_dt_names_lst:
    instance_names = get_instance_names(prob_name, target_dt_name)
    for instance_name in instance_names:
        try:
            data = get_data(prob_name, target_dt_name, instance_name)
            model, model_name = get_model(prob_name, config, data)
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
