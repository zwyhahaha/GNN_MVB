import numpy as np

try:
    from mvb_experiments.logisticUtils import *
    from mvb_experiments.MVB import *
    from mkpUtils import *
except:
    import sys
    import os
    sys.path.append(os.path.join("..", ".."))
    from logisticUtils import *
    from MVB import *
    from mkpUtils import *

from scipy.io import savemat, loadmat
from gurobipy import *
import pickle
import argparse

# os.system("rm *.txt")
# os.system("delete *.txt")

randomSeed = 24


def savelearner(fname, learner):
    with open(fname + 'learner.pkl', 'wb') as f:
        pickle.dump(learner, f, pickle.HIGHEST_PROTOCOL)


def loadlearner(fname):
    with open(fname + '.pkl', 'rb') as f:
        return pickle.load(f)


# Define a Gurobi Callback function
TMVBarray = [0, 0]
# 0: Time  1: Is good solution found  2: Value for the solution 3: Time for good solution
TOriginalArray = [0, 0, 0, 0]


def callBackDummy(model, where, time=TOriginalArray):
    if where == GRB.Callback.MIPSOL:
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        pass


def whenIsBestObjFound(model, where, time=TMVBarray):
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

        if objbst >= time[2]:
            time[1] = 1
            time[3] = model.cbGet(GRB.Callback.RUNTIME)
            model.terminate()


def whenIsOriginalBestObjFound(model: Model, where, time=TOriginalArray):
    if where == GRB.Callback.MIPSOL:
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        objnow = model.cbGet(GRB.Callback.MIPSOL_OBJ)

        # Get the time when the better objective is found
        if (objbst >= time[2]) and (time[1] == 0):
            time[1] = 1
            time[3] = model.cbGet(GRB.Callback.RUNTIME)
            model.terminate()

        if objnow >= objbst:
            bestTime = model.cbGet(GRB.Callback.RUNTIME)
            # print("Solution found at %3g" % bestTime)
            time[0] = bestTime


def computeObjLoss(mvbObj, originalObj):
    return (originalObj - mvbObj) / originalObj * 100


parser = argparse.ArgumentParser()
parser.add_argument("--ntest", type=int, default=5)
parser.add_argument("--maxtime", type=float, default=120.0)
parser.add_argument("--fixthresh", type=float, default=1.1)
parser.add_argument("--psucceed", type=float, default=0.75)
parser.add_argument("--warm", type=int, default=0)
parser.add_argument("--gap", type=float, default=0.0001)

if __name__ == "__main__":

    args = parser.parse_args()

    mlist = [10, 30]
    nlist = [250, 500]
    tlist = [25]
    focus = [0.1, 1]
    ntest = args.ntest
    maxtime = args.maxtime
    hideGRBlog = False
    gap = args.gap

    warm = args.warm

    fixThreshold = args.fixthresh
    mvbtmvb = [fixThreshold, 0.9]
    mvbpSucceed = [args.psucceed]

    excludeList = []
    useNewModel = True

    log_all_name = "log_all_warm_{0}_maxtime_{1}_fixthresh_{2}_psucceed_{3}_gap_{4}.txt".format(
        warm, maxtime, fixThreshold, mvbpSucceed[0], gap)

    with open(log_all_name, "a") as f:
        f.write("Experiment setup: \n")
        setting = "maxtime: {0}   ntest: {1}   fixThreshold: {2}   tmvb: {3}  warm: {4}  gap: {5}\n".format(
            maxtime, ntest, fixThreshold, *mvbtmvb, warm, gap)
        f.write(setting)

    f.close()

    goodlist = np.zeros((3, 3, 3))

    for i in range(len(mlist)):
        for j in range(len(nlist)):
            for k in range(len(tlist)):

                m = mlist[i]
                n = nlist[j]
                t = tlist[k]

                if (m, n) in excludeList:
                    continue

                for heuristic in focus:

                    with open(log_all_name, "a") as f:
                        f.write("**************************************************************** \n")
                        f.write("Running Experiment over m = %d, n = %d, t = %d, heur = %f \n" % (m, n, t, heuristic))
                        f.write("**************************************************************** \n")

                    varIdx = list(range(n))
                    path = "test"

                    try:
                        X = loadmat(os.path.join(path, "m_{0}_n_{1}_t_{2}_Xtrain.mat".format(m, n, t)))["X"]
                        y = loadmat(os.path.join(path, "m_{0}_n_{1}_t_{2}_ytrain.mat".format(m, n, t)))["y"]
                        Xtest = loadmat(os.path.join(path, "m_{0}_n_{1}_t_{2}_Xtest.mat".format(m, n, t)))["X"]
                        Xtest = Xtest[10:, :]
                    except:
                        with open(log_all_name, "a") as f:
                            f.write("------------------------------------------------ \n")
                            f.write("Data not found for m = %d, n = %d, t = %d \n" % (m, n, t))
                            f.write("------------------------------------------------ \n")
                        f.close()
                        continue

                    initgrbmodel = read(os.path.join("..", "data", "mps", "model_m_{0}_n_{1}_t_{2}.mps".format(m, n, t)))
                    mvbsolver = MVB(m, n)
                    mvbsolver.registerModel(initgrbmodel, solver="gurobi")
                    mvbsolver.registerVars(varIdx)
                    mvbsolver.registerFeature(getFeatureMKPGurobi)
                    mvbsolver.registerLearner(logisticTrainer, logisticPredictor)

                    if useNewModel and heuristic == focus[0]:
                        mvbsolver.train(X, y)
                        logisticModel = mvbsolver.getLearner()
                        savelearner("m_{0}_n_{1}_t_{2}".format(m, n, t), logisticModel)
                    else:
                        logisticModel = loadlearner("m_{0}_n_{1}_t_{2}learner".format(m, n, t))

                    mvbsolver.registerLearner(logisticTrainer, logisticPredictor, logisticModel)
                    mvbsolver.setParam(threshold=fixThreshold, tmvb=mvbtmvb, pSuccess=mvbpSucceed)

                    mvbGaps = np.zeros(ntest)
                    mvbRunTimes = np.zeros(ntest)
                    mvbObjs = np.zeros(ntest)

                    orgGaps = np.zeros(ntest)
                    orgRunTimes = np.zeros(ntest)
                    orgBestTimes = np.zeros(ntest)
                    orgObjs = np.zeros(ntest)

                    objLosses = np.zeros(ntest)
                    TimeDominance = np.zeros(ntest)

                    for l in range(ntest):

                        grbmodel = initgrbmodel.copy()
                        currentX = Xtest[l, :]

                        grbmodel.setAttr("RHS", grbmodel.getConstrs(), currentX)
                        grbmodel.update()
                        mvbsolver.registerModel(grbmodel)
                        mvbsolver.registerVars(varIdx)

                        model = mvbsolver.getMultiVarBranch(warm=warm)

                        # Original model
                        grbmodel.setParam("TimeLimit", maxtime)
                        grbmodel.setParam("Heuristics", heuristic)
                        grbmodel.setParam("MIPGap", gap)
                        grbmodel.setParam("Threads", 8)

                        if hideGRBlog:
                            grbmodel.setParam("LogToConsole", 0)
                            grbmodel.setParam("OutputFlag", 0)

                        TMVBarray[0] = 0
                        TMVBarray[1] = 0
                        grbmodel.optimize(whenIsBestObjFound)

                        # Extract information
                        originalgap = grbmodel.getAttr("MIPGap")
                        originalruntime = grbmodel.getAttr("RunTime")
                        originalObjVal = grbmodel.getAttr("ObjVal")

                        # The time when Gurobi achieves best time
                        originalbesttime = TMVBarray[0]

                        # MVB Model
                        model.setParam("TimeLimit", 3600)
                        model.setParam("Heuristics", heuristic)
                        model.setParam("MIPGap", 0.0)
                        model.setParam("Threads", 8)

                        if hideGRBlog:
                            model.setParam("LogToConsole", 0)
                            model.setParam("OutputFlag", 0)

                        TOriginalArray[0] = 0
                        TOriginalArray[1] = 0
                        TOriginalArray[2] = originalObjVal
                        TOriginalArray[3] = 0

                        model.optimize(whenIsMVBObjFound)

                        # Extract information
                        mvbruntime = model.getAttr("RunTime")
                        mvbObjVal = model.getAttr("ObjVal")

                        if TOriginalArray[1]:
                            TimeDominance[l] = TOriginalArray[3]
                        else:
                            TimeDominance[l] = maxtime

                        objLoss = computeObjLoss(mvbObjVal, originalObjVal)
                        objLosses[l] = objLoss

                        mvbRunTimes[l] = mvbruntime
                        mvbObjs[l] = mvbObjVal

                        orgGaps[l] = originalgap
                        orgRunTimes[l] = originalruntime
                        orgObjs[l] = originalObjVal
                        orgBestTimes[l] = originalbesttime

                        # Print summary
                        with open("log_m_{0}_n_{1}_t_{2}_heur_{3}_warm_{4}_maxtime_{5}"
                                  "_fixthresh_{6}_psucceed_{7}_gap_{8}.txt".format(
                            m, n, t, int(heuristic * 100), warm, maxtime, fixThreshold, mvbpSucceed[0], gap
                        ), "a") as f:
                            f.write("%d  | %5.5f | %5.5f  %5.5f | %8.5e  %8.5e |\n" %
                                    (l + 1, originalruntime, originalbesttime, mvbruntime, originalObjVal, mvbObjVal))
                        f.close()

                    with open(log_all_name, "a") as f:

                        f.write("OTime: %5.5f | MVBTime: %5.5f | Loss: %f \n" %
                                (float(np.mean(orgBestTimes)),
                                 float(np.mean(TimeDominance)),
                                 float(np.mean(objLosses))))

                        f.close()
