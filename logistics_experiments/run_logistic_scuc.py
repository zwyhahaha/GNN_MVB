from src.logisticUtils import *
from src.MVB import *
from src.scucUtils import *

from scipy.io import savemat, loadmat
from gurobipy import *
import os
import pickle

def savelearner(fname, learner):
    with open('results/' + fname + 'learner.pkl', 'wb') as f:
        pickle.dump(learner, f, pickle.HIGHEST_PROTOCOL)


def loadlearner(fname):
    with open('results/' + fname + '.pkl', 'rb') as f:
        return pickle.load(f)


# Define a Gurobi Callback function
TMVBarray = [0]
# 0: Time  1: Is good solution found  2: Value for the solution 3: Time for good solution
TOriginalArray = [0, 0, 0, 0]


def whenIsMVBBestObjFound(model, where, time=TMVBarray):
    if where == GRB.Callback.MIPSOL:
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        objnow = model.cbGet(GRB.Callback.MIPSOL_OBJ)

        if objnow <= objbst:
            bestTime = model.cbGet(GRB.Callback.RUNTIME)
            # print("Solution found at %3g" % bestTime)
            time[0] = bestTime


def whenIsOriginalBestObjFound(model, where, time=TOriginalArray):
    if where == GRB.Callback.MIPSOL:
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        objnow = model.cbGet(GRB.Callback.MIPSOL_OBJ)

        # Get the time when the better objective is found
        if (objbst <= time[2]) and (time[1] == 0):
            time[1] = 1
            time[3] = model.cbGet(GRB.Callback.RUNTIME)

        if objnow <= objbst:
            bestTime = model.cbGet(GRB.Callback.RUNTIME)
            # print("Solution found at %3g" % bestTime)
            time[0] = bestTime


def computeObjLoss(mvbObj, originalObj):
    return (mvbObj - originalObj) / originalObj * 100


def getVarIdx(model):
    vtypes = model.getAttr(GRB.Attr.VType)
    bin_id = np.where(np.asarray(vtypes) == GRB.BINARY)[0]
    return bin_id


if __name__ == "__main__":

    prob_range = ['ieee30','ieee118']
    focus = [0, 0.05, 1]
    ntest = 20
    maxtime = 1200
    hideGRBlog = False
    seed = 20221230

    fixThreshold = 1.1
    mvbtmvb = [fixThreshold, 0.9]
    mvbpSuccedd = [0.7]
    variance = 1.0

    n_time_period = 96

    useNewModel = True

    if not os.path.exists("results"):
        os.makedirs("results")

    with open("results/log_all.txt", "a") as f:
        f.write("Experiment setup: \n")
        f.write("maxtime: {0}   ntest: {1}   fixThreshold: {2}   tmvb: {3} \n".format(maxtime,
                                                                                      ntest,
                                                                                      fixThreshold,
                                                                                      *mvbtmvb))
    f.close()
    goodlist = np.zeros((3, 3, 3))

    np.random.seed(seed)

    for i in range(len(prob_range)):

        mps_fname = "ieee_data/mps/{0}.mps".format(prob_range[i])
        base_model = read(mps_fname)
        base_rhs = getFeatureMKPGurobi(base_model)
        varIdx = getVarIdx(base_model)        

        for heuristic in focus:

            with open("results/log_all.txt", "a") as f:
                f.write("**************************************************************** \n")
                f.write("Running Experiment over %s, heur = %f \n" % (prob_range[i], heuristic))
                f.write("**************************************************************** \n")

                try:
                    X = loadmat("ieee_data/mat/train_{0}.mat".format(prob_range[i]))['data']['X'][0][0]
                    y = loadmat("ieee_data/mat/train_{0}.mat".format(prob_range[i]))['data']['y'][0][0]
                except:
                    with open("results/log_all.txt", "a") as f:
                        f.write("------------------------------------------------ \n")
                        f.write("Data not found for %s \n" % prob_range[i])
                        f.write("------------------------------------------------ \n")
                    f.close()
                    continue

                initgrbmodel = read(mps_fname)
                mvbsolver = MVB(n_time_period, len(varIdx))
                mvbsolver.registerModel(initgrbmodel, solver="gurobi")
                mvbsolver.registerVars(varIdx)
                mvbsolver.registerFeature(getFeatureMKPGurobi)
                mvbsolver.registerLearner(logisticTrainer, logisticPredictor)

                if useNewModel and heuristic == 0:
                    mvbsolver.train(X, y)
                    logisticModel = mvbsolver.getLearner()
                    savelearner("{0}_".format(prob_range[i]), logisticModel)
                else:
                    logisticModel = loadlearner("{0}_learner".format(prob_range[i]))

                mvbsolver.registerLearner(logisticTrainer, logisticPredictor, logisticModel)
                mvbsolver.setParam(threshold=fixThreshold, tmvb=mvbtmvb, pSuccess=mvbpSuccedd)

                mvbGaps = np.zeros(ntest)
                mvbRunTimes = np.zeros(ntest)
                mvbBestTimes = np.zeros(ntest)
                mvbObjs = np.zeros(ntest)

                orgGaps = np.zeros(ntest)
                orgRunTimes = np.zeros(ntest)
                orgBestTimes = np.zeros(ntest)
                orgObjs = np.zeros(ntest)

                objLosses = np.zeros(ntest)
                TimeDominance = np.zeros(ntest)

                for l in range(ntest):

                    grbmodel = initgrbmodel.copy()
                    setFeatureMKPGurobi(grbmodel, base_rhs + np.random.randn(n_time_period) * variance)
                    grbmodel.update()
                    mvbsolver.registerModel(grbmodel)
                    mvbsolver.registerVars(varIdx)
                    model = mvbsolver.getMultiVarBranch()

                    # MVB model
                    model.setParam("TimeLimit", maxtime)
                    model.setParam("Heuristics", heuristic)

                    if hideGRBlog:
                        model.setParam("LogToConsole", 0)
                        model.setParam("OutputFlag", 0)

                    model.optimize(whenIsMVBBestObjFound)

                    # Extract information
                    mvbgap = model.getAttr("MIPGap")
                    mvbruntime = model.getAttr("RunTime")
                    mvbObjVal = model.getAttr("ObjVal")
                    mvbbestTime = TMVBarray[0]

                    # Original Model
                    grbmodel.setParam("TimeLimit", maxtime)
                    grbmodel.setParam("Heuristics", heuristic)

                    if hideGRBlog:
                        grbmodel.setParam("LogToConsole", 0)
                        grbmodel.setParam("OutputFlag", 0)

                    TOriginalArray[0] = 0
                    TOriginalArray[1] = 0
                    TOriginalArray[2] = mvbObjVal
                    TOriginalArray[3] = 0

                    grbmodel.optimize(whenIsOriginalBestObjFound)

                    # Extract information
                    originalgap = grbmodel.getAttr("MIPGap")
                    originalruntime = grbmodel.getAttr("RunTime")
                    originalObjVal = grbmodel.getAttr("ObjVal")
                    originalbestTime = TOriginalArray[0]

                    if TOriginalArray[1]:
                        TimeDominance[l] = TOriginalArray[3]
                    else:
                        TimeDominance[l] = maxtime

                    objLoss = computeObjLoss(mvbObjVal, originalObjVal)
                    objLosses[l] = objLoss

                    mvbGaps[l] = mvbgap
                    mvbRunTimes[l] = mvbruntime
                    mvbObjs[l] = mvbObjVal
                    mvbBestTimes[l] = mvbbestTime

                    orgGaps[l] = originalgap
                    orgRunTimes[l] = originalruntime
                    orgObjs[l] = originalObjVal
                    orgBestTimes[l] = originalbestTime

                    # Print summary
                    with open("results/log_{0}_heur_{1}.txt".format(prob_range[i], int(heuristic * 100)), "a") as f:
                        f.write("* Done {0} out of {1}\n".format(l + 1, ntest))
                        f.write("------------------------------------------------------------ \n")
                        f.write(
                            "| Problem  |   %-8s |   %-6s | %8s |   %-6s   | \n" % (
                                "Gap", "Time", "BestTime", "Obj"))
                        f.write("------------------------------------------------------------ \n")
                        f.write(
                            "| Original |   %-6.3e|   %-6.3f | %8.3f | % -6.3e | \n" % (originalgap,
                                                                                        originalruntime,
                                                                                        originalbestTime,
                                                                                        originalObjVal))
                        f.write("------------------------------------------------------------ \n")
                        f.write("|   MVB    |   %-6.3e|   %-6.3f | %8.3f | % -6.3e | \n" % (mvbgap,
                                                                                            mvbruntime,
                                                                                            mvbbestTime,
                                                                                            mvbObjVal))
                        f.write("------------------------------------------------------------ \n")
                        f.write(
                            "| Obj Loss:  %3f %% | Time2MVB: %3f               | \n" % (objLoss, TimeDominance[l]))
                        f.write("------------------------------------------------------------\n")

                    f.close()

                with open("results/log_all.txt", "a") as f:
                    f.write("* Test complete. Printing summary \n")
                    f.write("------------------------------------------------------------\n")
                    f.write(
                        "| Problem  |   %-8s |   %-6s | %8s |   %-6s   |\n" % ("Gap", "Time", "BestTime", "Obj"))
                    f.write("------------------------------------------------------------\n")
                    f.write("| Original |   %-6.3e|   %-6.3f | %8.3f | % -6.3e |\n" % (float(np.mean(orgGaps)),
                                                                                       float(np.mean(orgRunTimes)),
                                                                                       float(np.mean(orgBestTimes)),
                                                                                       float(np.mean(orgObjs))))
                    f.write("------------------------------------------------------------\n")
                    f.write("|   MVB    |   %-6.3e|   %-6.3f | %8.3f | % -6.3e |\n" % (float(np.mean(mvbGaps)),
                                                                                       float(np.mean(mvbRunTimes)),
                                                                                       float(np.mean(mvbBestTimes)),
                                                                                       float(np.mean(mvbObjs))))
                    f.write("------------------------------------------------------------\n")
                    f.write(
                        "| Obj Loss:  %3f %% | Time2MVB: %3f                   |\n" % (float(np.mean(objLosses)),
                                                                                       float(
                                                                                           np.mean(TimeDominance))))
                    f.write("------------------------------------------------------------\n")
                    f.close()
