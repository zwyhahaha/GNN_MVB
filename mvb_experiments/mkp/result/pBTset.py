import numpy as np
from scipy.io import savemat, loadmat
import cplex


def genTset(m, n, t, ntrain, ntest):
    model = cplex.Cplex()
    model.read("../data/mps/model_m_{0}_n_{1}_t_{2}.mps".format(m, n, t))
    model.objective.set_sense(model.objective.sense.maximize)
    model.solve()
    rhs    = np.asarray(model.linear_constraints.get_rhs())
    ub     = rhs * 1.2
    lb     = rhs * 0.8
    newrhs = np.random.uniform(lb, ub, (ntrain + ntest, m))
    sols   = np.zeros((ntrain, n))
    cnames = model.linear_constraints.get_names()
    isopt  = np.zeros((ntrain, 1))
    nopt   = 0

    # Save once 
    savemat("m_{0}_n_{1}_t_{2}_Xtrain.mat".format(m, n, t), {"X": newrhs[0 : ntrain]})
    savemat("m_{0}_n_{1}_t_{2}_ytrain.mat".format(m, n, t), {"y": sols})
    savemat("m_{0}_n_{1}_t_{2}_Xtest.mat".format(m, n, t), {"X": newrhs[ntrain:]})

    for i in range(ntrain):
        
        model.linear_constraints.set_rhs(zip(cnames, newrhs[i].tolist()))
        model.parameters.mip.strategy.heuristiceffort.set(1000)
        model.parameters.mip.tolerances.mipgap.set(0.001)
        model.parameters.timelimit.set(300)

        model.solve()
        
        status = model.solution.get_status()
        if (status == model.solution.status.optimal_tolerance or status == model.solution.status.optimal):
            isopt[i] = 1
            nopt += 1
        
        sols[i, :] = model.solution.get_values()
        savemat("m_{0}_n_{1}_t_{2}_Xtrain.mat".format(m, n, t), {"X": newrhs[0:ntrain]})
        savemat("m_{0}_n_{1}_t_{2}_ytrain.mat".format(m, n, t), {"y": sols})
        print("Done {0} of {1}".format(i + 1, ntrain))
        print("nSuccess: {0}".format(nopt))


if __name__ == "__main__":

    mlist  = [5, 10, 30]
    nlist  = [100, 250, 500]
    tlist  = [25, 50, 75]

    ntrain = 500
    ntest  = 50

    for i in range(3):
        for j in range(3):
            for k in range(3):

                m = mlist[i]
                n = nlist[j]
                t = tlist[k]
                print("\n Start with m = {0} n = {1} t = {2} \n".format(m, n, t))
                genTset(m, n, t, ntrain, ntest)
                print("\n Done with m = {0} n = {1} t = {2} \n".format(m, n, t))

print("Done with all datasets")