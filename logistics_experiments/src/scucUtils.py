import numpy as np

n_time_period = 96
constr_prefix = "P_D_constraint"
constr_index = list(range(n_time_period))
constr_names = ["{0}_{1}".format(constr_prefix, t) for t in range(n_time_period)]


def getFeatureMKPGurobi(model):
    constrs = [model.getConstrByName(constr_names[i]) for i in range(n_time_period)]
    return np.asarray(model.getAttr("RHS", constrs))


def setFeatureMKPGurobi(model, rhs):
    constrs = [model.getConstrByName(constr_names[i]) for i in range(n_time_period)]
    model.setAttr("RHS", constrs, rhs)


def getFeatureMKPCPLEX(model):
    return np.asarray(model.linear_constraints.get_rhs()[0:96])


def getFeatureMKPSCIP(model):
    # Not yet implemented
    pass
