import numpy as np


def getFeatureMKPGurobi(model):
    return np.asarray(model.getAttr("RHS"))


def getFeatureMKPCPLEX(model):
    return np.asarray(model.linear_constraints.get_rhs())


def getFeatureMKPSCIP(model):
    # Not yet implemented
    pass
