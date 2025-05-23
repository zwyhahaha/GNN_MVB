import numpy as np
from gurobipy import *


# import cplex


class MVB(object):
    """
    A class for multi-variable branching framework

    """
    # Learner status
    MVB_LEARNER_UNITIALIZED = 100
    MVB_LEARNER_INITIALZED = 101
    MVB_LEARNER_TRAINED = 102

    # Model status
    MVB_MODEL_UNINITIALIZED = 103
    MVB_MODEL_INITIALIZED = 104

    # Solver used
    MVB_SOLVER_GUROBI = 105
    MVB_SOLVER_CPLEX = 106
    MVB_SOLVER_SCIP = 107

    def __init__(self, mdim, ndim):

        self._mdim = mdim
        self._ndim = ndim
        self._learner = None
        self._model = None
        self._mvbmodel = None
        self._learnerStatus = self.MVB_LEARNER_UNITIALIZED
        self._modelStatus = self.MVB_MODEL_UNINITIALIZED
        self._solver = self.MVB_SOLVER_GUROBI
        self._varIdx = list(range(ndim))
        self._vars = None
        self._getFeature = None
        self._getLearner = None
        self._predictor = None
        self._threshold = 1.0
        self._tmvb = [self._threshold, 0.9]
        self._pSuccess = [0.95]
        self._nRegion = len(self._tmvb) - 1

        return

    def registerLearner(self, trainer, predictor, learner=None):

        """
        Note that for adaptivity the learner here is an abstract instance
        that supports the

        ******
        Training Method (including cross validation)

        Given training set (X, Y), the learner should support

        - learner = self._getLearner(X, y)

        and this gets the model initialized

        ******
        Predicting Method

        Given a trained learner, the learner should suppory

        - Xpred = self._predict(learner, feature)

        to output the probabilistic estimation of the (binary) variables
        """

        self._getLearner = trainer
        self._predictor = predictor
        self._learnerStatus = self.MVB_LEARNER_INITIALZED

        if learner is not None:
            self._learner = learner
            self._learnerStatus = self.MVB_LEARNER_TRAINED

        print("Learner has been registered")

        return

    def registerModel(self, model, solver="gurobi"):

        if solver == "gurobi":
            self._registerGurobiModel(model)
        elif solver == "scip":
            self._registerSCIPModel(model)
        else:
            self._registerSCIPModel(model)

        self._mvbmodel = self._cloneModel()
        print("Model has been registered")

        return

    def registerVars(self, varIdx):

        if self._modelStatus != self.MVB_MODEL_INITIALIZED:
            raise RuntimeError("Model is not initialized")

        if len(varIdx) > self._ndim:
            raise RuntimeError("Variable index cannot exceed the "
                               " total number of variables")
        self._varIdx = varIdx

        if self._solver == self.MVB_SOLVER_GUROBI:
            self._vars = [self._mvbmodel.getVars()[i] for i in self._varIdx]
        else:
            # CPLEX does not support extration of variables directly
            return

        return

    def registerFeature(self, getFeatures):

        """
        Register the extraction utility of the feature given a model

        Function getFeatures should support

        feature = getFeatures(model)

        """
        self._getFeature = getFeatures

        return

    def train(self, X, Y):

        """
        Train the registered learner

        Note that X is of shape (nSample, mdim)
         and that Y is of shape (nSample, ndim)

        """
        # Check dimension
        (nSample1, mDim) = X.shape
        (nSample2, nDim) = Y.shape

        assert (nSample1 == nSample2)
        assert (mDim == self._mdim)
        assert (nDim == self._ndim)

        if self._learnerStatus == self.MVB_LEARNER_UNITIALIZED:
            raise RuntimeError("Learner is not registered")
        elif self._modelStatus == self.MVB_MODEL_UNINITIALIZED:
            raise RuntimeError("Model is not initialized")

        self._learner = self._getLearner(X, Y)
        self._learnerStatus = self.MVB_LEARNER_TRAINED

        return

    def getMultiVarBranch(self):

        """
        Implement the MVB framework to solve the embedded model

        """
        if self._learnerStatus != self.MVB_LEARNER_TRAINED:
            raise RuntimeError("Learner is not trained")
        elif self._modelStatus == self.MVB_MODEL_UNINITIALIZED:
            raise RuntimeError("Model is not initialized")

        feature = self._getFeature(self._model).reshape(-1, 1)
        Xpred = self._predictor(self._learner, feature)

        (fixUpIdx, fixLowIdx) = self._getFixIdx(Xpred, self._threshold)

        self._fixVars(fixUpIdx, isUpper=True)
        self._fixVars(fixLowIdx, isUpper=False)

        for i in range(self._nRegion):
            (mvbUpIdx, mvbUpProb, mvbLowIdx, mvbLowProb) = self._getMVBIdx(Xpred, self._tmvb[i + 1], self._tmvb[i])
            (ksiUp, ksiLow) = self._getHoeffdingBound(mvbUpProb, mvbLowProb, self._pSuccess[i])

            # Run MVB
            self._addCut(mvbUpIdx, ksiUp, isGeq=True)
            self._addCut(mvbLowIdx, ksiLow, isGeq=False)

            print("- {0} variables are involved in the MVB within interval [{1}, {2})".format(len(mvbUpIdx) +
                                                                                              len(mvbLowIdx),
                                                                                              self._tmvb[i + 1],
                                                                                              self._tmvb[i]))

        print("MVB model is generated")

        return self._mvbmodel

    def getLearner(self):

        if self._learnerStatus != self.MVB_LEARNER_TRAINED:
            raise RuntimeError("Learner is not trained")

        return self._learner

    def setParam(self, threshold, tmvb, pSuccess):

        # Recall that tmvb = [threshold, t1, t2, ..., tn]
        nRegion = len(tmvb) - 1
        assert (threshold == tmvb[0])
        assert (nRegion == len(pSuccess))

        for i in range(nRegion - 1):
            if tmvb[i] < tmvb[i + 1]:
                raise RuntimeError("tmvb must be a descending array")

        self._threshold = threshold
        self._tmvb = tmvb
        self._pSuccess = pSuccess
        self._nRegion = nRegion

        return

    def _cloneModel(self):

        if self._modelStatus != self.MVB_MODEL_INITIALIZED:
            raise RuntimeError("Model is not initialized")

        if self._solver == self.MVB_SOLVER_GUROBI:
            return self._model.copy()
        elif self._solver == self.MVB_SOLVER_GUROBI:
            return self._model.clone()
        else:
            raise NotImplementedError("Support for SCIP solver is not yet added")

    @staticmethod
    def _getFixIdx(Xpred, threshold):

        """
        Get Fixed variable indices

        """
        fixUpIdx = np.where(Xpred >= threshold)[0]
        fixLowIdx = np.where(Xpred <= 1 - threshold)[0]

        return fixUpIdx, fixLowIdx

    @staticmethod
    def _getMVBIdx(Xpred, tLow, tUp):

        """
        Get MVB indices

        """
        mvbUpIdx = np.where((Xpred >= tLow) & (Xpred < tUp))[0]
        mvbUpProb = Xpred[mvbUpIdx]
        mvbLowIdx = np.where((Xpred <= 1 - tLow) & (Xpred > 1 - tUp))[0]
        mvbLowProb = Xpred[mvbLowIdx]

        return mvbUpIdx, mvbUpProb, mvbLowIdx, mvbLowProb

    @staticmethod
    def _getHoeffdingBound(mvbUpProb, mvbLowProb, pSuccess):

        """
        Compute the tail bound based on Hoeffding's inequality

        """
        mvbUpProb = mvbUpProb.reshape(-1)
        mvbLowProb = mvbLowProb.reshape(-1)
        nUpVars = len(mvbUpProb)
        nLowVars = len(mvbLowProb)
        pFail = 1 - pSuccess
        ksiUp = np.sum(mvbUpProb) - np.sqrt(nUpVars * np.log(1 / np.maximum(pFail, 1e-03)) / 2)
        ksiUp = np.minimum(ksiUp, nUpVars)
        ksiLow = np.sum(mvbLowProb) + np.sqrt(nLowVars * np.log(1 / np.maximum(pFail, 1e-03)) / 2)
        ksiLow = np.maximum(ksiLow, 0)

        return ksiUp, ksiLow

    def _addCut(self, varIdx, bound, isGeq):

        nCutVars = len(varIdx)
        if isGeq:
            self._mvbmodel.addConstr(quicksum(self._vars[varIdx[i]] for i in range(nCutVars))
                                     >= np.ceil(bound))
        else:
            self._mvbmodel.addConstr(quicksum(self._vars[varIdx[i]] for i in range(nCutVars))
                                     <= np.floor(bound))
        return

    def _fixVars(self, varIdx, isUpper):

        nFixVars = len(varIdx)
        if isUpper:
            self._mvbmodel.addConstr(quicksum(self._vars[varIdx[i]] for i in range(nFixVars))
                                     >= nFixVars)
        else:
            self._mvbmodel.addConstr(quicksum(self._vars[varIdx[i]] for i in range(nFixVars))
                                     <= 0)
        return

    def _registerCplexModel(self, cplexModel):

        self._solver = self.MVB_SOLVER_CPLEX
        self._model = cplexModel
        self._modelStatus = self.MVB_MODEL_INITIALIZED

        return

    def _registerGurobiModel(self, gurobiModel):

        self._solver = self.MVB_SOLVER_GUROBI
        self._model = gurobiModel
        self._modelStatus = self.MVB_MODEL_INITIALIZED

        return

    def _registerSCIPModel(self, scipModel):

        self._solver = self.MVB_SOLVER_SCIP
        self._model = scipModel
        self._modelStatus = self.MVB_MODEL_INITIALIZED

        return
