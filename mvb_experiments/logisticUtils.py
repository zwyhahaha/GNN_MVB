import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler

randomSeed = 8


def logisticTrainer(X, y):

    y = np.round(y)
    scaler = StandardScaler()
    Xscale = scaler.fit_transform(X)
    (m, n) = y.shape
    logisticLearners = []
    accuracy = np.zeros((n, 1))
    isallone = np.zeros((n, 1))
    isallzero = np.zeros((n, 1))

    for i in range(n):

        learner = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, tol=1e-03)
        # learner = LogisticRegressionCV(cv=3, penalty="l1", solver="liblinear", random_state=randomSeed,
        # max_iter=1000, tol=1e-02)

        try:
            learner.fit(Xscale, y[:, i])
        except:
            acc = 1.0
            if np.sum(y[:, i]) == m:
                isallone[i] = 1
            elif np.sum(y[:, i]) == 0:
                isallzero[i] = 1
            else:
                raise RuntimeError("Unidentified pattern in the label")
        else:
            acc = sum(learner.predict(Xscale) == y[:, i]) / 5

        accuracy[i] = acc
        logisticLearners.append(learner)

        if np.mod(i, 10) == 0:
            print("Training over the variable {0}".format(i + 1))

    return {"learners": logisticLearners, "scaler": scaler,
            "allones": isallone, "allzeros": isallzero, "acc": accuracy}


def getSol(X, scaler, learners, allones, allzeros):
    """
    Predict binary variables given learners for logistic regression

    """
    X = scaler.transform(X.reshape(1, -1))
    n = len(learners)
    Xpred = np.zeros(n)

    for i in range(n):
        if allones[i][0]:
            Xpred[i] = 1.0
        elif allzeros[i][0]:
            Xpred[i] = 0.0
        else:
            Xpred[i] = learners[i].predict_proba(X)[0][1]

    return Xpred

def logisticPredictor(logisticLearner, X):
    learners = logisticLearner["learners"]
    scaler = logisticLearner["scaler"]
    allones = logisticLearner["allones"]
    allzeros = logisticLearner["allzeros"]
    Xpred = getSol(X, scaler, learners, allones, allzeros)

    return Xpred
