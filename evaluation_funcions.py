import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import pylab

# produce confusion matrix
def confusionMatrix(predictions, labels):
    d = np.unique(labels).size
    confusionM = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            confusionM[i, j] += ((predictions == i) * (labels == j)).sum()
    return confusionM

# optimal threshold with given prior
def optimal_threshold(optimum_prior):
    return -np.log(optimum_prior/(1-optimum_prior))

# given prior and costs finds a threshold
def thresholdBinary(priorProb, CostFN, CostFP):
    threshold = -np.log((priorProb * CostFN) / ((1 - priorProb) * CostFP))
    return threshold

# given prior and cost or threshold it makes prediction
def predictorBinary(llr, priorProb, Cfn, Cfp, threshold=None):
    if threshold is None:
        threshold = thresholdBinary(priorProb, Cfn, Cfp)
    P = (llr > threshold)
    return np.int32(P)

# give confusion matrix, prior and costs computes the risk
def empBayesRiskBinary(CM, priorProb, Cfn, Cfp):
    FNR, FPR = (CM[0, 1] / (CM[0, 1] + CM[1, 1])), (CM[1, 0] / (CM[1, 0] + CM[0, 0]))
    risk = (priorProb * Cfn * FNR) + ((1 - priorProb) * Cfp * FPR)
    return risk

# give confusion matrix, prior and costs computes the emprical risk
def empBayesRiskBinary2(CM, priorProb, Cfn, Cfp):
    risk = 0
    for i in range(2):
        N = CM[0, i] + CM[1, i]
        if i == 0:
            risk += (1-priorProb) * (1/N) * (CM[1, 0] * Cfp)
        else:
            risk += priorProb * (1/N) * (CM[0,1] * Cfn)
    return risk

# risk normalizer
def normalizedEmpBayesRiskBinary(CM, priorProb, Cfn, Cfp):
    Bdummy = min(priorProb * Cfn, (1 - priorProb) * Cfp)
    risk = empBayesRiskBinary(CM, priorProb, Cfn, Cfp)
    return risk / Bdummy

# actual cost calculator
def actualDCFBinary(scores, labels, priorProb, Cfn, Cfp, threshold=None):
    prediction = predictorBinary(scores, priorProb, Cfn, Cfp, threshold=threshold)
    CM = confusionMatrix(prediction, labels)
    result = normalizedEmpBayesRiskBinary(CM, priorProb, Cfn, Cfp)
    return result

# minimum cost calculator
def minDCFBinary(scores, labels, priorProb, Cfn, Cfp):
    thresholds = np.array(scores)
    thresholds.sort()
    np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])
    DCFList = []
    for t in thresholds:
        DCFList.append(actualDCFBinary(scores, labels, priorProb, Cfn, Cfp, threshold=t))
    return np.array(DCFList).min()

# my ad hoc function for finding optimum threshold
def optimum_threshold_finder(scores, labels, priorProb, Cfn, Cfp):
    thresholds = np.array(scores)
    thresholds.sort()
    np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])
    best_threshold = 0
    best_score = 100
    for t in thresholds:
        current_score = actualDCFBinary(scores, labels, priorProb, Cfn, Cfp, threshold=t)
        if current_score < best_score:
            best_threshold = t
            best_score = current_score
    return best_threshold


# Roc curve plotting
def plotRocCurveForGivenThresholds(llr, labels):
    for best in range(llr.shape[0]):
        thresholds = np.array(llr[best, :])
        thresholds.sort()
        thresholds = np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])
        FPR = np.zeros(thresholds.size)
        TPR = np.zeros(thresholds.size)
        for idx, t in enumerate(thresholds):
            Pred = np.int32(llr[best, :] > t)
            CM = confusionMatrix(Pred, labels)
            TPR[idx] = CM[1, 1] / (CM[1, 1] + CM[0, 1])
            FPR[idx] = CM[1, 0] / (CM[0, 0] + CM[1, 0])
        if best == 0:
            plt.plot(FPR, TPR, color='blue', alpha=0.75, label='SVM')
        if best == 1:
            plt.plot(FPR, TPR, color='red', alpha=0.75, label='GMM')
        if best == 2:
            plt.plot(FPR, TPR, color='green', alpha=0.75, label='fusion')

    plt.legend()
    plt.show()

# bayes error plotting
def bayes_error_plot(pArray, scores, labels, minCost=False, threshold=None):
    y = []
    for i, p in enumerate(pArray):
        pi = 1.0 / (1.0 + np.exp(-p))
        if minCost:
            y.append(minDCFBinary(scores, labels, pi, 1, 1))
        else:
            y.append(actualDCFBinary(scores, labels, pi, 1, 1, threshold=threshold))
    return np.array(y)

# bayes error plotting for 3 different threshold to optimum threshold experiment
def bayes_error_plot_2(pArray, scores, labels, minCost=False, threshold=None):
    y = []
    if (threshold != None):
        c_th = threshold[2]
    C = pArray.size
    for i, p in enumerate(pArray):
        if (i > C * 0.33) & (i < C * 0.66) & (threshold != None):
            c_th = threshold[1]
        if (i >= C * 0.66) & (threshold is not None):
            c_th = threshold[0]
        pi = 1.0 / (1.0 + np.exp(-p))
        if minCost:
            y.append(minDCFBinary(scores, labels, pi, 1, 1))
        else:
            y.append(actualDCFBinary(scores, labels, pi, 1, 1, threshold=c_th))
    return np.array(y)
