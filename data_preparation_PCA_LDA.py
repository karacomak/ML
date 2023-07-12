import numpy as np
import sys
from plottingFunctions import *
import scipy.linalg
from scipy.stats import norm, rankdata, ranksums
import itertools

# column matrix maker
def oneToX(array):
    return array.reshape((1, array.size))
# row matrix maker
def xToOne(array):
    return array.reshape((array.size, 1))

#flatten given list
def flatten(lists):
    merged = list(itertools.chain.from_iterable(lists))
    return merged

#data loader and makes it array
def loadData(file):
    try:
        with open(file, 'r') as f:
            data = []
            labels = []
            for line in f:
                attributes = line.split(',')[0:12]
                attributes = xToOne(np.array([float(i) for i in attributes]))
                label = line.split(',')[12].strip()
                data.append(attributes)
                labels.append(label)
            data = np.hstack(data)
            labels = np.asfarray(labels, dtype=float)
            return data, labels

    except:
        raise

# z score normalization
def z_score(matrix):
    _, means = covarianceMatrix(matrix)
    differences = (matrix - means)**2
    sum = differences.sum(1)
    standard_deviations = (sum / (matrix.shape[1] -1))**0.5
    normalized_matrix = (matrix - means) / xToOne(standard_deviations)
    return normalized_matrix

# z score normalization for test set everything calculated with train data information
def z_score_test(matrix, test_matrix):
    _, means = covarianceMatrix(matrix)
    differences = (matrix - means)**2
    sum = differences.sum(1)
    standard_deviations = (sum / (matrix.shape[1] -1))**0.5
    normalized_matrix = (test_matrix - means) / xToOne(standard_deviations)
    return normalized_matrix

# covariance matrix calculator
def covarianceMatrix(matrix):
    means = xToOne(matrix.mean(1))
    C = 1.0/float(matrix.shape[1]) * np.dot((matrix-means), (matrix-means).T)
    return C, means

# Principal component analysis and reducer of dimension with required dimension number
def PCA(data, k):
    cMatrix, means = covarianceMatrix(data)
    s, U = np.linalg.eigh(cMatrix)
    P = U[:, ::-1][:, 0:k]
    PCAedData = np.dot(P.T, data)
    return PCAedData

# Principal component analysis and reducer of dimension with required dimension number for test set
def PCA_test(data, test_data, k):
    cMatrix, means = covarianceMatrix(data)
    s, U = np.linalg.eigh(cMatrix)
    P = U[:, ::-1][:, 0:k]
    PCAedData = np.dot(P.T, test_data)
    return PCAedData


# Linear Discriminant Analysis and reducer of dimension with required dimension number
def LDA(data, labels, k):
    SB = 0
    SW = 0
    globalMeans = xToOne(data.mean(1))
    for i in np.unique(labels):
        currentData = data[:, labels==i]
        means = xToOne(currentData.mean(1))
        currentSB = np.dot((means-globalMeans), (means-globalMeans).T) * currentData.shape[1]
        SB += currentSB

        C, _ = covarianceMatrix(currentData)
        SW += C * currentData.shape[1]
    SB = SB / data.shape[1]
    SW = SW / data.shape[1]

    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:k]

    # If we want, we can find a basis U for the subspace spanned by W using the singular value decomposition of W
    UW, s, _ = np.linalg.svd(W)
    U = UW[:, 0:k]

    P = W[:, 0:k]
    LDAedData = np.dot(P.T, data)

    return LDAedData

# Linear Discriminant Analysis and reducer of dimension with required dimension number for test set
def LDA_test(data, labels, test_data, k):
    SB = 0
    SW = 0
    globalMeans = xToOne(data.mean(1))
    for i in np.unique(labels):
        currentData = data[:, labels==i]
        means = xToOne(currentData.mean(1))
        currentSB = np.dot((means-globalMeans), (means-globalMeans).T) * currentData.shape[1]
        SB += currentSB

        C, _ = covarianceMatrix(currentData)
        SW += C * currentData.shape[1]
    SB = SB / data.shape[1]
    SW = SW / data.shape[1]

    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:k]

    # If we want, we can find a basis U for the subspace spanned by W using the singular value decomposition of W
    UW, s, _ = np.linalg.svd(W)
    U = UW[:, 0:k]

    P = W[:, 0:k]
    LDAedData = np.dot(P.T, test_data)

    return LDAedData



def gaussianizer(data):
    """Return a Gaussianized version of the Data"""
    gaussianized_datas = []

    print("Data Gaussianization")

    for sample in range(data.shape[1]):
        # data[:, sample] column of i-th sample
        s = data[:, sample:sample + 1]
        x = (data < s).sum(axis=1) + 1
        gaussianized_datas.append(x)


    gaussianized_datas = np.array(gaussianized_datas).T / (data.shape[1] + 2)
    gaussianized_datas = norm.ppf(gaussianized_datas)

    print("Data correctly Gaussianizated")

    return gaussianized_datas


def preprocess_gaussianization(DTR: np.ndarray, DTE: np.ndarray):
    gauss_DTR = np.zeros(DTR.shape)
    for f in range(DTR.shape[0]):
        gauss_DTR[f, :] = scipy.stats.norm.ppf(scipy.stats.rankdata(DTR[f, :], method="min") / (DTR.shape[1] + 2))

    gauss_DTE = np.zeros(DTE.shape)
    for f in range(DTR.shape[0]):
        for idx, x in enumerate(DTE[f, :]):
            rank = 0
            for x_i in DTR[f, :]:
                if (x_i < x):
                    rank += 1
            rank = (rank + 1) / (DTR.shape[1] + 2)
            gauss_DTE[f][idx] = scipy.stats.norm.ppf(rank)
    return gauss_DTR, gauss_DTE


def gaussianizer_test_data(data, test_data):
    """Return a Gaussianized version of the test data"""
    gaussianized_datas = []

    print("Data Gaussianization")

    for sample in range(test_data.shape[1]):
        # data[:, sample] column of i-th sample
        s = test_data[:, sample:sample + 1]
        x = (data < s).sum(axis=1) + 1
        gaussianized_datas.append(x)


    rank = np.array(gaussianized_datas).T / (test_data.shape[1] + 2)
    gaussianized_datas = norm.ppf(rank)

    print("Data correctly Gaussianizatead")

    return gaussianized_datas





