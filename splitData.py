import numpy as np

#spilits the given data 4 to 1
def split_4to1(matrix, labels, seed=0):
    nTrain = int(matrix.shape[1]*4.0/5.0)
    np.random.seed(seed)
    idx = np.random.permutation(matrix.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    trainingData = matrix[:, idxTrain]
    testData = matrix[:, idxTest]
    trainingLabels = labels[idxTrain]
    testLabels = labels[idxTest]
    return (trainingData, trainingLabels), (testData, testLabels)

#spilits the given data 1 to 1
def split_1to1(matrix, labels, seed=0):
    nTrain = int(matrix.shape[0]*1.0/2.0)
    np.random.seed(seed)
    idx = np.random.permutation(matrix.shape[0])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    trainingData = matrix[idxTrain]
    testData = matrix[idxTest]
    trainingLabels = labels[idxTrain]
    testLabels = labels[idxTest]
    return (trainingData, trainingLabels), (testData, testLabels)

# k fold cross validation with given k counter for eack fold
def kFoldCrossValSplit(matrix, labels, k, counter, seed=0):
    nTest = int(matrix.shape[1] * 1.0 / k)
    np.random.seed(seed)
    idx = np.random.permutation(matrix.shape[1])
    idxTest = idx[counter*nTest:(counter+1)*nTest]
    testData = matrix[:, idxTest]
    idxTrain = [i for i in idx if i not in idxTest]

    trainingData = matrix[:, idxTrain]
    trainingLabels = labels[idxTrain]
    testLabels = labels[idxTest]
    return (trainingData, trainingLabels), (testData, testLabels)
