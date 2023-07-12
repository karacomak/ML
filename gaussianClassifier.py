import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.linalg
from mpl_toolkits import mplot3d
from data_preparation_PCA_LDA import *
from evaluation_funcions import confusionMatrix
from splitData import *
from plottingFunctions import *
import argparse

##### DEFAULT PARSER SETTINGS #######################
data_file = 'projects/Gender_Detection/Train.txt'
test_file = 'projects/Gender_Detection/Test.txt'
k_fold = None

dimension_reduction_caption = 'specify reduction technique PCA or LDA and the dimension that required e.g. --dimension_reduction PCA-4'
save = False
evaluation_or_validation = 'validations'
priors = [0.5, 0.5]
validation_data_size = [1200, 4000]
######################################################

#for given matrix returns likelihood in this case log likelihood
def logpdf_GAU_ND(matrix, C, mu):
    Y = []
    P = np.linalg.pinv(C)
    const = -0.5 * np.linalg.slogdet(C)[1]
    const -= 0.5 * matrix.shape[0] * np.log(2 * np.pi)
    for i in range(matrix.shape[1]):
        xc = matrix[:, i:i + 1] - mu
        res = const - 0.5 * np.dot(xc.T, np.dot(P, xc))
        Y.append(res)
    return np.array(Y).ravel()

# for specific covariance matrix application sets the covariance matrix and means and calls log likelihood function
def gaussianClassifier(trainingData, trainingLabels, validationData, tied=False, naive=False, save=False):
    # result matrixes
    logSjoint = np.zeros((np.unique(trainingLabels).size, validationData.shape[1]))
    likelihood = np.zeros((np.unique(trainingLabels).size, validationData.shape[1]))

    #loop for each class
    for i in np.unique(trainingLabels):
        #class separetion
        currentClassTrainingData = trainingData[:, trainingLabels == i]

        #mean of each feature and ful covariance matrix
        cMatrix, meansOfFeatures = covarianceMatrix(currentClassTrainingData)

        #class prior
        classPrior = priors[int(i)]

        # for naive tied covariance matrix
        if (tied == True) & (naive == True):
            #covariance matrix for all traing data
            tiedC = tiedCovariance(trainingData, trainingLabels)
            #diagonal version of covariance
            naiveTiedC = extractDiagonal(tiedC)
            likelihood[int(i), :] = logpdf_GAU_ND(validationData, naiveTiedC, meansOfFeatures).ravel()
            logSjoint[int(i), :] = logpdf_GAU_ND(validationData, naiveTiedC, meansOfFeatures).ravel() + np.log(
                classPrior)
            caption = 'Naive Tied Gaussian Classifier'

        #tied covariance matrix
        if (tied == True) & (naive == False):
            tiedC = tiedCovariance(trainingData, trainingLabels)
            likelihood[int(i), :] = logpdf_GAU_ND(validationData, tiedC, meansOfFeatures).ravel()
            logSjoint[int(i), :] = logpdf_GAU_ND(validationData, tiedC, meansOfFeatures).ravel() + np.log(classPrior)
            caption = 'Tied Gaussian Classifier'

        #full covariance matrix
        if (tied == False) & (naive == False):
            likelihood[int(i), :] = logpdf_GAU_ND(validationData, cMatrix, meansOfFeatures).ravel()
            logSjoint[int(i), :] = logpdf_GAU_ND(validationData, cMatrix, meansOfFeatures).ravel() + np.log(classPrior)
            caption = 'Gaussian Classifier'

        #naive covariance matrix
        if (tied == False) & (naive == True):
            naiveC = extractDiagonal(cMatrix)
            likelihood[int(i), :] = logpdf_GAU_ND(validationData, naiveC, meansOfFeatures).ravel()
            logSjoint[int(i), :] = logpdf_GAU_ND(validationData, naiveC, meansOfFeatures).ravel() + np.log(classPrior)
            caption = 'Naive Gaussian Classifier'

    #log joint likelihoods
    logSjoint = np.exp(logSjoint)
    predictions = logSjoint.argmax(0)

    #likelihood ratio for cost detection
    binary_likelihood_ratio = likelihood[1,:] - likelihood[0,:]

    return predictions, caption, logSjoint, binary_likelihood_ratio

# calculate the tied covariance matrix
def tiedCovariance(matrix, labels):
    cstar = 0
    for i in range(np.unique(labels).size):
        currentMatrix = matrix[:, labels == i]
        _, meansOfClassFeatures = covarianceMatrix(matrix)
        cstar += np.dot((currentMatrix - meansOfClassFeatures), (currentMatrix - meansOfClassFeatures).T)
    return cstar / matrix.shape[1]

# extracts the diaonal matix
def extractDiagonal(matrix):
    IMatrix = np.identity(matrix.shape[0])
    return IMatrix * matrix


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--data_file", type=str, default=data_file, help="test or training data file (default "
                                                                         "projects/Wine_Quality_Detection/Train-Test.txt)", required=False)
    PARSER.add_argument("--test", type=bool, default=False, help="if this is test type something", required=False)
    PARSER.add_argument("--dimension_reduction", type=str, default=None, help=dimension_reduction_caption, required=False)
    PARSER.add_argument("--naive", type=bool, default=False, help='if you want naive gausian classifier type 1 e.g. --naive 1', required=False)
    PARSER.add_argument("--tied", type=bool, default=False, help='if you want tied gausian classifier type 1 e.g. --naive 1', required=False)
    PARSER.add_argument("--save", type=bool, default=False, help='type 1 if want to save llr', required=False)
    PARSER.add_argument("--DR_exp", type=bool, default=False, help='dimesion reduction experiment for all possible dimensions', required=False)
    PARSER.add_argument("--gaussianizer", type=bool, default=False, help='if you want to use Gaussian cumulative distribution, type 1', required=False)
    ARGS, UNKNOWN = PARSER.parse_known_args()

    # Data loading
    if ARGS.data_file is not None:
        data_file = ARGS.data_file
    data, labels = loadData(data_file)

    if ARGS.test == True:
        evaluation_or_validation = 'evaluation'

    # Gaussian cumulative distribution function and z-score
    if (ARGS.gaussianizer is True) & (ARGS.test is False):
        data = gaussianizer(data)
        data = z_score(data)

    # Dimension reduction
    if (ARGS.dimension_reduction is not None) & (ARGS.test is False):
        save =True
        reduction_type = ARGS.dimension_reduction.split('-')[0]
        required_dimension = int(ARGS.dimension_reduction.split('-')[1])
        if reduction_type == 'PCA':
            data = PCA(data, required_dimension)
            print(f"PCA applied and dimension has been decreased to {required_dimension}")
        else:
            data = LDA(data, labels, required_dimension)
            print(f"LDA applied and dimension has been decreased to {required_dimension}")


    # Training
    if ARGS.test is False:
        # Spritting data to train and validation
        (train_data, train_labels), (validation_data, validation_labels) = split_4to1(data, labels)

        # accuracy of model and likelihoods
        predictions, caption, logSjoint, llr = gaussianClassifier(train_data, train_labels, validation_data, tied=ARGS.tied, naive=ARGS.naive, save=ARGS.save)
        accuracy = (predictions == validation_labels).sum() / predictions.size
        err = (1 - accuracy) * 100
        print(f"{caption} Error rate: {err} on validation data")

        # Saving likelihoods to future experiment
        if (ARGS.save == True):
            np.save(
                f'projects/gender_experiments/{evaluation_or_validation}/gaussian_classifier_4to1_likelihoods/naive_{ARGS.naive}_tied_{ARGS.tied}_gaussianizer_{ARGS.gaussianizer}_DR_{ARGS.dimension_reduction}_llr.npy',
                llr)
            np.save(
                f'projects/gender_experiments/{evaluation_or_validation}/gaussian_classifier_4to1_likelihoods/labels.npy',
                validation_labels)



    # Test
    else:
        # Test data loading
        test_data, test_labels = loadData(test_file)

        # Dimension reduction for test data
        if ARGS.dimension_reduction != None:
            reduction_type = ARGS.dimension_reduction.split("-")[0]
            required_dimension = int(ARGS.dimension_reduction.split("-")[1])
            if reduction_type == 'PCA':
                test_data = PCA_test(data, test_data, required_dimension)
                data = PCA(data, required_dimension)
                print(f"PCA applied and dimension has been decreased to {required_dimension}")
            else:
                test_data = LDA_test(data, labels, test_data, required_dimension)
                data = LDA(data, labels, required_dimension)
                print(f"LDA applied and dimension has been decreased to {required_dimension}")

        # Gaussian cumulative distribution function and z-score
        if ARGS.gaussianizer is True:
            test_data = z_score_test(data, test_data)
            data = z_score(data)
            data, test_data = preprocess_gaussianization(data, test_data)

        # accuracy of model and likelihoods
        predictions, caption, logSjoint, llr = gaussianClassifier(data, labels, test_data, tied=ARGS.tied, naive=ARGS.naive)
        accuracy = (predictions == test_labels).sum() / predictions.size
        accuracy = (1 - accuracy) * 100
        print(f"{caption} accuracy: {accuracy} on test data")


        # Saving likelihoods to future experiment
        if (ARGS.save == True):
            np.save(
                f'projects/gender_experiments/{evaluation_or_validation}/gaussian_classifier_4to1_likelihoods/naive_{ARGS.naive}_tied_{ARGS.tied}_gaussianizer_{ARGS.gaussianizer}_DR_{ARGS.dimension_reduction}_llr.npy',
                llr)
            np.save(
                f'projects/gender_experiments/{evaluation_or_validation}/gaussian_classifier_4to1_likelihoods/labels.npy',
                test_labels)

    # dimension reduction analysis
    if ARGS.DR_exp == True:
        llr_array = np.zeros((data.shape[0]-1, validation_data_size[0]))
        for i in range(data.shape[0]-1):
            new_data = LDA(data, labels, i+1)
            print(f"LDA applied and dimension has been decreased to {i+1}")

            # Spritting data to train and validation
            (train_data, train_labels), (validation_data, validation_labels) = split_4to1(new_data, labels)

            # accuracy of model and likelihoods
            predictions, caption, logSjoint, llr = gaussianClassifier(train_data, train_labels, validation_data,
                                                                      tied=ARGS.tied, naive=ARGS.naive, save=ARGS.save)
            accuracy = (predictions == validation_labels).sum() / predictions.size
            err = (1 - accuracy) * 100
            print(f"{caption} Error rate: {err} on validation data")

            llr_array[i, :] += llr

        np.save(f'projects/gender_experiments/dimension_experiment/MVG/MVG_naive_{ARGS.naive}_tied_{ARGS.tied}_gaussianizer_{ARGS.gaussianizer}_llr.npy', llr_array)
        np.save(f'projects/gender_experiments/dimension_experiment/MVG/labels.npy', validation_labels)