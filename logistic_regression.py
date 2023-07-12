import numpy as np
from evaluation_funcions import *
from data_preparation_PCA_LDA import *
from splitData import *
from plottingFunctions import *
import scipy.optimize
import argparse

##### DEFAULT PARSER SETTINGS #######################
data_file = 'projects/Gender_Detection/Train.txt'
test_file = 'projects/Gender_Detection/Test.txt'
k_fold = None
my_lambda = '0.0001'
lambdas_for_k_fold = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5]
setOfPriorAndCostsToExperiment = np.array([[0.5, 1, 1], [0.1, 1, 1], [0.9, 1, 1]])

learning_rate_caption = 'specify the learning rate e.g. 0.1 default 0.01.'
dimension_reduction_caption = 'specify reduction technique PCA or LDA and the dimension that required e.g. --dimension_reduction PCA-4'
k_fold_caption = "k for k-fold cross validation specify k that is required for best learning rate. Please set learning rates e.g. --learning_rate 0.1-0.01... if you want to specify (however, there is default array)"

validation_data_size = [1200, 4000]
######################################################

#logistic regression wrap function
def logreg_obj_wrap_rebalance(DTR, LTR, l, prior):
    Z = LTR * 2 - 1
    M = DTR.shape[0]

    nf = len(LTR[LTR == 0])
    nt = len(LTR[LTR == 1])

    def logreg_obj(v):
        w = xToOne(v[0:M])
        b = v[-1]
        cxef = 0
        cxe0 = 0
        S = np.dot(w.T, DTR) + b
        cxef = np.logaddexp(0, -S * Z).sum() * (1-prior) * (1/nf) /2
        cxep = np.logaddexp(0, -S * Z).sum() * prior * (1/nt) / 2
        return l / 2.0 * np.linalg.norm(w) ** 2 + cxef + cxep

    return logreg_obj

#logistic regression wrap function
def logreg_obj_wrap(DTR, LTR, l):
    Z = LTR * 2 - 1
    M = DTR.shape[0]

    def logreg_obj(v):
        w = xToOne(v[0:M])
        b = v[-1]
        cxe = 0
        S = np.dot(w.T, DTR) + b
        cxe = np.logaddexp(0, -S * Z).mean()
        return l / 2.0 * np.linalg.norm(w) ** 2 + cxe

    return logreg_obj


def l_bfgs_optimizer_multi_lr(trainData, trainLabels, given_lambdas, prior=None):
    w_star_array, b_array = [], []
    starting_points = np.zeros(trainData.shape[0] + 1)

    for lamb in given_lambdas:
        if prior is None:
            logreg_obj = logreg_obj_wrap(trainData, trainLabels, lamb)
        else:
            logreg_obj = logreg_obj_wrap_rebalance(trainData, trainLabels, lamb, prior)
        w_star_b, J, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, starting_points, approx_grad=True, iprint=-1)
        w_star = w_star_b[0:-1]
        b = w_star_b[-1]
        w_star_array.append(w_star)
        b_array.append(b)
    return np.array(w_star_array), np.array(b_array)


def l_bfgs_optimizer(trainData, trainLabels, given_lambda, prior=None):
    starting_points = np.zeros(trainData.shape[0] + 1)
    if prior is None:
        logreg_obj = logreg_obj_wrap(trainData, trainLabels, given_lambda)
    else:
        logreg_obj = logreg_obj_wrap_rebalance(trainData, trainLabels, given_lambda, prior)
    w_star_b, J, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, starting_points, approx_grad=True, iprint=-1)
    w_star = w_star_b[0:-1]
    b = w_star_b[-1]
    return w_star, b

def transformation_for_quadratic(data):
    return data**2

def transformation_for_quadratic_reverse(data):
    return data**0.5


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--data_file", type=str, default=data_file, help="test or training data file (default "
                                                                         "projects/Wine_Quality_Detection/Train-Test.txt)",
                        required=False)
    PARSER.add_argument("--k_fold", type=int, default=k_fold,
                        help="k for k-fold cross validation specify k that is required for best learning rate",
                        required=False)
    PARSER.add_argument("--test", type=bool, default=False, help="if this is test type something", required=False)
    PARSER.add_argument("--dimension_reduction", type=str, default=None, help=dimension_reduction_caption,
                        required=False)
    PARSER.add_argument("--my_lambda", type=str, default=my_lambda, help=learning_rate_caption, required=False)
    PARSER.add_argument("--confusion_matrix", type=bool, default=False, help='type 1 if confusion matrix is needed',
                        required=False)
    PARSER.add_argument("--save", type=bool, default=False, help='type 1 if want to save scores', required=False)
    PARSER.add_argument("--gaussianizer", type=bool, default=False,
                        help='if you want to use Gaussian cumulative distribution, type 1', required=False)
    PARSER.add_argument("--quadratic", type=str, default=None, help='type 1 if want to quadratic hyperplane', required=False)
    PARSER.add_argument("--DR_exp", type=bool, default=False,
                        help='dimesion reduction experiment for all possible dimensions', required=False)
    ARGS, UNKNOWN = PARSER.parse_known_args()

    # Data loading
    if ARGS.data_file is not None:
        data_file = ARGS.data_file
    data, labels = loadData(data_file)

    # Gaussian cumulative distribution function and z-score
    if (ARGS.gaussianizer is True) & (ARGS.test is False):
        data = z_score(data)
        data = gaussianizer(data)

    # Learning rate
    if ARGS.my_lambda != my_lambda:
        my_lambda = float(ARGS.my_lambda)
        lambdas_for_k_fold = ARGS.my_lambda.split("-")

    # Dimension reduction
    if (ARGS.dimension_reduction is not None) & (ARGS.test is False):
        reduction_type = ARGS.dimension_reduction.split('-')[0]
        required_dimension = int(ARGS.dimension_reduction.split('-')[1])
        if reduction_type == 'PCA':
            data = PCA(data, required_dimension)
            print(f"PCA applied and dimension has been decreased to {required_dimension}")
        else:
            data = LDA(data, labels, required_dimension)
            print(f"LDA applied and dimension has been decreased to {required_dimension}")

    ''' k-fold cross validation for parameter selection in this case is learning rate '''
    if (ARGS.test is False) | (ARGS.k_fold is not None):

        ''' preparation of the loops and result arrays '''
        if ARGS.k_fold is not None:
            lambdas = np.zeros((len(lambdas_for_k_fold)))
            minDCFs = np.zeros((3, len(lambdas_for_k_fold)))
            for j, current_lambda in enumerate(lambdas_for_k_fold):
                # lists to keep scores with their labels
                evaluate_or_validate = 1 if ARGS.test else 0
                set_of_scores = np.zeros((ARGS.k_fold, validation_data_size[evaluate_or_validate]))
                set_of_validation_labels = np.zeros((ARGS.k_fold, validation_data_size[evaluate_or_validate]))

                ''' loop for each fold '''
                for i in range(ARGS.k_fold):
                    if ARGS.k_fold == 1:
                        (train_data, train_labels), (validation_data, validation_labels) = split_4to1(data, labels)
                    else:
                        (train_data, train_labels), (validation_data, validation_labels) = kFoldCrossValSplit(data,
                                                                                                              labels,
                                                                                                              ARGS.k_fold,
                                                                                                              i)
                    # if this is test in evaluation test data preparation
                    if ARGS.test == True:
                        # Test data loading
                        test_data, test_labels = loadData(test_file)
                        # Gaussian cumulative distribution function and z-score
                        if ARGS.gaussianizer is True:
                            test_data = z_score_test(data, test_data)
                            data = z_score(data)
                            data, test_data = preprocess_gaussianization(data, test_data)

                        # *** on the test we are using 100% of train set and test set ***
                        # *** And I runned it with k_fold 1 which means single fold ***
                        train_data = data
                        train_labels = labels
                        validation_data = test_data
                        validation_labels = test_labels

                    # if we want quadratic hyperplane transformation on data
                    if (ARGS.quadratic == True):
                        train_data = transformation_for_quadratic(train_data)
                        validation_data = transformation_for_quadratic(validation_data)

                    w_star, b = l_bfgs_optimizer(train_data, train_labels, float(current_lambda))
                    S = np.dot(w_star.T, validation_data) + b
                    predictions = (S > 0)
                    error_rate = 1 - (predictions == validation_labels).sum() / np.array(predictions).size
                    error_rate = error_rate * 100

                    # keep the scores than calculate min DCF for all
                    set_of_scores[i,:] = S
                    set_of_validation_labels[i,:] = validation_labels

                # min DCF calculator for balanced and un balanced applicaitons
                for i, subset in enumerate(setOfPriorAndCostsToExperiment):
                    priorProb, Cfn, Cfp = subset
                    minDCF = minDCFBinary(set_of_scores.ravel(), set_of_validation_labels.ravel(), priorProb, Cfn, Cfp)

                    if i == 0:
                        minDCF_for_05 = minDCF
                    if i == 1:
                        minDCF_for_01 = minDCF
                    if i == 2:
                        minDCF_for_09 = minDCF

                lambdas[j:j+1] = current_lambda
                minDCFs[0, j:j+1] = minDCF_for_05
                minDCFs[1, j:j + 1] = minDCF_for_01
                minDCFs[2, j:j + 1] = minDCF_for_09

            # Saving for experiments
            if (ARGS.save == True) & (ARGS.test == False):
                np.save(
                    f'projects/gender_experiments/validations/LR_classifier_k-fold_minDCFs_for_best_lambda/quadratic_{ARGS.quadratic}/gaussianizer_{ARGS.gaussianizer}_minDCFs.npy',
                    minDCFs)
                np.save(
                    f'projects/gender_experiments/validations/LR_classifier_k-fold_minDCFs_for_best_lambda/quadratic_{ARGS.quadratic}/lambdas.npy',
                    lambdas_for_k_fold)
            if (ARGS.save == True) & (ARGS.test == True):
                np.save(
                    f'projects/gender_experiments/evaluation/LR_classifier_k-fold_minDCFs_for_best_lambda/quadratic_{ARGS.quadratic}/gaussianizer_{ARGS.gaussianizer}_minDCFs.npy',
                    minDCFs)
                np.save(
                    f'projects/gender_experiments/evaluation/LR_classifier_k-fold_minDCFs_for_best_lambda/quadratic_{ARGS.quadratic}/lambdas.npy',
                    lambdas_for_k_fold)

        #training for specific parameters
        else:
            (train_data, train_labels), (validation_data, validation_labels) = split_4to1(data, labels)

            # if we want quadratic hyperplane transformation on data
            if (ARGS.quadratic == True):
                train_data = transformation_for_quadratic(train_data)
                validation_data = transformation_for_quadratic(validation_data)

            # Ws and b for given train data
            w_star, b = l_bfgs_optimizer(train_data, train_labels, float(my_lambda))
            S = np.dot(w_star.T, validation_data) + b
            predictions = (S > 0)
            error_rate = 1 - (predictions == validation_labels).sum() / np.array(predictions).size
            error_rate = error_rate * 100
            print(f"Error rate: {error_rate} with given learning rate: {ARGS.my_lambda} on validation data")


            # Saving for experiments
            if (ARGS.save == True):
                np.save(f'projects/gender_experiments/validations/LR_classifier_4to1_scores/quadratic_{ARGS.quadratic}/lr_{my_lambda}_gaussianizer_{ARGS.gaussianizer}_DR_{ARGS.dimension_reduction}_scores.npy', S)
                # only for one time to be keep right labels
                np.save(f'projects/gender_experiments/validations/LR_classifier_4to1_scores/quadratic_{ARGS.quadratic}/labels', validation_labels)
    else:
        # Test data loaing
        test_data, test_labels = loadData(test_file)
        # Dimension reduction for test data
        if ARGS.dimension_reduction is not None:
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

        # if we want quadratic hyperplane also we need to optimize c for better results
        if ARGS.quadratic == True:
            data = transformation_for_quadratic(data)
            test_data = transformation_for_quadratic(test_data)

        # Ws and b for given train data
        w_star, b = l_bfgs_optimizer(data, labels, float(my_lambda))
        S = np.dot(w_star.T, test_data) + b
        predictions = (S > 0)
        error_rate = 1 - (predictions == test_labels).sum() / np.array(predictions).size
        error_rate = error_rate * 100
        print(f"Error rate: {error_rate} with given learning rate: {ARGS.my_lambda} on test data")

        # Saving for experiments
        if (ARGS.save == True):
            np.save(
                f'projects/gender_experiments/evaluation/LR_classifier_4to1_scores/quadratic_{ARGS.quadratic}/lr_{my_lambda}_gaussianizer_{ARGS.gaussianizer}_DR_{ARGS.dimension_reduction}_scores.npy',
                S)
            # only for one time to be keep right labels
            np.save(f'projects/gender_experiments/evaluation/LR_classifier_4to1_scores/quadratic_{ARGS.quadratic}/labels.npy',
                    test_labels)

    # dimension reduction analysis
    if ARGS.DR_exp == True:
        score_array = np.zeros((data.shape[0]-1, validation_data_size[0]))
        for i in range(data.shape[0]-1):
            new_data = PCA(data, i+1)
            print(f"LDA applied and dimension has been decreased to {i+1}")

            # Spritting data to train and validation
            (train_data, train_labels), (validation_data, validation_labels) = split_4to1(new_data, labels)

            # Ws and b for given train data
            w_star, b = l_bfgs_optimizer(train_data, train_labels, float(my_lambda))
            S = np.dot(w_star.T, validation_data) + b
            predictions = (S > 0)
            error_rate = 1 - (predictions == validation_labels).sum() / np.array(predictions).size
            error_rate = error_rate * 100
            print(f"Error rate: {error_rate} with given learning rate: {ARGS.my_lambda} on validation data")

            score_array[i, :] += S

        np.save(f'projects/gender_experiments/dimension_experiment/LR/scores.npy', score_array)
        np.save(f'projects/gender_experiments/dimension_experiment/LR/labels.npy', validation_labels)