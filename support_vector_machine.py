import numpy as np
import scipy.optimize
import sklearn.datasets
from evaluation_funcions import *
from data_preparation_PCA_LDA import *
from splitData import *
from plottingFunctions import *
import argparse
import json

##### DEFAULT PARSER SETTINGS #######################
data_file = 'projects/Gender_Detection/Train.txt'
test_file = 'projects/Gender_Detection/Test.txt'
C = 10
K = 0.1
D = 2
c_of_kernel = 1
pi_true = 0.5
default_gamma = 0.01

set_of_C = [0.01, 0.1, 1, 10, 100]
set_of_K_or_c_or_gamma = [1e-2, 1e-1, 1, 10, 100]

k_fold = None

setOfPriorAndCostsToExperiment = np.array([[0.5, 1, 1], [0.1, 1, 1], [0.9, 1, 1]])

dimension_reduction_caption = 'specify reduction technique PCA or LDA and the dimension that required e.g. --dimension_reduction PCA 4'

validation_data_size = [1200, 4000]
evaluation_or_validation = 'validations'
######################################################
# given w calculates the prediction and scores for linear SVM
def predict_for_linear(test_data, w, K=K):
    e_test_data = extend_matrix(test_data, K=K)
    prediction = (np.dot(w.T, e_test_data) > 0)
    scores = np.dot(w.T, e_test_data).reshape((prediction.shape[1],))
    return prediction, scores

# extend each sample with given K with is our b in linear SVM
def extend_matrix(matrix, K=K):
    extended_matrix = np.zeros((matrix.shape[0] + 1, matrix.shape[1])) + K
    extended_matrix[0:matrix.shape[0], :] = matrix
    return extended_matrix

#dualization
def extendedH(z, extended_x):
    G_ij = np.dot(extended_x.T, extended_x)
    extended_H = oneToX(z) * xToOne(z) * G_ij
    return extended_H

#(min, max) pairs for each element in training data set our classification constraints cruel for classification
def bounds(C, training_data):
    bounds_list = []
    for i in range(training_data.shape[1]):
        bounds_list.append((0, C))
    return bounds_list

#(min, max) pairs for each element in training data set our classification constraints cruel for classification for re-balanced applications
def bounds_re_balance(C, training_label, prior):
    nf = training_label[training_label == 0].size
    nt = training_label[training_label == 1].size
    pi_emp_f = nf / training_label.size
    pi_emp_t = nt / training_label.size

    Cf = C * (1-prior) * (1/pi_emp_f)
    Ct = C * prior * (1/pi_emp_t)

    bounds_list = []
    for i in training_label:
        if i == 0:
            bounds_list.append((0, Cf))
        else:
            bounds_list.append((0, Ct))
    return bounds_list, Ct, Cf


def duality_gap(J, J_D):
    return J - J_D

# reverse version of dualization
def recover_dual_to_primal(alpha, Z, extended_x_i):
    extended_w_star = np.dot(extended_x_i, xToOne(alpha) * xToOne(Z))
    return extended_w_star

#training wrap function recall itself until stop condition
def train_SVM_linear(train_data, train_labels, C=C, K=K, re_balance=None):

    extended_x_i = extend_matrix(train_data, K=K)
    Z = train_labels * 2 - 1

    extended_H = extendedH(Z, extended_x_i)

    def J(extended_w_star):
        return (0.5 * np.linalg.norm(extended_w_star) ** 2) + (
                C * np.maximum(0, (1 - Z * (np.dot(extended_w_star.T, extended_x_i)))).sum())

    def J_D(alpha):
        Ha = np.dot(extended_H, xToOne(alpha))
        aHa = np.dot(oneToX(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + np.ones(alpha.size)

    def L_D(alpha):
        loss, grad = J_D(alpha)
        return -loss, -grad

    if re_balance is None:
        b = [(0, C)] * train_data.shape[1]
    else:
        b, Ct, Cf = bounds_re_balance(C, train_labels, re_balance)
        print(f'the classes are re-balanced with given prior {re_balance} and Ct:{Ct} , Cf:{Cf}')

    alpha_star, f, d = scipy.optimize.fmin_l_bfgs_b(
        L_D,
        np.zeros(train_data.shape[1]),
        bounds=b,
        factr=1.0,
        maxiter=100000,
        maxfun=100000
    )

    w_star = recover_dual_to_primal(alpha_star, Z, extended_x_i)

    primal_loss = J(w_star)
    dual_loss = J_D(alpha_star)[0]

    print(f"Primal loss: {primal_loss}")
    print(f"Dual loss: {dual_loss}")
    print(f"Duality gap: {duality_gap(primal_loss, dual_loss)}")

    return w_star


########################################################
# Non linear SVM with kernel
#
#########################################################
# dualization with polynomial kernel
def extendedH_with_polynomial_kernel(z, x, c, d=2, k=0):
    kernel = (np.dot(x.T, x) + c) ** d
    extended_H = oneToX(z) * xToOne(z) * (kernel + k)
    return extended_H

# dualization with radial basis function
def extendedH_with_RBF_kernel(z, x, k=0, gamma=default_gamma):
    dist = oneToX((x ** 2).sum(0)) + xToOne((x ** 2).sum(0)) - 2 * np.dot(x.T, x)
    kernel = np.exp( -gamma *dist)
    extended_H = oneToX(z) * xToOne(z) * (kernel + k)
    return extended_H

# prediction makes with given data for specific kernel
def predict_non_linear(test_data, train_data, trainLabel, alpha_star, c=1, d=2, k=0, gamma=default_gamma, final=False):
    Z = trainLabel * 2 - 1
    if final:
        kernel = RBF_kernel(test_data, train_data, gamma=gamma)
    else:
        if ARGS.quadratic == 'Poly':
            kernel = polynomial_kernel(test_data, train_data, c, d, k)
        if ARGS.quadratic == 'RBF':
            kernel = RBF_kernel(test_data, train_data, gamma=gamma)
    prediction = (np.dot(kernel, xToOne(alpha_star) * xToOne(Z)) > 0)
    scores = np.dot(kernel, xToOne(alpha_star) * xToOne(Z)).reshape((prediction.shape[0],))
    return prediction, scores

# polynomial kernel for prediction
def polynomial_kernel(test_data, train_data, c, d, k):
    kernel = (np.dot(test_data.T, train_data) + c) ** d + k
    return kernel

# RVF kernel for prediction
def RBF_kernel(test_data, train_data, k=0, gamma=default_gamma):
    dist = oneToX((train_data ** 2).sum(0)) + xToOne((test_data ** 2).sum(0)) - 2 * np.dot(test_data.T, train_data)
    kernel = np.exp( -gamma * dist) + k
    return kernel

# non-linear wrap function
def train_SVM_nonlinear(training_data, training_labels, C, c=0, d=2, k=0, gamma=default_gamma, re_balance=None, final=False):
    print(f'C:{C}, c:{c}, d:{d}, k:{k}, gamma: {gamma}')
    Z = training_labels * 2 - 1

    if final:
        extended_H = extendedH_with_RBF_kernel(Z, training_data, gamma=gamma, k=k)
    else:
        if ARGS.quadratic == 'Poly':
            extended_H = extendedH_with_polynomial_kernel(Z, training_data, c=c, d=d, k=k)
        else:
            extended_H = extendedH_with_RBF_kernel(Z, training_data, gamma=gamma, k=k)


    def J_D(alpha):
        Ha = np.dot(extended_H, xToOne(alpha))
        aHa = np.dot(oneToX(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + np.ones(alpha.size)

    def L_D(alpha):
        loss, grad = J_D(alpha)
        return -loss, -grad

    if re_balance is None:
        b = [(0, C)] * training_data.shape[1]
    else:
        b, Ct, Cf = bounds_re_balance(C, training_labels, re_balance)
        print(f'the classes are re-balanced with given prior {re_balance} and Ct:{Ct} , Cf:{Cf}')

    alpha_star, f, d = scipy.optimize.fmin_l_bfgs_b(
        L_D,
        np.zeros(training_data.shape[1]),
        bounds=b,
        factr=1.0,
        maxiter=100000,
        maxfun=100000
    )

    dual_loss = J_D(alpha_star)[0]
    print(dual_loss)
    return alpha_star


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--data_file", type=str, default=data_file, help="test or training data file (default "
                                                                         "projects/Wine_Quality_Detection/Train.txt)",
                        required=False)
    PARSER.add_argument("--C", type=float, default=C, help="C of SVM (default 0.01)", required=False)
    PARSER.add_argument("--K", type=float, default=K, help="bias of SVM (default 10)", required=False)
    PARSER.add_argument("--c_of_kernel", type=float, default=c_of_kernel, help="c of polynominal kernel (default 1)",
                        required=False)
    PARSER.add_argument("--D", type=float, default=D, help="d of polynominal kernel (default 2)", required=False)
    PARSER.add_argument("--pi_true", type=float, default=pi_true, help="prior of true class (default 0.5)", required=False)
    PARSER.add_argument("--gamma", type=float, default=default_gamma, help="gamma of RBF kernel (default 100)", required=False)
    PARSER.add_argument("--k_fold", type=int, default=k_fold, help="k for k-fold cross validation to find best C and K",
                        required=False)
    PARSER.add_argument("--test", type=bool, default=False, help="if this is test type something", required=False)
    PARSER.add_argument("--dimension_reduction", type=str, default=None, help=dimension_reduction_caption,
                        required=False)
    PARSER.add_argument("--confusion_matrix", type=bool, default=False, help='type 1 if confusion matrix is needed',
                        required=False)
    PARSER.add_argument("--gaussianizer", type=bool, default=False,
                        help='if you want to use Gaussian cumulative distribution, type 1', required=False)
    PARSER.add_argument("--save", type=bool, default=False, help='type 1 if want to save scores', required=False)
    PARSER.add_argument("--quadratic", type=str, default=None, help='specify the kernel type if want to quadratic hyperplane RBF or Poly',
                        required=False)
    PARSER.add_argument("--class_balance", type=bool, default=False,
                        help='type 1 if want to make classes balanced with costs', required=False)
    PARSER.add_argument("--re_balance", type=float, default=None,
                        help='enter prior of true class if want to make  re-balanced application with C', required=False)
    PARSER.add_argument("--DR_exp", type=bool, default=False,
                        help='dimension reduction experiment for all possible dimensions', required=False)
    ARGS, UNKNOWN = PARSER.parse_known_args()

    if ARGS.data_file is not None:
        data_file = ARGS.data_file
    if ARGS.C is not None:
        C = ARGS.C
    if ARGS.K is not None:
        K = ARGS.K
    if ARGS.gamma is not None:
        default_gamma = ARGS.gamma
    if ARGS.c_of_kernel is not None:
        c_of_kernel = ARGS.c_of_kernel
    if ARGS.D is not None:
        D = ARGS.D
    if ARGS.pi_true is not None:
        pi_true = ARGS.pi_true
    if ARGS.test is True:
        evaluation_or_validation = 'evaluation'


    # Data loading
    data, labels = loadData(data_file)
    PIempT = labels[labels == 1].size / labels.size
    PIempF = labels[labels == 0].size / labels.size

    # Dimension reduction
    if (ARGS.dimension_reduction is not None) & (ARGS.test is False):
        save = True
        reduction_type = ARGS.dimension_reduction.split('-')[0]
        required_dimension = int(ARGS.dimension_reduction.split('-')[1])
        if reduction_type == 'PCA':
            data = PCA(data, required_dimension)
            print(f"PCA applied and dimension has been decreased to {required_dimension}")
        else:
            data = LDA(data, labels, required_dimension)
            print(f"LDA applied and dimension has been decreased to {required_dimension}")

    # Gaussian cumulative distribution function and z-score
    if (ARGS.gaussianizer is True) & (ARGS.test is False):
        data = z_score(data)
        data = gaussianizer(data)

    if (ARGS.test is True) & (ARGS.k_fold is not None):
        test_data, test_labels = loadData(test_file)
        if (ARGS.gaussianizer is True) :
            test_data = z_score_test(data, test_data)
            data = z_score(data)
            data, test_data = preprocess_gaussianization(data, test_data)

    # Grid search and k-fold cross validation to find the best general K and C or c for polynominal kernel
    if ARGS.k_fold is not None:
        # Grid search values
        possible_C = set_of_C
        possible_K_or_C_quadratic = set_of_K_or_c_or_gamma
        best_C = 0
        best_K = 0
        best_minDCF = 10
        c_list = []
        k_list = []
        minDCF_array = np.zeros((3, len(possible_C) * len(possible_K_or_C_quadratic)))
        for j, C_loop in enumerate(possible_C):
            for y, K_or_c_or_gamma in enumerate(possible_K_or_C_quadratic):
                # lists to keep scores with their labels
                evaluate_or_validate = 1 if ARGS.test else 0
                set_of_scores = np.zeros((ARGS.k_fold, validation_data_size[evaluate_or_validate]))
                set_of_validation_labels = np.zeros((ARGS.k_fold, validation_data_size[evaluate_or_validate]))
                # K-hold cross validation
                for i in range(ARGS.k_fold):
                    if ARGS.k_fold == 1:
                        (train_data, train_labels), (validation_data, validation_labels) = split_4to1(data, labels)
                    else:
                        (train_data, train_labels), (validation_data, validation_labels) = kFoldCrossValSplit(data, labels,
                                                                                                              ARGS.k_fold,
                                                                                                              i)

                    if ARGS.test == True:
                        # *** on the test we are using 100% of train set and test set ***
                        # *** And I runned it with k_fold 1 which means single fold ***
                        train_data = data
                        train_labels = labels
                        validation_labels = test_labels
                        validation_data = test_data

                    # if we want linear hyperplane
                    if (ARGS.quadratic == None):
                        w_star = train_SVM_linear(train_data, train_labels, C=C_loop, K=K_or_c_or_gamma)
                        prediction, scores = predict_for_linear(validation_data, w_star, K=K_or_c_or_gamma)
                        accuracy = (prediction == validation_labels).sum() / validation_labels.size

                        # keep the scores than calculate min DCF for all
                        set_of_scores[i, :] = scores
                        set_of_validation_labels[i, :] = validation_labels

                    # if we want quadratic hyperplane also we need to optimize c for better results
                    else:
                        alpha_star = train_SVM_nonlinear(train_data, train_labels, C_loop, c=K_or_c_or_gamma, d=2, k=K, gamma=K_or_c_or_gamma)
                        prediction, scores = predict_non_linear(validation_data, train_data, train_labels, alpha_star,
                                                                c=K_or_c_or_gamma, d=2, k=K, gamma=K_or_c_or_gamma)
                        accuracy = (prediction.T == validation_labels).sum() / validation_labels.size

                        # keep the scores than calculate min DCF for all
                        set_of_scores[i, :] = scores
                        set_of_validation_labels[i, :] = validation_labels

                    # information message K-fold current accuracy and other C and k and which fold
                    print(
                        f'K-fold cross validation with grid search error rate: {(1 - accuracy) * 100} current C is {C_loop}, K_or_c is {K_or_c_or_gamma} & k-fold is {i + 1}')

                for ii, subset in enumerate(setOfPriorAndCostsToExperiment):
                    priorProb, Cfn, Cfp = subset
                    minDCF = minDCFBinary(set_of_scores.ravel(), set_of_validation_labels.ravel(), priorProb, Cfn, Cfp)
                    if ii == 0:
                        current_minDCF = minDCF
                        minDCF_for_05 = minDCF
                    if ii == 1:
                        minDCF_for_01 = minDCF
                    if ii == 2:
                        minDCF_for_09 = minDCF

                print(f"Current min DCF is {current_minDCF}")
                if current_minDCF < best_minDCF:
                    best_K = K_or_c_or_gamma
                    best_C = C
                    best_minDCF = current_minDCF
                k_list.append(str(K_or_c_or_gamma))
                c_list.append(str(C_loop))
                counter = len(possible_K_or_C_quadratic) * j + y
                minDCF_array[0, counter:counter + 1] = minDCF_for_05
                minDCF_array[1, counter:counter + 1] = minDCF_for_01
                minDCF_array[2, counter:counter + 1] = minDCF_for_09


        # Saving for experiments as a numpy array
        if (ARGS.save == True):
            np.save(
                f'projects/gender_experiments/{evaluation_or_validation}/SVM_classifier_k-fold_minDCFs_for_best_C/quadratic_{ARGS.quadratic}/balanced_{ARGS.class_balance}_gaussianizer_{ARGS.gaussianizer}_minDCFs.npy',
                minDCF_array)
            np.save(
                f'projects/gender_experiments/{evaluation_or_validation}/SVM_classifier_k-fold_minDCFs_for_best_C/quadratic_{ARGS.quadratic}/Cs.npy',
                c_list)
            np.save(
                f'projects/gender_experiments/{evaluation_or_validation}/SVM_classifier_k-fold_minDCFs_for_best_C/quadratic_{ARGS.quadratic}/Ks.npy',
                k_list)

        print(f"Best C is {best_C}, best K is {best_K} with best minDCF: {best_minDCF}")
    # 4 to 1 split training and test
    if (ARGS.k_fold == None) & (ARGS.test == False):
        # Training with selected C and K
        (train_data, train_labels), (validation_data, validation_labels) = split_4to1(data, labels)

        # if we want linear hyperplane
        if (ARGS.quadratic == None):
            w_star = train_SVM_linear(train_data, train_labels, C=C, K=K, re_balance=ARGS.re_balance)
            predictions, scores = predict_for_linear(validation_data, w_star, K=K)
            accuracy = (predictions == validation_labels).sum() / validation_labels.size
            print(f'Validation error rate: {(1 - accuracy) * 100}')

            # Saving for experiments as a numpy array
            if (ARGS.save == True):
                np.save(
                    f'projects/gender_experiments/validations/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/C_{C}_K_{K}_gaussianizer_{ARGS.gaussianizer}_scores.npy',
                    scores)
                # only for one time to be keep right labels
                np.save(
                    f'projects/gender_experiments/validations/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/labels.npy',
                    validation_labels)
                # for balanced class experiment save the empirical PIs
                np.save(
                    f'projects/gender_experiments/validations/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/PIempT.npy',
                    PIempT)
                np.save(
                    f'projects/gender_experiments/validations/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/PIempF.npy',
                    PIempF)

        # if we want quadratic hyperplane also we need to optimize c for better results
        else:
            alpha_star = train_SVM_nonlinear(train_data, train_labels, C, c=c_of_kernel, d=D, k=K, gamma=default_gamma, re_balance=ARGS.re_balance)
            prediction, scores = predict_non_linear(validation_data, train_data, train_labels, alpha_star,
                                                    c=c_of_kernel, d=D, k=K, gamma=default_gamma)
            accuracy = (prediction.T == validation_labels).sum() / validation_labels.size
            print(f'Validation error rate: {(1 - accuracy) * 100}')

            # Saving for experiments as a numpy array
            if (ARGS.save == True):
                np.save(
                    f'projects/gender_experiments/validations/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/C_{C}_c_{c_of_kernel}_d_{D}_K_{K}_gamma_{default_gamma}_gaussianizer_{ARGS.gaussianizer}_scores.npy',
                    scores)
                # only for one time to be keep right labels
                np.save(
                    f'projects/gender_experiments/validations/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/labels.npy',
                    validation_labels)
                # for balanced class experiment save the empirical PIs
                np.save(
                    f'projects/gender_experiments/validations/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/PIempT.npy',
                    PIempT)
                np.save(
                    f'projects/gender_experiments/validations/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/PIempF.npy',
                    PIempF)

    # test
    if ARGS.test is not False:
        # Test data loading
        test_data, test_labels = loadData(test_file)

        # Gaussian cumulative distribution function and z-score
        if ARGS.gaussianizer is True:
            test_data = z_score_test(data, test_data)
            data = z_score(data)
            data, test_data = preprocess_gaussianization(data, test_data)

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

        if ARGS.quadratic == None:
            # Training with full data
            w_star = train_SVM_linear(data, labels, C=C, K=K, re_balance=ARGS.re_balance)

            # Prediction and accuracy for test data
            predictions, scores = predict_for_linear(test_data, w_star, K=K)
            accuracy = (predictions == test_labels).sum() / test_labels.size
            print(f'Test error rate: {(1 - accuracy) * 100}')

            # Saving for experiments as a numpy array
            if (ARGS.save == True):
                np.save(
                    f'projects/gender_experiments/evaluation/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/C_{C}_K_{K}_gaussianizer_{ARGS.gaussianizer}_scores.npy',
                    scores)
                # only for one time to be keep right labels
                np.save(
                    f'projects/gender_experiments/evaluation/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/labels.npy',
                    test_labels)


        # if we want quadratic hyperplane also we need to optimize c for better results
        if ARGS.quadratic != None:
            # Training with full data
            alpha_star = train_SVM_nonlinear(data, labels, C, c=c_of_kernel, d=D, k=K, gamma=default_gamma, re_balance=ARGS.re_balance)
            prediction, scores = predict_non_linear(test_data, data, labels, alpha_star,
                                                    c=c_of_kernel, d=D, k=K, gamma=default_gamma)
            accuracy = (prediction.T == test_labels).sum() / test_labels.size
            print(f'Test error rate: {(1 - accuracy) * 100}')

            # Saving for experiments as a numpy array
            if (ARGS.save == True):
                np.save(
                    f'projects/gender_experiments/evaluation/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/C_{C}_c_{c_of_kernel}_d_{D}_K_{K}_gamma_{default_gamma}_gaussianizer_{ARGS.gaussianizer}_scores.npy',
                    scores)
                # only for one time to be keep right labels
                np.save(
                    f'projects/gender_experiments/evaluation/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/labels.npy',
                    test_labels)
    # dimension reduction analysis
    if ARGS.DR_exp == True:
        score_array = np.zeros((data.shape[0]-1, validation_data_size[0]))
        for i in range(data.shape[0]-1):
            new_data = PCA(data, i+1)
            print(f"LDA applied and dimension has been decreased to {i+1}")

            # Spritting data to train and validation
            (train_data, train_labels), (validation_data, validation_labels) = split_4to1(new_data, labels)

            # if we want linear hyperplane
            if (ARGS.quadratic == None):
                w_star = train_SVM_linear(train_data, train_labels, C=C, K=K)
                predictions, scores = predict_for_linear(validation_data, w_star, K=K)
                accuracy = (predictions == validation_labels).sum() / validation_labels.size
                print(f'Validation error rate: {(1 - accuracy) * 100}')
                # if we want quadratic hyperplane also we need to optimize c for better results
                score_array[i, :] += scores
            else:
                alpha_star = train_SVM_nonlinear(train_data, train_labels, C, c=c_of_kernel, d=D, k=K,
                                                 gamma=default_gamma)
                prediction, scores = predict_non_linear(validation_data, train_data, train_labels, alpha_star,
                                                        c=c_of_kernel, d=D, k=K, gamma=default_gamma)
                accuracy = (prediction.T == validation_labels).sum() / validation_labels.size
                print(f'Validation error rate: {(1 - accuracy) * 100}')
                score_array[i, :] += scores

        np.save(f'projects/gender_experiments/dimension_experiment/SVM/{ARGS.quadratic}scores.npy', score_array)
        np.save(f'projects/gender_experiments/dimension_experiment/SVM/labels.npy', validation_labels)
