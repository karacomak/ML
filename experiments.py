import numpy as np
from evaluation_funcions import *
from plottingFunctions import *
import argparse
from data_preparation_PCA_LDA import *
from gaussian_mixture_model import GMM_classifier
from logistic_regression import l_bfgs_optimizer
from support_vector_machine import train_SVM_nonlinear, predict_non_linear
from statistics import *

##### DEFAULT PARSER SETTINGS #######################


dimension_reduction_caption = 'specify reduction technique PCA or LDA and the dimension that required e.g. --dimension_reduction PCA-4'
setOfPriorAndCostsToExperiment = np.array([[0.5, 1, 1], [0.1, 1, 1], [0.9, 1, 1]])
setOfPriorAndCostsToExperiment_sub = np.array([[0.9, 1, 1], [0.5, 1, 1], [0.1, 1, 1]])

C = 10
K = 0.1
D = 2
c_of_kernel = 1
gamma = 0.01
pi_true = 0.5
component = 64
k_fold = 5
my_lambda = '0.0001'

data_file = 'projects/Gender_Detection/Train.txt'
test_file = 'projects/Gender_Detection/Test.txt'
evaluation_or_validation = 'validations'
validation_data_size = [1200, 4000]
######################################################

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--data_file", type=str, default=data_file, help="test or training data file (default "
                                                                         "projects/Wine_Quality_Detection/Train-Test.txt)",
                        required=False)
    PARSER.add_argument("--classifier", type=str,
                        help="please choose the classifier could be MVG, LR, SVM, GMM. After selection you can specifiy it",
                        required=True)
    PARSER.add_argument("--test", type=bool, default=False, help="if this is test type something", required=False)
    PARSER.add_argument("--plot", type=bool, default=False,
                        help="plot all the gender_experiments for specific classifier", required=False)
    PARSER.add_argument("--naive", type=bool, default=False,
                        help='if you want naive gausian classifier type 1 e.g. --naive 1', required=False)
    PARSER.add_argument("--tied", type=bool, default=False,
                        help='if you want tied gausian classifier type 1 e.g. --tied 1', required=False)
    PARSER.add_argument("--gaussianizer", type=bool, default=False,
                        help='if you want to use Gaussian cumulative distribution, type 1', required=False)
    PARSER.add_argument("--dimension_reduction", type=str, default=None, help=dimension_reduction_caption,
                        required=False)
    PARSER.add_argument("--my_lambda", type=str, default=my_lambda,
                        help='please specify the lambda if you experiment logistic regression 1 e.g. --my_lambda 0.0001',
                        required=False)
    PARSER.add_argument("--quadratic", type=str, default=None,
                        help='for logistic regression type 1 if want to quadratic hyperplane, for SVM specify type RBF or Poly',
                        required=False)
    PARSER.add_argument("--class_balance", type=bool, default=False,
                        help='type 1 if want to make classes balanced with costs', required=False)
    PARSER.add_argument("--C", type=float, default=C, help="C of SVM (default 0.1)", required=False)
    PARSER.add_argument("--K", type=float, default=K, help="bias of SVM (default 10)", required=False)
    PARSER.add_argument("--type", type=str, default=None, help="the type of experiment you can find in the report",
                        required=False)
    PARSER.add_argument("--c_of_kernel", type=int, default=c_of_kernel, help="c of kernel (default 1)",
                        required=False)
    PARSER.add_argument("--gamma", type=float, default=gamma, help="gamma of RBF kernel (default 100)",
                        required=False)
    PARSER.add_argument("--d_of_kernel", type=int, default=D, help="d of kernel (default 2)", required=False)
    PARSER.add_argument("--pi_true", type=float, default=pi_true,
                        help="prior for true class for re-balanced applications (default 0.5, just with class_balance command)",
                        required=False)
    PARSER.add_argument("--component", type=int, default=component, help="number of component for GMM (default 64)",
                        required=False)
    PARSER.add_argument("--k_fold", type=int, default=k_fold,
                        help="k for k-fold cross validation specify k that is required for best parameters",
                        required=False)
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
    if ARGS.d_of_kernel is not None:
        D = ARGS.d_of_kernel
    if ARGS.component is not None:
        component = ARGS.component
    if ARGS.pi_true is not None:
        pi_true = ARGS.pi_true

    if ARGS.test == True:
        evaluation_or_validation = 'evaluation'

    # Data loading
    if ARGS.data_file is not None:
        data_file = ARGS.data_file
    data, labels = loadData(data_file)

    # Gaussian cumulative distribution function and z-score
    if ARGS.gaussianizer is True:
        data = z_score(data)
        data = gaussianizer(data)

    # Dimension reduction
    if ARGS.dimension_reduction is not None:
        reduction_type = ARGS.dimension_reduction.split('-')[0]
        required_dimension = int(ARGS.dimension_reduction.split('-')[1])
        if reduction_type == 'PCA':
            data = PCA(data, required_dimension)
            print(f"PCA applied and dimension has been decreased to {required_dimension}")
        else:
            data = LDA(data, labels, required_dimension)
            print(f"LDA applied and dimension has been decreased to {required_dimension}")

    if ARGS.classifier == 'analysis':
        # dimension reduction analysis
        if ARGS.type == 'DR_analysis':
            # for MVG
            # scores_llr = np.load(f'projects/gender_experiments/dimension_experiment/MVG/MVG_naive_{ARGS.naive}_tied_{ARGS.tied}_gaussianizer_{ARGS.gaussianizer}_llr.npy', 'r')
            # for LR
            # scores_llr = np.load(f'projects/gender_experiments/dimension_experiment/LR/scores.npy', 'r')
            # for SVM
            scores_llr = np.load(f'projects/gender_experiments/dimension_experiment/SVM/RBFscores.npy', 'r')
            labels = np.load(f'projects/gender_experiments/dimension_experiment/SVM/labels.npy', 'r')
            minDCFs = np.zeros((3, scores_llr.shape[0]))
            for j, attributes in enumerate(setOfPriorAndCostsToExperiment):
                priorProb, Cfn, Cfp = attributes
                for i in range(scores_llr.shape[0]):
                    minDCFs[j, i] += minDCFBinary(scores_llr[i, :], labels, priorProb, Cfn, Cfp)

            dimension_reduction_exp_plot(minDCFs)

        # k-fold cross validation for final decision in validation data (this experiment is applied only on validation data)
        # k = 5
        if ARGS.type == 'k_fold_final':
            data, labels = loadData('projects/Gender_Detection/Train.txt')
            set_of_scores = np.zeros((5, 1200))
            set_of_llr = np.zeros((5, 1200))
            set_of_validation_labels = np.zeros((5, 1200))
            for i in range(5):
                (train_data, train_labels), (validation_data, validation_labels) = kFoldCrossValSplit(data, labels, 5,
                                                                                                      i)

                # Training of SVM
                alpha_star = train_SVM_nonlinear(train_data, train_labels, 1, c=1, d=2, k=0.1, gamma=0.01, final=True)
                prediction, scores = predict_non_linear(validation_data, train_data, train_labels, alpha_star,
                                                        c=1, d=2, k=0.1, gamma=0.01, final=True)

                # Training of GMM
                predictions, logSjoint, llr = GMM_classifier(train_data, train_labels, validation_data, tied=False,
                                                             naive=False, C_number=64)

                # keep the scores than calculate min DCF for all
                set_of_scores[i, :] = scores
                set_of_llr[i, :] = llr
                set_of_validation_labels[i, :] = validation_labels

            for ii, subset in enumerate(setOfPriorAndCostsToExperiment):
                priorProb, Cfn, Cfp = subset
                minDCF_SVM = minDCFBinary(set_of_scores.ravel(), set_of_validation_labels.ravel(), priorProb, Cfn, Cfp)
                minDCF_GMM = minDCFBinary(set_of_llr.ravel(), set_of_validation_labels.ravel(), priorProb, Cfn, Cfp)
                if ii == 0:
                    print(f'minDCF for SVM model: {minDCF_SVM}, prior = 0.5')
                    print(f'minDCF for GMM model: {minDCF_GMM}, prior = 0.5')
                if ii == 1:
                    print(f'minDCF for SVM model: {minDCF_SVM}, prior = 0.1')
                    print(f'minDCF for GMM model: {minDCF_GMM}, prior = 0.1')
                if ii == 2:
                    print(f'minDCF for SVM model: {minDCF_SVM}, prior = 0.9')
                    print(f'minDCF for GMM model: {minDCF_GMM}, prior = 0.9')

            np.save('projects/gender_experiments/bests/llr.npy', set_of_llr.ravel())
            np.save('projects/gender_experiments/bests/scores.npy', set_of_scores.ravel())
            np.save('projects/gender_experiments/bests/labels.npy', set_of_validation_labels.ravel())

        if ARGS.test is True:
            SVM_best_scores = np.load(
                f'projects/gender_experiments/evaluation/SVM_classifier_4to1_scores/quadratic_RBF/C_1_c_1_d_2_K_0.1_gamma_0.01_gaussianizer_False_scores.npy',
                'r')
            GMM_best_llr = np.load(
                f'projects/gender_experiments/evaluation/GMM_classifier_4to1_likelihoods/naive_False_tied_False_component_64_gaussianizer_False_llr.npy',
                'r')
            validation_labels = np.load(
                f'projects/gender_experiments/evaluation/GMM_classifier_4to1_likelihoods/labels.npy', 'r')
        else:
            SVM_best_scores = np.load(f'projects/gender_experiments/bests/scores.npy', 'r')
            GMM_best_llr = np.load(f'projects/gender_experiments/bests/llr.npy', 'r')
            validation_labels = np.load(f'projects/gender_experiments/bests/labels.npy', 'r')

        # two feature scatter for GMM
        # scatterPlot(data[1:4, :], labels)

        # ROC plot analysis for validation data
        if ARGS.type == 'ROC':
            for i in setOfPriorAndCostsToExperiment:
                priorProb, Cfn, Cfp = i
                minDCF = minDCFBinary(SVM_best_scores, validation_labels, priorProb, Cfn, Cfp)
                print('minDCF:', minDCF)

            # ROC for bests
            given_array = np.zeros((2, validation_labels.shape[0]))
            given_array[0, :] = SVM_best_scores
            given_array[1, :] = GMM_best_llr
            plotRocCurveForGivenThresholds(given_array, validation_labels)

        # actual cost analysis for validation data
        if ARGS.type == 'actDCF':
            # minDCF vs actDCF for bests
            for j in [0.5, 0.1, 0.9]:
                # theoretical threshold
                theoretical_t = -np.log(j / (1 - j))
                print(f'current threshold is {theoretical_t}')
                # prior and the costs are not considered since we determined a threshold in the next functions
                SVM_actDCF = actualDCFBinary(SVM_best_scores, validation_labels, j, 1, 1, threshold=theoretical_t)
                GMM_actDCF = actualDCFBinary(GMM_best_llr, validation_labels, j, 1, 1, threshold=theoretical_t)
                print('SVM actDCF:', SVM_actDCF)
                print('GMM actDCF:', GMM_actDCF)

        # bayes plot analyisis for validation data
        if ARGS.type == 'bayes_err':
            effPriorLogOdds = np.linspace(-4, 4, 100)
            SVM_actDCF = bayes_error_plot(effPriorLogOdds, SVM_best_scores, validation_labels, minCost=False)
            SVM_minDCF = bayes_error_plot(effPriorLogOdds, SVM_best_scores, validation_labels, minCost=True)
            plt.plot(effPriorLogOdds, SVM_actDCF, label='SVM_actDCF', color='b')
            plt.plot(effPriorLogOdds, SVM_minDCF, label='SVM_minDCF', color='b', linestyle='--')
            GMM_actDCF = bayes_error_plot(effPriorLogOdds, GMM_best_llr, validation_labels, minCost=False)
            GMM_minDCF = bayes_error_plot(effPriorLogOdds, GMM_best_llr, validation_labels, minCost=True)
            plt.plot(effPriorLogOdds, GMM_actDCF, label='GMM_actDCF', color='r')
            plt.plot(effPriorLogOdds, GMM_minDCF, label='GMM_minDCF', color='r', linestyle='--')

            plt.ylim([0, 1.1])
            plt.xlim([-4, 4])
            plt.legend()
            plt.show()

        # efficient threshold analyisis for validation data
        if ARGS.type == 'optimum_th':
            # optimal threshold
            (GMM_train_llr, GMM_train_label), (GMM_val_llr, GMM_val_label) = split_1to1(GMM_best_llr, validation_labels)
            (SVM_train_scores, SVM_train_label), (SVM_val_scores, SVM_val_label) = split_1to1(SVM_best_scores,
                                                                                              validation_labels)

            for j, i in enumerate(setOfPriorAndCostsToExperiment_sub):
                priorProb, Cfn, Cfp = i
                # theoretical threshold
                theoretical_t = -np.log(priorProb / (1 - priorProb))
                print(f'current theoretical threshold is {theoretical_t}')

                SVM_threshold = optimum_threshold_finder(SVM_train_scores, SVM_train_label, priorProb, Cfn, Cfp)
                GMM_threshold = optimum_threshold_finder(GMM_train_llr, GMM_train_label, priorProb, Cfn, Cfp)

                print(f'SVM-threshold: {SVM_threshold}, GMM-threshold:{GMM_threshold}')

                SVM_minDCF = minDCFBinary(SVM_val_scores, SVM_val_label, priorProb, Cfn, Cfp)
                GMM_minDCF = minDCFBinary(GMM_val_llr, GMM_val_label, priorProb, Cfn, Cfp)
                print('SVM minDCF:', SVM_minDCF, f'prior: {priorProb}')
                print('GMM minDCF:', GMM_minDCF, f'prior: {priorProb}')
                SVM_actDCF = actualDCFBinary(SVM_val_scores, SVM_val_label, priorProb, Cfn, Cfp,
                                             threshold=theoretical_t)
                GMM_actDCF = actualDCFBinary(GMM_val_llr, GMM_val_label, priorProb, Cfn, Cfp, threshold=theoretical_t)
                print('SVM actDCF:', SVM_actDCF, f'prior: {priorProb}')
                print('GMM actDCF:', GMM_actDCF, f'prior: {priorProb}')
                SVM_actDCF_star = actualDCFBinary(SVM_val_scores, SVM_val_label, priorProb, Cfn, Cfp,
                                                  threshold=SVM_threshold)
                GMM_actDCF_star = actualDCFBinary(GMM_val_llr, GMM_val_label, priorProb, Cfn, Cfp,
                                                  threshold=GMM_threshold)
                print('SVM actDCF star:', SVM_actDCF_star, f'prior: {priorProb}')
                print('GMM actDCF star:', GMM_actDCF_star, f'prior: {priorProb}')

            SVM_threshold = optimum_threshold_finder(SVM_train_scores, SVM_train_label, 0.5, 1, 1)
            GMM_threshold = optimum_threshold_finder(GMM_train_llr, GMM_train_label, 0.5, 1, 1)

            effPriorLogOdds = np.linspace(-4, 4, 100)
            SVM_actDCF = bayes_error_plot(effPriorLogOdds, SVM_best_scores, validation_labels, minCost=False,
                                          threshold=SVM_threshold)
            SVM_minDCF = bayes_error_plot_2(effPriorLogOdds, SVM_best_scores, validation_labels, minCost=True)
            plt.plot(effPriorLogOdds, SVM_actDCF, label='SVM_actDCF', color='b')
            plt.plot(effPriorLogOdds, SVM_minDCF, label='SVM_minDCF', color='b', linestyle='--')
            GMM_actDCF = bayes_error_plot(effPriorLogOdds, GMM_best_llr, validation_labels, minCost=False,
                                          threshold=GMM_threshold)
            GMM_minDCF = bayes_error_plot(effPriorLogOdds, GMM_best_llr, validation_labels, minCost=True)
            plt.plot(effPriorLogOdds, GMM_actDCF, label='GMM_actDCF', color='r')
            plt.plot(effPriorLogOdds, GMM_minDCF, label='GMM_minDCF', color='r', linestyle='--')

            plt.ylim([0, 1.1])
            plt.xlim([-4, 4])
            plt.legend()
            plt.show()

        # Score calibration for validation data
        if ARGS.type == 'calibration':
            (GMM_train_llr, GMM_train_label), (GMM_val_llr, GMM_val_label) = split_1to1(GMM_best_llr, validation_labels)
            (SVM_train_scores, SVM_train_label), (SVM_val_scores, SVM_val_label) = split_1to1(SVM_best_scores,
                                                                                              validation_labels)

            prior = 0.5

            w_SVM, b_SVM = l_bfgs_optimizer(oneToX(SVM_train_scores), SVM_train_label, 1e-4, prior=prior)
            calibrated_SVM = w_SVM * SVM_val_scores + b_SVM - np.log(prior / (1 - prior))

            w_GMM, b_GMM = l_bfgs_optimizer(oneToX(GMM_train_llr), GMM_train_label, 1e-4, prior=prior)
            calibrated_GMM = w_GMM.T * GMM_val_llr + b_GMM - np.log(prior / (1 - prior))

            # minDCF vs actDCF for bests
            for j in setOfPriorAndCostsToExperiment:
                priorProb, Cfn, Cfp = j
                t_threshold = - np.log(priorProb/(1-priorProb))
                print(f'prior:{priorProb}')

                SVM_minDCF = minDCFBinary(SVM_val_scores, SVM_val_label, priorProb, Cfn, Cfp)
                GMM_minDCF = minDCFBinary(GMM_val_llr, GMM_val_label, priorProb, Cfn, Cfp)
                print('SVM minDCF:', SVM_minDCF)
                print('GMM minDCF:', GMM_minDCF)

                SVM_minDCF = actualDCFBinary(SVM_val_scores, SVM_val_label, priorProb, Cfn, Cfp, threshold=t_threshold)
                GMM_minDCF = actualDCFBinary(GMM_val_llr, GMM_val_label, priorProb, Cfn, Cfp, threshold=t_threshold)
                print('SVM actDCF:', SVM_minDCF)
                print('GMM actDCF:', GMM_minDCF)

                SVM_actDCF = actualDCFBinary(calibrated_SVM, SVM_val_label, priorProb, Cfn, Cfp, threshold=t_threshold)
                GMM_actDCF = actualDCFBinary(calibrated_GMM, GMM_val_label, priorProb, Cfn, Cfp, threshold=t_threshold)
                print('SVM actDCF (cal):', SVM_actDCF)
                print('GMM actDCF (cal):', GMM_actDCF)

            effPriorLogOdds = np.linspace(-4, 4, 100)
            SVM_actDCF = bayes_error_plot(effPriorLogOdds, calibrated_SVM, SVM_val_label, minCost=False)
            SVM_minDCF = bayes_error_plot_2(effPriorLogOdds, SVM_val_scores, GMM_val_label, minCost=True)
            plt.plot(effPriorLogOdds, SVM_actDCF, label='SVM_actDCF', color='b')
            plt.plot(effPriorLogOdds, SVM_minDCF, label='SVM_minDCF', color='b', linestyle='--')
            GMM_actDCF = bayes_error_plot(effPriorLogOdds, calibrated_GMM, SVM_val_label, minCost=False)
            GMM_minDCF = bayes_error_plot(effPriorLogOdds, GMM_val_llr, GMM_val_label, minCost=True)
            plt.plot(effPriorLogOdds, GMM_actDCF, label='GMM_actDCF', color='r')
            plt.plot(effPriorLogOdds, GMM_minDCF, label='GMM_minDCF', color='r', linestyle='--')

            plt.ylim([0, 1.1])
            plt.xlim([-4, 4])
            plt.legend()
            plt.show()

        # Fusion for validation part
        if ARGS.type == 'fusion':
            (GMM_train_llr, GMM_train_label), (GMM_val_llr, GMM_val_label) = split_1to1(GMM_best_llr, validation_labels)
            (SVM_train_scores, SVM_train_label), (SVM_val_scores, SVM_val_label) = split_1to1(SVM_best_scores,
                                                                                              validation_labels)

            prior = 0.5
            w_SVM, b_SVM = l_bfgs_optimizer(oneToX(SVM_train_scores), SVM_train_label, 1e-4, prior=prior)
            calibrated_SVM = w_SVM * SVM_val_scores + b_SVM - np.log(prior / (1 - prior))

            w_GMM, b_GMM = l_bfgs_optimizer(oneToX(GMM_train_llr), GMM_train_label, 1e-4, prior=prior)
            calibrated_GMM = w_GMM.T * GMM_val_llr + b_GMM - np.log(prior / (1 - prior))

            fusion = (w_SVM * SVM_val_scores) + (w_GMM.T * GMM_val_llr) + b_SVM + b_GMM - np.log(prior / (1 - prior))

            # actual costs
            for j in [0.5, 0.1, 0.9]:
                # theoretical threshold
                theoretical_t = -np.log(j / (1 - j))
                print(f'current threshold is {theoretical_t}')
                # prior and the costs are not considered since we determined a threshold in the next functions
                SVM_actDCF = actualDCFBinary(calibrated_SVM, SVM_val_label, j, 1, 1, threshold=theoretical_t)
                GMM_actDCF = actualDCFBinary(calibrated_GMM, GMM_val_label, j, 1, 1, threshold=theoretical_t)
                fusion_actDCF = actualDCFBinary(fusion, GMM_val_label, j, 1, 1, threshold=theoretical_t)
                print('SVM actDCF:', SVM_actDCF)
                print('GMM actDCF:', GMM_actDCF)
                print('fusion actDCF:', fusion_actDCF)

                #min DCFs
                SVM_minDCF = minDCFBinary(SVM_val_scores, SVM_val_label, j, 1, 1)
                GMM_minDCF = minDCFBinary(GMM_val_llr, GMM_val_label, j, 1, 1)
                fusion_minDCF = minDCFBinary(fusion, GMM_val_label, j, 1, 1)
                print('SVM minDCF:', SVM_minDCF)
                print('GMM minDCF:', GMM_minDCF)
                print('fusion minDCF:', fusion_minDCF)


            # ROC for bests and fusion
            given_array = np.zeros((3, SVM_val_label.shape[0]))
            given_array[0, :] = calibrated_SVM
            given_array[1, :] = calibrated_GMM
            given_array[2, :] = fusion
            plotRocCurveForGivenThresholds(given_array, SVM_val_label)

            #bayes plot with fusion
            effPriorLogOdds = np.linspace(-4, 4, 100)
            SVM_actDCF = bayes_error_plot(effPriorLogOdds, calibrated_SVM, SVM_val_label, minCost=False)
            plt.plot(effPriorLogOdds, SVM_actDCF, label='SVM_actDCF_cal', color='b')

            fusion_actDCF = bayes_error_plot(effPriorLogOdds, fusion, SVM_val_label, minCost=False)
            fusion_minDCF = bayes_error_plot_2(effPriorLogOdds, fusion, SVM_val_label, minCost=True)
            plt.plot(effPriorLogOdds, fusion_actDCF, label='fusion_actDCF', color='green')
            plt.plot(effPriorLogOdds, fusion_minDCF, label='fusion_minDCF', color='green', linestyle='--')

            GMM_actDCF = bayes_error_plot(effPriorLogOdds, calibrated_GMM, GMM_val_label, minCost=False)
            plt.plot(effPriorLogOdds, GMM_actDCF, label='GMM_actDCF_cal', color='r')

            plt.ylim([0, 1.1])
            plt.xlim([-4, 4])
            plt.legend()
            plt.show()

        # efficient threshoold for evaluation data
        if ARGS.type == 'optimum_th_test':
            # optimal threshold
            SVM_best_scores_test = np.load(
                f'projects/gender_experiments/evaluation/SVM_classifier_4to1_scores/quadratic_RBF/C_1_c_1_d_2_K_0.1_gamma_0.01_gaussianizer_False_scores.npy',
                'r')
            GMM_best_llr_test = np.load(
                f'projects/gender_experiments/evaluation/GMM_classifier_4to1_likelihoods/naive_False_tied_False_component_64_gaussianizer_False_llr.npy',
                'r')
            evaluation_labels = np.load(
                f'projects/gender_experiments/evaluation/GMM_classifier_4to1_likelihoods/labels.npy', 'r')

            for j, i in enumerate(setOfPriorAndCostsToExperiment_sub):
                priorProb, Cfn, Cfp = i
                # theoretical threshold
                theoretical_t = -np.log(priorProb / (1 - priorProb))
                print(f'current theoretical threshold is {theoretical_t}')

                SVM_threshold = optimum_threshold_finder(SVM_best_scores, validation_labels, priorProb, Cfn, Cfp)
                GMM_threshold = optimum_threshold_finder(GMM_best_llr, validation_labels, priorProb, Cfn, Cfp)

                print(f'SVM-threshold: {SVM_threshold}, GMM-threshold:{GMM_threshold}')

                SVM_minDCF = minDCFBinary(SVM_best_scores_test, evaluation_labels, priorProb, Cfn, Cfp)
                GMM_minDCF = minDCFBinary(GMM_best_llr_test, evaluation_labels, priorProb, Cfn, Cfp)
                print('SVM minDCF:', SVM_minDCF, f'prior: {priorProb}')
                print('GMM minDCF:', GMM_minDCF, f'prior: {priorProb}')
                SVM_actDCF = actualDCFBinary(SVM_best_scores_test, evaluation_labels, priorProb, Cfn, Cfp,
                                             threshold=theoretical_t)
                GMM_actDCF = actualDCFBinary(GMM_best_llr_test, evaluation_labels, priorProb, Cfn, Cfp, threshold=theoretical_t)
                print('SVM actDCF:', SVM_actDCF, f'prior: {priorProb}')
                print('GMM actDCF:', GMM_actDCF, f'prior: {priorProb}')
                SVM_actDCF_star = actualDCFBinary(SVM_best_scores_test, evaluation_labels, priorProb, Cfn, Cfp,
                                                  threshold=SVM_threshold)
                GMM_actDCF_star = actualDCFBinary(GMM_best_llr_test, evaluation_labels, priorProb, Cfn, Cfp,
                                                  threshold=GMM_threshold)
                print('SVM actDCF star:', SVM_actDCF_star, f'prior: {priorProb}')
                print('GMM actDCF star:', GMM_actDCF_star, f'prior: {priorProb}')

        # fusion for evaluation part
        if ARGS.type == 'fusion_test':
            SVM_best_scores_test = np.load(
                f'projects/gender_experiments/evaluation/SVM_classifier_4to1_scores/quadratic_RBF/C_1_c_1_d_2_K_0.1_gamma_0.01_gaussianizer_False_scores.npy',
                'r')
            GMM_best_llr_test = np.load(
                f'projects/gender_experiments/evaluation/GMM_classifier_4to1_likelihoods/naive_False_tied_False_component_64_gaussianizer_False_llr.npy',
                'r')
            evaluation_labels = np.load(
                f'projects/gender_experiments/evaluation/GMM_classifier_4to1_likelihoods/labels.npy', 'r')


            prior = 0.5
            w_SVM, b_SVM = l_bfgs_optimizer(oneToX(SVM_best_scores), validation_labels, 1e-4, prior=prior)
            calibrated_SVM = w_SVM * SVM_best_scores_test + b_SVM - np.log(prior / (1 - prior))

            w_GMM, b_GMM = l_bfgs_optimizer(oneToX(GMM_best_llr), validation_labels, 1e-4, prior=prior)
            calibrated_GMM = w_GMM.T * GMM_best_llr_test + b_GMM - np.log(prior / (1 - prior))

            fusion = (w_SVM * SVM_best_scores_test) + (w_GMM.T * GMM_best_llr_test) + b_SVM + b_GMM - np.log(prior / (1 - prior))

            # actual costs
            for j in [0.5, 0.1, 0.9]:
                # theoretical threshold
                theoretical_t = -np.log(j / (1 - j))
                print(f'current threshold is {theoretical_t}')
                # prior and the costs are not considered since we determined a threshold in the next functions
                SVM_actDCF = actualDCFBinary(SVM_best_scores_test, evaluation_labels, j, 1, 1, threshold=theoretical_t)
                GMM_actDCF = actualDCFBinary(GMM_best_llr_test, evaluation_labels, j, 1, 1, threshold=theoretical_t)
                fusion_actDCF = actualDCFBinary(fusion, evaluation_labels, j, 1, 1, threshold=theoretical_t)
                print('SVM actDCF:', SVM_actDCF)
                print('GMM actDCF:', GMM_actDCF)
                print('fusion actDCF:', fusion_actDCF)


                #calibrated scores
                SVM_actDCF_cal = actualDCFBinary(calibrated_SVM, evaluation_labels, j, 1, 1, threshold=theoretical_t)
                GMM_actDCF_cal = actualDCFBinary(calibrated_GMM, evaluation_labels, j, 1, 1, threshold=theoretical_t)
                fusion_actDCF_cal = actualDCFBinary(fusion, evaluation_labels, j, 1, 1, threshold=theoretical_t)
                print('SVM actDCF calibrated:', SVM_actDCF_cal)
                print('GMM actDCF calibrated:', GMM_actDCF_cal)
                print('fusion actDCF calibrated:', fusion_actDCF_cal)

                # min DCFs
                SVM_minDCF = minDCFBinary(SVM_best_scores_test, evaluation_labels, j, 1, 1)
                GMM_minDCF = minDCFBinary(GMM_best_llr_test, evaluation_labels, j, 1, 1)
                fusion_minDCF = minDCFBinary(fusion, evaluation_labels, j, 1, 1)
                print('SVM minDCF:', SVM_minDCF)
                print('GMM minDCF:', GMM_minDCF)
                print('fusion minDCF:', fusion_minDCF)


            # ROC for bests and fusion
            given_array = np.zeros((3, evaluation_labels.shape[0]))
            given_array[0, :] = calibrated_SVM
            given_array[1, :] = calibrated_GMM
            given_array[2, :] = fusion
            plotRocCurveForGivenThresholds(given_array, evaluation_labels)

            # bayes plot with fusion
            effPriorLogOdds = np.linspace(-4, 4, 100)
            SVM_actDCF = bayes_error_plot(effPriorLogOdds, calibrated_SVM, evaluation_labels, minCost=False)
            #SVM_minDCF = bayes_error_plot_2(effPriorLogOdds, SVM_best_scores_test, evaluation_labels, minCost=True)
            plt.plot(effPriorLogOdds, SVM_actDCF, label='SVM_actDCF (cal)', color='b')
            #plt.plot(effPriorLogOdds, SVM_minDCF, label='SVM_minDCF', color='b', linestyle='--')

            fusion_actDCF = bayes_error_plot(effPriorLogOdds, fusion, evaluation_labels, minCost=False)
            fusion_minDCF = bayes_error_plot_2(effPriorLogOdds, fusion, evaluation_labels, minCost=True)
            plt.plot(effPriorLogOdds, fusion_actDCF, label='fusion_actDCF', color='green')
            plt.plot(effPriorLogOdds, fusion_minDCF, label='fusion_minDCF', color='green', linestyle='--')

            GMM_actDCF = bayes_error_plot(effPriorLogOdds, calibrated_GMM, evaluation_labels, minCost=False)
            #GMM_minDCF = bayes_error_plot(effPriorLogOdds, GMM_best_llr_test, evaluation_labels, minCost=True)
            plt.plot(effPriorLogOdds, GMM_actDCF, label='GMM_actDCF (cal)', color='r')
            #plt.plot(effPriorLogOdds, GMM_minDCF, label='GMM_minDCF', color='r', linestyle='--')

            plt.ylim([0, 1.1])
            plt.xlim([-4, 4])
            plt.legend()
            plt.show()


    # multivariate gaussian classifier experiment
    elif ARGS.classifier == 'MVG':
        # Gaussian classifier with required type
        gaussian_classifier_4to1_likelihoods = np.load(
            f'projects/gender_experiments/{evaluation_or_validation}/gaussian_classifier_4to1_likelihoods/naive_{ARGS.naive}_tied_{ARGS.tied}_gaussianizer_{ARGS.gaussianizer}_DR_{ARGS.dimension_reduction}_llr.npy',
            'r')
        gaussian_classifier_4to1_labels = np.load(
            f'projects/gender_experiments/{evaluation_or_validation}/gaussian_classifier_4to1_likelihoods/labels.npy',
            'r')
        print(
            f'File name: naive_{ARGS.naive}_tied_{ARGS.tied}_gaussianizer_{ARGS.gaussianizer}_DR_{ARGS.dimension_reduction}_llr.npy')
        for i in setOfPriorAndCostsToExperiment:
            priorProb, Cfn, Cfp = i
            minDCF = minDCFBinary(gaussian_classifier_4to1_likelihoods, gaussian_classifier_4to1_labels, priorProb, Cfn,
                                  Cfp)
            print('minDCF:', minDCF)

    # logistic regression classifier experiment
    elif ARGS.classifier == 'LR':
        # ---linear----
        if ARGS.quadratic == None:

            if ARGS.plot == True:
                # Best LR selection with k-fold 5
                my_lambdas = np.load(
                    f'projects/gender_experiments/validations/LR_classifier_k-fold_minDCFs_for_best_lambda/quadratic_{ARGS.quadratic}/lambdas.npy',
                    'r')
                minDCFs = np.load(
                    f'projects/gender_experiments/validations/LR_classifier_k-fold_minDCFs_for_best_lambda/quadratic_{ARGS.quadratic}/gaussianizer_{ARGS.gaussianizer}_minDCFs.npy',
                    'r')
                if ARGS.test == True:
                    minDCFs_test = np.load(
                        f'projects/gender_experiments/evaluation/LR_classifier_k-fold_minDCFs_for_best_lambda/quadratic_{ARGS.quadratic}/gaussianizer_{ARGS.gaussianizer}_minDCFs.npy',
                        'r')
                    best_lambda_plot_test(minDCFs, minDCFs_test)
                else:
                    # line graph to virtualize changes of minDFC with different parameter
                    best_lambda_plot(minDCFs)

            # minDCF results with selected parameters
            S = np.load(
                f'projects/gender_experiments/{evaluation_or_validation}/LR_classifier_4to1_scores/quadratic_{ARGS.quadratic}/lr_{ARGS.my_lambda}_gaussianizer_{ARGS.gaussianizer}_DR_{ARGS.dimension_reduction}_scores.npy')
            labels = np.load(
                f'projects/gender_experiments/{evaluation_or_validation}/LR_classifier_4to1_scores/quadratic_{ARGS.quadratic}/labels.npy')

            print(
                f'File name: lr_{ARGS.my_lambda}_gaussianizer_{ARGS.gaussianizer}_DR_{ARGS.dimension_reduction}_scores.npy')
            if ARGS.class_balance == False:
                for i in setOfPriorAndCostsToExperiment:
                    priorProb, Cfn, Cfp = i
                    minDCF = minDCFBinary(S, labels, priorProb, Cfn, Cfp)
                    print('minDCF:', minDCF)
            else:
                PIempT = np.load(
                    f'projects/gender_experiments/{evaluation_or_validation}/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/PIempT.npy',
                    'r')
                PIempF = np.load(
                    f'projects/gender_experiments/{evaluation_or_validation}/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/PIempF.npy',
                    'r')
                setOfPriorAndCostsToExperiment = np.array(
                    [[0.5, 1 * pi_true / PIempT, 1 * (1 - pi_true) / PIempF],
                     [0.1, 1 * pi_true / PIempT, 1 * (1 - pi_true) / PIempF],
                     [0.9, 1 * pi_true / PIempT, 1 * (1 - pi_true) / PIempF]])
                for i in setOfPriorAndCostsToExperiment:
                    priorProb, Cfn, Cfp = i
                    minDCF = minDCFBinary(S, labels, priorProb, Cfn, Cfp)
                    print('minDCF:', minDCF)

        # ---quadratic version of logistic reggression but not reported it produces bad results----
        else:
            # Best LR selection with k-fold 5
            my_lambdas = np.load(
                f'projects/gender_experiments/{evaluation_or_validation}/LR_classifier_k-fold_minDCFs_for_best_alpha/quadratic_{ARGS.quadratic}/LRs.npy',
                'r')
            minDCFs = np.load(
                f'projects/gender_experiments/{evaluation_or_validation}/LR_classifier_k-fold_minDCFs_for_best_alpha/quadratic_{ARGS.quadratic}/gaussianizer_{ARGS.gaussianizer}_minDCFs.npy',
                'r')

            # line graph to virtualize changes of minDFC with different parameters
            best_lambda_plot()

            # minDCF results with selected parameters
            S = np.load(
                f'projects/gender_experiments/{evaluation_or_validation}/LR_classifier_4to1_scores/quadratic_{ARGS.quadratic}/lr_{ARGS.my_lambda}_gaussianizer_{ARGS.gaussianizer}_DR_{ARGS.dimension_reduction}_scores.npy')
            labels = np.load(
                f'projects/gender_experiments/{evaluation_or_validation}/LR_classifier_4to1_scores/quadratic_{ARGS.quadratic}/labels.npy')

            print(
                f'File name: lr_{ARGS.my_lambda}_gaussianizer_{ARGS.gaussianizer}_DR_{ARGS.dimension_reduction}_scores.npy')
            for i in setOfPriorAndCostsToExperiment:
                priorProb, Cfn, Cfp = i
                minDCF = minDCFBinary(S, labels, priorProb, Cfn, Cfp)
                print('minDCF:', minDCF)

    # support vector machine classifier
    elif ARGS.classifier == 'SVM':
        # ---linear----
        if ARGS.quadratic == None:
            if ARGS.plot == True:
                # Best parameter selection with k-fold 5 or single fold
                Cs = np.load(
                    f'projects/gender_experiments/{evaluation_or_validation}/SVM_classifier_k-fold_minDCFs_for_best_C/quadratic_{ARGS.quadratic}/Cs.npy',
                    'r')
                Ks = np.load(
                    f'projects/gender_experiments/{evaluation_or_validation}/SVM_classifier_k-fold_minDCFs_for_best_C/quadratic_{ARGS.quadratic}/Ks.npy',
                    'r')
                minDCFs = np.load(
                    f'projects/gender_experiments/{evaluation_or_validation}/SVM_classifier_k-fold_minDCFs_for_best_C/quadratic_{ARGS.quadratic}/balanced_{ARGS.class_balance}_gaussianizer_{ARGS.gaussianizer}_minDCFs.npy',
                    'r')

                # line graph to virtualize changes of minDFC with different C but constant K
                # best_C_plot(minDCFs, Cs, Ks)

                # for narrowed plots
                Ks = Ks[Cs != "100"]
                minDCFs = minDCFs[:, Cs != "100"]
                Cs = Cs[Cs != "100"]

                # 3D bar chart to find the optimal combination of C and K
                threeD_bar_chart(Cs, Ks, minDCFs[0, :], 'C in log', 'K in log', 'minDFCs')

            scores = np.load(
                f'projects/gender_experiments/{evaluation_or_validation}/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/C_{C}_K_{K}_gaussianizer_{ARGS.gaussianizer}_scores.npy',
                'r')
            labels = np.load(
                f'projects/gender_experiments/{evaluation_or_validation}/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/labels.npy',
                'r')

            print(f'File name: C_{C}_K_{K}_gaussianizer_{ARGS.gaussianizer}_scores.npy')

            if (ARGS.class_balance == False):
                for i in setOfPriorAndCostsToExperiment:
                    priorProb, Cfn, Cfp = i
                    minDCF = minDCFBinary(scores, labels, priorProb, Cfn, Cfp)
                    print('minDCF:', minDCF)
            else:
                # emprival numbers of each class
                PIempT = np.load(
                    f'projects/gender_experiments/{evaluation_or_validation}/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/PIempT.npy',
                    'r')
                PIempF = np.load(
                    f'projects/gender_experiments/{evaluation_or_validation}/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/PIempF.npy',
                    'r')
                # minDCF calculator
                setOfPriorAndCostsToExperiment = np.array(
                    [[0.5, 1 * pi_true / PIempT, 1 * (1 - pi_true) / PIempF],
                     [0.1, 1 * pi_true / PIempT, 1 * (1 - pi_true) / PIempF],
                     [0.9, 1 * pi_true / PIempT, 1 * (1 - pi_true) / PIempF]])
                for i in setOfPriorAndCostsToExperiment:
                    priorProb, Cfn, Cfp = i
                    minDCF = minDCFBinary(scores, labels, priorProb, Cfn, Cfp)
                    print('minDCF:', minDCF)
        # quadratic support vector machine classifier
        else:
            if ARGS.plot == True:
                # Best C and c selection with k-fold 5
                Cs = np.load(
                    f'projects/gender_experiments/{evaluation_or_validation}/SVM_classifier_k-fold_minDCFs_for_best_C/quadratic_{ARGS.quadratic}/Cs.npy',
                    'r')
                c_or_gamma = np.load(
                    f'projects/gender_experiments/{evaluation_or_validation}/SVM_classifier_k-fold_minDCFs_for_best_C/quadratic_{ARGS.quadratic}/Ks.npy',
                    'r')
                minDCFs = np.load(
                    f'projects/gender_experiments/{evaluation_or_validation}/SVM_classifier_k-fold_minDCFs_for_best_C/quadratic_{ARGS.quadratic}/balanced_{ARGS.class_balance}_gaussianizer_{ARGS.gaussianizer}_minDCFs.npy',
                    'r')

                if ARGS.test:
                    minDCFs_val = np.load(
                        f'projects/gender_experiments/validations/SVM_classifier_k-fold_minDCFs_for_best_C/quadratic_{ARGS.quadratic}/balanced_{ARGS.class_balance}_gaussianizer_{ARGS.gaussianizer}_minDCFs.npy',
                        'r')

                # for narrowed plots
                for i in ["0.1", "1", "10", "100"]:
                    if ARGS.test:
                        minDCFs_val = minDCFs_val[:, c_or_gamma != i]
                    minDCFs = minDCFs[:, c_or_gamma != i]
                    Cs = Cs[c_or_gamma != i]
                    c_or_gamma = c_or_gamma[c_or_gamma != i]

                # minDCFs = minDCFs[:, c_or_gamma == '0.1']
                # Cs = Cs[c_or_gamma == '0.1']

                if ARGS.test is False:
                    # line graph to virtualize changes of minDFC with different parameters
                    best_C_plot(minDCFs, c_or_gamma, Cs)

                    # 3D bar chart to find the optimal combination of C and c of quadratic kernel
                    # threeD_bar_chart(Cs, c_or_gamma, minDCFs[0,:], 'C in log', 'c of kernel in log', 'minDFCs')
                else:
                    # line graph to virtualize changes of minDFC with different parameters
                    best_C_plot_test(minDCFs, minDCFs_val, Cs)

                    # 3D bar chart to find the optimal combination of C and c of quadratic kernel
                    # threeD_bar_chart(Cs, c_or_gamma, minDCFs[0,:], 'C in log', 'gamma in log', 'minDFCs')

            scores = np.load(
                f'projects/gender_experiments/{evaluation_or_validation}/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/C_{ARGS.C}_c_{ARGS.c_of_kernel}_d_{D}_K_{K}_gamma_{gamma}_gaussianizer_{ARGS.gaussianizer}_scores.npy',
                'r')
            labels = np.load(
                f'projects/gender_experiments/{evaluation_or_validation}/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/labels.npy',
                'r')

            print(f'File name: C_{C}_c_{c_of_kernel}_d_{D}_K_{K}_gaussianizer_{ARGS.gaussianizer}_scores.npy')

            if (ARGS.class_balance == False):
                for i in setOfPriorAndCostsToExperiment:
                    priorProb, Cfn, Cfp = i
                    minDCF = minDCFBinary(scores, labels, priorProb, Cfn, Cfp)
                    print('minDCF:', minDCF)
            else:
                PIempT = np.load(
                    f'projects/gender_experiments/{evaluation_or_validation}/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/PIempT.npy',
                    'r')
                PIempF = np.load(
                    f'projects/gender_experiments/{evaluation_or_validation}/SVM_classifier_4to1_scores/quadratic_{ARGS.quadratic}/PIempF.npy',
                    'r')
                setOfPriorAndCostsToExperiment = np.array(
                    [[0.5, 1 * pi_true / PIempT, 1 * (1 - pi_true) / PIempF],
                     [0.1, 1 * pi_true / PIempT, 1 * (1 - pi_true) / PIempF],
                     [0.9, 1 * pi_true / PIempT, 1 * (1 - pi_true) / PIempF]])
                for i in setOfPriorAndCostsToExperiment:
                    priorProb, Cfn, Cfp = i
                    minDCF = minDCFBinary(scores, labels, priorProb, Cfn, Cfp)
                    print('minDCF:', minDCF)

    # gaussian mixture model classifier
    elif ARGS.classifier == 'GMM':
        # load the lists that experimented

        if ARGS.plot == True:
            minDCFs_gau = np.load(
                f'projects/gender_experiments/{evaluation_or_validation}/GMM_classifier_component_experiment/naive_{ARGS.naive}_tied_{ARGS.tied}_gaussianizer_{ARGS.gaussianizer}_until_{ARGS.component}_minDCFs_gau.npy',
                'r')
            minDCFs_raw = np.load(
                f'projects/gender_experiments/{evaluation_or_validation}/GMM_classifier_component_experiment/naive_{ARGS.naive}_tied_{ARGS.tied}_gaussianizer_{ARGS.gaussianizer}_until_{ARGS.component}_minDCFs_raw.npy',
                'r')
            list_of_components = np.load(
                f'projects/gender_experiments/{evaluation_or_validation}/GMM_classifier_component_experiment/naive_{ARGS.naive}_tied_{ARGS.tied}_gaussianizer_{ARGS.gaussianizer}_until_{ARGS.component}_list_of_components.npy',
                'r')

            bar_chart_special_for_component_analysis(list_of_components, minDCFs_raw, minDCFs_gau,
                                                     'Component Experiment', 'number of component', 'minDCF')
            if ARGS.test == True:
                minDCFs_gau2 = np.load(
                    f'projects/gender_experiments/validations/GMM_classifier_component_experiment/naive_{ARGS.naive}_tied_{ARGS.tied}_gaussianizer_{ARGS.gaussianizer}_until_{ARGS.component}_minDCFs_gau.npy',
                    'r')
                minDCFs_raw2 = np.load(
                    f'projects/gender_experiments/validations/GMM_classifier_component_experiment/naive_{ARGS.naive}_tied_{ARGS.tied}_gaussianizer_{ARGS.gaussianizer}_until_{ARGS.component}_minDCFs_raw.npy',
                    'r')

                bar_chart_special_for_component_analysis_test(list_of_components, minDCFs_raw2, minDCFs_gau2,
                                                              minDCFs_raw, minDCFs_gau, 'Component Experiment',
                                                              'number of component', 'minDCF')

        # experiment for specific component
        labels = np.load(
            f'projects/gender_experiments/{evaluation_or_validation}/GMM_classifier_4to1_likelihoods/labels.npy', 'r')
        llr = np.load(
            f'projects/gender_experiments/{evaluation_or_validation}/GMM_classifier_4to1_likelihoods/naive_{ARGS.naive}_tied_{ARGS.tied}_component_{ARGS.component}_gaussianizer_{ARGS.gaussianizer}_llr.npy',
            'r')

        print(
            f'naive_{ARGS.naive}_tied_{ARGS.tied}_component_{ARGS.component}_llr.npy')

        for i in setOfPriorAndCostsToExperiment:
            priorProb, Cfn, Cfp = i
            minDCF = minDCFBinary(llr, labels, priorProb, Cfn, Cfp)
            print('minDCF:', minDCF)
