import scipy.special
from gaussianClassifier import *
import numpy as np
from evaluation_funcions import minDCFBinary

##### DEFAULT PARSER SETTINGS #######################
data_file = 'projects/Gender_Detection/Train.txt'
test_file = 'projects/Gender_Detection/Test.txt'
k_fold = None

PSI = 0.001
criteria = 1e-6
number_of_component = 64
alpha = 0.0001
priors = [0.5, 0.5]
list_of_components = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]

dimension_reduction_caption = 'specify reduction technique PCA or LDA and the dimension that required e.g. --dimension_reduction PCA-4'

evaluation_or_validation = 'validations'


######################################################

# calculates the likelihood of given samples for the components
def GMM_ll_per_sample(data, gmm):
    G = len(gmm)
    N = data.shape[1]
    S = np.zeros((G, N))
    for g in range(G):
        S[g, :] = logpdf_GAU_ND(data, gmm[g][2], gmm[g][1]) + np.log(gmm[g][0])
    return scipy.special.logsumexp(S, axis=0)

# for the tied models
def GMM_tied(GMM, Z_vec, N):
    tied_Sigma = np.zeros((GMM[0][2].shape[0], GMM[0][2].shape[0]))
    for g in range((len(GMM))):
        tied_Sigma += GMM[g][2] * Z_vec[g]
    tied_Sigma = (1 / N) * tied_Sigma
    for g in range((len(GMM))):
        GMM[g] = (GMM[g][0], GMM[g][1], tied_Sigma)
    return GMM


def GMM_EM(data, gmm, psi=PSI, cri=criteria, tied=False, naive=False):
    ll_new = None
    ll_old = None
    G = len(gmm)
    N = data.shape[1]

    while ll_old is None or ll_new - ll_old > cri:
        ll_old = ll_new
        SJ = np.zeros((G, N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND(data, gmm[g][2], gmm[g][1]) + np.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis=0)
        ll_new = SM.sum() / N
        P = np.exp(SJ - SM)
        gmmNew = []

        Z_list = np.zeros((G))
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            Z_list[g] = Z
            F = (oneToX(gamma) * data).sum(1)
            S = np.dot(data, (oneToX(gamma) * data).T)

            w = Z / N
            mu = xToOne(F / Z)
            Sigma = S / Z - np.dot(mu, mu.T)

            U, s, _ = np.linalg.svd(Sigma)
            s[s < psi] = psi
            Sigma = np.dot(U, xToOne(s) * U.T)

            if naive:
                Sigma = extractDiagonal(Sigma)

            gmmNew.append((w, mu, Sigma))
        if tied:
            gmmNew = GMM_tied(gmmNew, Z_list, N)
        gmm = gmmNew

        print(f'll_new: {ll_new}')
    print(f'll_new-ll_old: {ll_new - ll_old}')

    return gmm


# creator of components doubles the given components
def LGB(gmm, alpha):
    G = len(gmm)
    new_gmm_list = []
    for i in range(G):
        w, means, C = gmm[i]
        U, s, Vh = np.linalg.svd(C)
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        child_1 = (w / 2, means + d, C)
        child_2 = (w / 2, means - d, C)
        new_gmm_list.append(child_1)
        new_gmm_list.append(child_2)

    return new_gmm_list


def GMM_classifier(train_data, train_labels, validation_data, C_number=number_of_component, tied=False, naive=False,
                   class_prior=priors):
    logSjoint = np.zeros((np.unique(train_labels).size, validation_data.shape[1]))
    likelihood = np.zeros((np.unique(train_labels).size, validation_data.shape[1]))
    for i in np.unique(train_labels):
        current_class_training_data = train_data[:, train_labels == i]
        C, means = covarianceMatrix(current_class_training_data)
        if tied == True:
            C = tiedCovariance(train_data, train_labels)
        if naive == True:
            C = extractDiagonal(C)
        new_gmm = [(1.0, means, C)]
        while len(new_gmm) < C_number:
            optimized_gmm = GMM_EM(current_class_training_data, new_gmm, tied=tied, naive=naive)
            new_gmm = LGB(optimized_gmm, alpha)
        S = GMM_ll_per_sample(validation_data, new_gmm)
        logSjoint[int(i), :] = S + np.log(class_prior[int(i)])
        likelihood[int(i), :] = S

    logSjoint = np.exp(logSjoint)
    predictions = logSjoint.argmax(0)

    binary_likelihood_ratio = likelihood[1, :] - likelihood[0, :]

    return predictions, logSjoint, binary_likelihood_ratio


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--data_file", type=str, default=data_file, help="test or training data file (default "
                                                                         "projects/Wine_Quality_Detection/Train-Test.txt)",
                        required=False)
    PARSER.add_argument("--test", type=bool, default=False, help="if this is test type something", required=False)
    PARSER.add_argument("--dimension_reduction", type=str, default=None, help=dimension_reduction_caption,
                        required=False)
    PARSER.add_argument("--naive", type=bool, default=False,
                        help='if you want naive gausian classifier type 1 e.g. --naive 1', required=False)
    PARSER.add_argument("--tied", type=bool, default=False,
                        help='if you want tied gausian classifier type 1 e.g. --tied 1', required=False)
    PARSER.add_argument("--confusion_matrix", type=bool, default=False, help='type 1 if confusion matrix is needed',
                        required=False)
    PARSER.add_argument("--save", type=bool, default=False, help='type 1 if want to save llr', required=False)
    PARSER.add_argument("--c_exp", type=bool, default=False,
                        help='component experiment, type 1 (set manually the set of experiment deault until 64)',
                        required=False)
    PARSER.add_argument("--PSI", type=float, default=PSI, help='PSI of Constraining, (0.01 default)', required=False)
    PARSER.add_argument("--criteria", type=float, default=criteria, help='stop condition (1e-6 default)',
                        required=False)
    PARSER.add_argument("--alpha", type=float, default=alpha, help='alpha of LBG funtion (1e-4 default)',
                        required=False)
    PARSER.add_argument("--gaussianizer", type=bool, default=False,
                        help='if you want to use Gaussian cumulative distribution, type 1', required=False)
    PARSER.add_argument("--component", type=int, default=number_of_component,
                        help="number of component for GMM (default 64)",
                        required=False)
    ARGS, UNKNOWN = PARSER.parse_known_args()

    if ARGS.test == True:
        evaluation_or_validation = 'evaluation'

    # Data loading
    if ARGS.data_file is not None:
        data_file = ARGS.data_file
    data, labels = loadData(data_file)

    # Gaussian cumulative distribution function and z-score
    if (ARGS.gaussianizer is True) & (ARGS.test is False):
        data = z_score(data)
        data = gaussianizer(data)

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

    # Training
    if ARGS.test is False & ARGS.c_exp == False:
        # Spritting data to train and validation
        (train_data, train_labels), (validation_data, validation_labels) = split_4to1(data, labels)

        # predictions, log joint likelihoods and likelihood ratio
        predictions, logSjoint, binary_likelihood_ratio = GMM_classifier(train_data, train_labels, validation_data,
                                                                         tied=ARGS.tied, naive=ARGS.naive,
                                                                         C_number=ARGS.component)
        # plane accuracy calculator to follow the procedure
        accuracy = (predictions == validation_labels).sum() / predictions.size
        err_rate = (1 - accuracy) * 100
        print(f"tied_{ARGS.tied} naive_{ARGS.naive} error rate: {err_rate} on validation data")

        # Saving likelihoods to future experiment
        if (ARGS.save == True):
            np.save(
                f'projects/gender_experiments/validations/GMM_classifier_4to1_likelihoods/naive_{ARGS.naive}_tied_{ARGS.tied}_component_{ARGS.component}_gaussianizer_{ARGS.gaussianizer}_llr.npy',
                binary_likelihood_ratio)
            np.save(f'projects/gender_experiments/validations/GMM_classifier_4to1_likelihoods/labels.npy',
                    validation_labels)

    # component experiment with minDFCs, given component number it calculates the minDCF and save it
    if ARGS.c_exp == True:
        minDCFs_raw = []
        minDCFs_gau = []
        if (ARGS.test is True):
            test_data, test_labels = loadData(test_file)
        for type in ['raw', 'gaussianized']:

            if type == 'gaussianized':
                if (ARGS.test is True):
                    test_data = z_score_test(data, test_data)
                    data = z_score(data)
                    data, test_data = preprocess_gaussianization(data, test_data)
                else:
                    data = z_score(data)
                    data = gaussianizer(data)
            # Spritting data to train and validation
            (train_data, train_labels), (validation_data, validation_labels) = split_4to1(data, labels)

            if (ARGS.test is True):
                validation_data = test_data
                validation_labels = test_labels
                train_data = data
                train_labels = labels

            for c in list_of_components:
                predictions, logSjoint, binary_likelihood_ratio = GMM_classifier(train_data, train_labels,
                                                                                 validation_data, tied=ARGS.tied,
                                                                                 naive=ARGS.naive, C_number=c, )
                minDCF = minDCFBinary(binary_likelihood_ratio, validation_labels, 0.5, 1, 1)
                if type == 'gaussianized':
                    minDCFs_gau.append(minDCF)
                else:
                    minDCFs_raw.append(minDCF)
        # save the list that experimented
        np.save(
            f'projects/gender_experiments/{evaluation_or_validation}/GMM_classifier_component_experiment/naive_{ARGS.naive}_tied_{ARGS.tied}_gaussianizer_{ARGS.gaussianizer}_until_{max(list_of_components)}_minDCFs_gau.npy',
            minDCFs_gau)
        np.save(
            f'projects/gender_experiments/{evaluation_or_validation}/GMM_classifier_component_experiment/naive_{ARGS.naive}_tied_{ARGS.tied}_gaussianizer_{ARGS.gaussianizer}_until_{max(list_of_components)}_minDCFs_raw.npy',
            minDCFs_raw)
        np.save(
            f'projects/gender_experiments/{evaluation_or_validation}/GMM_classifier_component_experiment/naive_{ARGS.naive}_tied_{ARGS.tied}_gaussianizer_{ARGS.gaussianizer}_until_{max(list_of_components)}_list_of_components.npy',
            list_of_components)

    # Test
    if (ARGS.test is True):
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

        # accuracy of model and likelihoods
        predictions, logSjoint, llr = GMM_classifier(data, labels, test_data, tied=ARGS.tied,
                                                     naive=ARGS.naive, C_number=ARGS.component)
        accuracy = (predictions == test_labels).sum() / predictions.size
        accuracy = (1 - accuracy) * 100
        print(f"accuracy: {accuracy} on test data")

        # Saving likelihoods to future experiment
        if (ARGS.save == True):
            np.save(
                f'projects/gender_experiments/evaluation/GMM_classifier_4to1_likelihoods/naive_{ARGS.naive}_tied_{ARGS.tied}_component_{ARGS.component}_gaussianizer_{ARGS.gaussianizer}_llr.npy',
                llr)
            np.save(f'projects/gender_experiments/evaluation/GMM_classifier_4to1_likelihoods/labels.npy', test_labels)
