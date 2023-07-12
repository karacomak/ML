## Instruction of the files
 * All the classifiers have its own python file.
 * Data preparation functions are in data_preparation_PCA_LDA file.
 * Split function is in split_data file.
 * Functions for visualization are in plotting_functions.
 * Functions to use in evaluation such as cost detection, ROC curve, Bayes Error plot are in the evaluation_functions file.
 * Experiment file contains experiments (such as fusion models or results with best parameters) which are required by the mentor of course.


## How to produce new experiments and see them
All the classifier function has own help guide. you can write 

    python project/<classifier_file>.py --help 

For specific experiment classification must be specified.

    python project/experiments.py --classification <name_of_classifier>

The classifier name can be: 'MVG' for multivariate gaussian classifier, 'LR' logistic regression
'SVM' for support vector machine, 'GMM' for gaussian mixture model

For the results on test set, type:

    --test 1 

For specific experiment, type:

    --classifier analysis --type <type_of_analyis_for_bests>

Experiment type can be "ROC" for ROC plot, "actDCF" for actual vs minimum cost comparison, "bayes_err" for Bayes Error plot ,  "optimum_th" for optimum threshold  experiment. There is no need to specify something for validation results, but for evaluation, --test 1 must be typed.

My contribution: All the codes have been developed as individual in personal computer. 