## Instruction of the files
 * All the classifiers have its own python file 
 * Experiment file for repetaing the saved experiences commands will be mention later
 * all the preparation functions are in data_preparation_PCA_LDA file
 * splitting functions in splitData function
 * all the plot functions are in plottingFunctions
 * evaluation functions such as cost detection, roc, bayes error plot e.g. in the evaluation_functions file


## How to produce new experiments and see them
All the classifier function has own help guide. you can write 

    python project/classifier_file.py --help 

For specific experiment you should specifiy classication

    python project/experiments.py --classification name_of_classifier

The classifier name can be: 'MVG' for multivariate gaussian classifier, 'LR' logistic regression
'SVM' for support vector machine, 'GMM' for gaussian mixture model

    --test 1 #for evaluation exercises 

For specific experiment you should first type:

    --classifier analysis --type type_of_analyis_for_bests

Experiment type can be ROC for roc plot, actDCF for actual vs min cost comparison, bayes_err for bayes error plot , 
optimum_th for optimum threshold  experiment. dont forget dont specify anything for validation, but for evaluation
you should type --test 1