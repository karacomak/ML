# Machine Learning & Pattern Recognition
The repository contains machine learning algorithms, dimension reduction techniques and necessary functions which are developed to be final project of machine learning & pattern recognition course in PoliTo. (Only NumPy library  is used in development.)

### The machine learning algorithms

**Gaussian Classifier:**

Gaussian classifiers are a generalization of the Gaussian probability distribution and can be used as the basis for sophisticated non-parametric machine learning algorithms for classification and regression.

**Logistic regression:**

Logistic regression (LR) is a data analysis technique that uses mathematics to find the relationships between two data factors. It then uses this relationship to predict the value of one of those factors based on the other. The prediction usually has a finite number of outcomes, like yes or no.

**Support vector machines:**

Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection. The advantages of support vector machines are: Effective in high dimensional spaces. Still effective in cases where number of dimensions is greater than the number of samples.

**Gaussian mixture model:**

A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.

### Dimension reduction techniques

**Principal component analysis.**

Principal component analysis, or PCA, is a statistical procedure that allows you to summarize the information content in large data tables by means of a smaller set of “summary indices” that can be more easily visualized and analyzed.

**Linear discriminant analysis:**

Linear discriminant analysis (LDA), normal discriminant analysis (NDA), or discriminant function analysis is a generalization of Fisher's linear discriminant, a method used in statistics and other fields, to find a linear combination of features that characterizes or separates two or more classes of objects or events.


## Developed Codes

 * All the classifiers have its own python file.
 * Data preparation functions are in "Data_preparation_PCA_LDA" file.
 * Split function is in "Split_data" file.
 * Functions for visualization are in "Plotting_functions".
 * Functions to use in evaluation such as cost detection, ROC curve, Bayes Error plot are in the "Evaluation_functions" file.
 * Experiment file contains experiments (such as fusion models or results with best parameters) which are required by the mentor of course.


## How to repeat the experiments.

All the classifiers and experiments has a basic guide.

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

My contribution: All the codes have been developed as individual in personal computer. Detailed information can be found in the report.