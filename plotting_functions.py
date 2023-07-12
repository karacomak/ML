import json
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.colors import Normalize
import numpy as np
from matplotlib import cm
from data_preparation_PCA_LDA import *
from splitData import *
import scipy.stats
import seaborn
import pandas as pd
import seaborn as sb


def threeDPrint(dataMatrix, labels):
    dataMatrix1 = dataMatrix[:, labels == 0]
    dataMatrix2 = dataMatrix[:, labels == 1]

    fig = plt.figure()

    # syntax for 3-D projection
    ax = plt.axes(projection='3d')

    # defining axes
    x, y, z = zip(*dataMatrix1.T)
    ax.scatter(x, y, z, zdir=z, s=2, c='red', alpha=0.5)
    x, y, z = zip(*dataMatrix2.T)
    ax.scatter(x, y, z, zdir=z, s=2, c='green', alpha=0.5)

    plt.show()


def scatterPlot(dataMatrix, labelList):
    dataMatrix1 = dataMatrix[:, labelList == 0]
    dataMatrix2 = dataMatrix[:, labelList == 1]

    plt.scatter(dataMatrix1[0], dataMatrix1[1], alpha=0.5, label='Male', color='red')
    plt.scatter(dataMatrix2[0], dataMatrix2[1], alpha=0.5, label='Female', color='green')

    plt.legend()
    plt.tight_layout()
    plt.show()


def lineGraph(x, y, title, xLabel, yLabel, dataNames):
    for i in range(y.shape[0]):
        dataName = dataNames.split('-')[i]
        plt.plot(x, y[i, :], label=dataName, marker='o')
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.grid(True)
    plt.legend()
    plt.show()


def bar_chart(x, y, title, x_label, y_label):
    # creating the bar plot

    plt.bar(x, y, color='blue', width=0.4)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def bar_chart_special_for_component_analysis(x, y1, y2, title, x_label, y_label):
    # creating the bar plot
    x_place = np.array([float(i+1) for i in range(x.size)])
    x_labels = np.array(['1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024'])

    y1 = y1[::-1]
    y2 = y2[::-1]
    width = 0.40

    plt.bar(x_place - width/2, y1, color='blue', label='raw', width=0.4)
    plt.bar(x_place + width/2, y2, color='red', label='gaussianized', width=0.4)

    plt.xticks(x_place, x_labels[0 : x.size])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='lower left')
    plt.show()


def bar_chart_special_for_component_analysis_test(x, y1, y2, y3, y4, title, x_label, y_label):
    x_place = np.array([float(i + 1) for i in range(x.size)])
    x_labels = np.array(['1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024'])
    # creating the bar plot

    y1 = y1[::-1]
    y2 = y2[::-1]
    y3 = y3[::-1]
    y4 = y4[::-1]

    plt.bar(x_place - 0.3, y1, color='blue', label='raw-val', width=0.2)
    plt.bar(x_place - 0.1, y2, color='red', label='gaussianized-val', width=0.2)
    plt.bar(x_place + 0.1, y3, color='green', label='raw-eva', width=0.2)
    plt.bar(x_place + 0.3, y4, color='yellow', label='gaussianized-eva', width=0.2)

    plt.xticks(x_place, x_labels[0: x.size])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()#loc='lower left'
    plt.show()


def histogram_special_for_fature_analysis(data, labels):
    # given matrix plot shorted bar chart for each attribute:
    for i in range(data.shape[0]):
        trueClass = data[i, labels == 1]
        trueClass.sort()

        falseClass = data[i, labels == 0]
        falseClass.sort()

        plt.hist(trueClass.ravel(), bins=50, density=True, color='green', alpha=0.5, label='Female')
        plt.hist(falseClass.ravel(), bins=50, density=True, color='red', alpha=0.5, label='Male')

        plt.legend()
        plt.show()


def threeD_bar_chart(x, y, dz, x_title, y_title, z_title):
    x = [np.log(float(i)) for i in x]
    y = [np.log(float(i)) for i in y]
    dz = [float(i) for i in dz]

    style.use('ggplot')

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    z = list(np.zeros(len(dz)))

    dx = np.ones(len(x)) * 0.5
    dy = np.ones(len(y)) * 0.5

    ax1.set_xlabel(x_title)
    ax1.set_ylabel(y_title)
    ax1.set_zlabel(z_title)

    cmap = cm.get_cmap('plasma')
    norm = Normalize(vmin=min(dz), vmax=max(dz))
    colors = cmap(norm(dz))
    sc = cm.ScalarMappable(cmap=cmap, norm=norm)
    sc.set_array([])
    plt.colorbar(sc)
    ax1.bar3d(x, y, z, dx, dy, dz, shade=True, color=colors)
    plt.show()


def threedplottingold():
    with open("projects/k_hold_C_and_K_search.json", 'r') as f:
        k_hold_C_and_K_search = json.load(f)

    c_list = k_hold_C_and_K_search["c_list"]
    k_list = k_hold_C_and_K_search["k_list"]
    acc_list = k_hold_C_and_K_search["average_accuracy_list"]

    threeD_bar_chart(c_list, k_list, acc_list, 'selected C', 'selected K', 'Accuracy')


def pearson_correlartion_coefficient(data):
    corr = np.corrcoef(data)
    seaborn.heatmap(corr, cmap="Reds", vmin=-1.0)
    plt.show()
    cov_mat, means = covarianceMatrix(data)
    return np.linalg.norm((corr / cov_mat))


def plot_correlation(data, labels):
    color = ['Blues', 'Reds', 'Greens']
    for i in range(3):
        if i == 1:
            m_data = data[:, labels == 0]
            corr = np.corrcoef(m_data)
            seaborn.heatmap(corr, cmap=color[i], vmin=-1.0)
            plt.show()
        elif i == 2:
            f_data = data[:, labels == 1]
            corr = np.corrcoef(f_data)
            seaborn.heatmap(corr, cmap=color[i], vmin=-1.0)
            plt.show()
        else:
            corr = np.corrcoef(data)
            seaborn.heatmap(corr, cmap=color[i], vmin=-1.0)
            plt.show()


def best_lambda_plot(minDCFs):
    LRs = ['0.00001', '0.0001', '0.0001', '0.001', '0.01', '0.1', '1', '10', '100', '1000', '10000']
    plt.plot(LRs, minDCFs[2, :], label='π= 0.9', color='blue')
    plt.plot(LRs, minDCFs[1, :], label='π= 0.1', color='red')
    plt.plot(LRs, minDCFs[0, :], label='π= 0.5', color='green')

    plt.title('Lambda search - single fold')
    plt.xlabel('Lambdas')
    plt.ylabel('minDCFs')
    plt.grid(True)
    plt.legend()
    plt.show()


def best_lambda_plot_test(minDCFs, minDCFs_test):
    LRs = ['0.00001', '0.0001', '0.0001', '0.001', '0.01', '0.1', '1', '10', '100', '1000', '10000']
    plt.plot(LRs, minDCFs[2, :], label='π= 0.9', color='blue', linestyle='--')
    plt.plot(LRs, minDCFs[1, :], label='π= 0.1', color='red', linestyle='--')
    plt.plot(LRs, minDCFs[0, :], label='π= 0.5', color='green', linestyle='--')

    plt.plot(LRs, minDCFs_test[2, :], label='π= 0.9-eva', color='blue')
    plt.plot(LRs, minDCFs_test[1, :], label='π= 0.1-eva', color='red')
    plt.plot(LRs, minDCFs_test[0, :], label='π= 0.5-eva', color='green')

    plt.title('lambda search - single fold')
    plt.xlabel('lambda')
    plt.ylabel('minDCF')
    plt.grid(True)
    plt.legend()
    plt.show()


def best_C_plot(minDCFs, C, Korc):
    plt.plot(C, minDCFs[2, :], label='π= 0.9', color='blue')
    plt.plot(C, minDCFs[1, :], label='π= 0.1', color='red')
    plt.plot(C, minDCFs[0, :], label='π= 0.5', color='green')

    plt.title('Gamma search - single fold')
    plt.xlabel('Gamma of kernel')
    plt.ylabel('minDCF')
    plt.grid(True)
    plt.legend()
    plt.show()

def best_C_plot_test(minDCFs, minDCFs2,  C):
    plt.plot(C, minDCFs[2, :], label='π= 0.9 - eva', color='blue')
    plt.plot(C, minDCFs[1, :], label='π= 0.1 - eva', color='red')
    plt.plot(C, minDCFs[0, :], label='π= 0.5 - eva', color='green')

    plt.plot(C, minDCFs2[2, :], label='π= 0.9', color='blue', linestyle='--')
    plt.plot(C, minDCFs2[1, :], label='π= 0.1', color='red', linestyle='--')
    plt.plot(C, minDCFs2[0, :], label='π= 0.5', color='green', linestyle='--')


    plt.title('C search - single fold')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.grid(True)
    plt.legend()
    plt.show()

def dimension_reduction_exp_plot(minDCFs):
    dimensions = [str(i+1) for i in range(11)]
    plt.plot(dimensions, minDCFs[2, :], label='π= 0.9', color='blue')
    plt.plot(dimensions, minDCFs[1, :], label='π= 0.1', color='red')
    plt.plot(dimensions, minDCFs[0, :], label='π= 0.5', color='green')

    plt.title('Dimension Reduction Experiment - single fold')
    plt.xlabel('Feature Number')
    plt.ylabel('minDCF')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    data, labels = loadData("projects/Gender_Detection/Train.txt")
    test_data , test_labels = loadData('projects/Gender_Detection/Test.txt')
    # to plot gaussianized features
    #data = gaussianizer(data)
    #data, _ = preprocess_gaussianization(data, data)

    print(data.shape)
    # analysis of each feature
    #histogram_special_for_fature_analysis(data, labels)

    # correlation analysis
    #plot_correlation(data, labels)

    #scatter plot of 2 dimension data with LDA
    #data  = LDA(data, labels, 3)
    #threeDPrint(data, labels)

    # test features after gaussianization
    test_data = z_score_test(data, test_data)
    data = z_score(data)
    data, test_data = preprocess_gaussianization(data, test_data)
    # analysis of each feature
    histogram_special_for_fature_analysis(test_data, test_labels)
