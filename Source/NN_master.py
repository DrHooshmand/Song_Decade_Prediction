#!/bin/python3
# Shahriar Hooshmand,
# CSE5523 final project,
# Neural Network Optimization,
# Ohio State University

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFECV, SelectKBest, SelectPercentile, chi2
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from scipy.interpolate import griddata
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math
import numpy as np
import time
from itertools import cycle
from tqdm import tqdm
from operator import itemgetter
import sys

sys.path.insert(0, '../')



def evaluate(y_test, y_predict):
    '''
    Script for evaluating accuracy score
    # https://scikit-learn.org/stable/modules/classes.html#classification-metrics
    :param y_test: true labels
    :param y_predict: predicted labels
    :return: score (%)
    '''

    return accuracy_score(y_test, y_predict) *100

def round_down(num):
    '''
    round down a value
    :param num: input value
    :return: rounded value
    '''
    return num - (num%10)

def tup (n_layers, n_neur):
    '''
    creates tuples based on the number of neurons within each layer, suitable for input of model NN in scikit-learn
    :param n_layers: number of layers
    :param n_neur: number of neurons
    :return:
    '''
    dum =()
    for i in range(n_layers):
        dum += (n_neur,)
    return dum


def out_layer_neuron(input_file, skip=None, mesh_max=40, maxiter=1000, output= "mesh_file.npy"):
    '''
    Stores the hyper parameter layer/neuron tuning

    :param input_file: file to read data
    :param skip: number of entries to skip
    :param mesh_max: maximum mesh grid number for number of neuron/layer search
    :param maxiter: maximum iteration of the model NN classifier
    :param output: name of the file to output the tuning analysis
    :return: output: file name is output of the routine
    '''
    # total = 515345

    # load the data
    if skip is not None:
        data = np.loadtxt(input_file, delimiter=',', skiprows=skip)
    else:
        data = np.loadtxt(input_file, delimiter=',')


    Y = data[:, 0]
    X = data[:, 1:]

    # rounding the years to the decade
    for i in range(0, len(Y)):
        Y[i] = round_down(np.abs(Y[i]) % 100)

    X, y = StandardScaler().fit_transform(X), Y     # fitting and training the model
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    acc = []    # to store the accuracy for each neuron/layer variable data point
    plt.figure()
    lin = ['--', '-', ':']
    cycol = cycle('bgrkmyc')

    # loop over the nest of neuron/layer to find the optimized combination
    for i in tqdm(range(1, mesh_max), desc="layer loop"):
        for j in tqdm(range(1, mesh_max), desc="Neuron loop"):
            mlp = MLPClassifier(random_state=5, hidden_layer_sizes=tup(i, j), max_iter=maxiter)
            mlp.fit(X_train, y_train)
            predictions = mlp.predict(X_test)
            ev = evaluate(y_test, predictions)
            # print(i, j, ev)
            acc.append((i, j, ev))

    A = np.array(acc)
    np.save(output, A)      #store the result of analysis in a separate file
    print("Analysis is stored in:".format(output))

    return output

def plot_layer_neuron(input_file, mesh_max=40, output= "neuron_layers_contour.pdf"):

    '''
    Generates the controur plot of hyperparameters
    :param input_file: file to read data
    :param mesh_max: maximum mesh grid number for number of neuron/layer search
    :param output: name of the file to output the tuning process
    :return: (optimized #of layers, optimized #of neurons)
    '''


    print("Plotting layer/neuron analysis is initiated")
    data = np.load(input_file)  #load the data

    # grid the search domain
    grid_x, grid_y = np.meshgrid(np.linspace(0, mesh_max-1, 100), np.linspace(0, mesh_max-1, 100))
    grid_z = griddata(data[:, 0:2], data[:, 2], (grid_x, grid_y), method='nearest')
    z_min, z_max = grid_z.min(), grid_z.max()

    #plotting the results
    plt.figure()
    cp = plt.contourf(grid_x, grid_y, grid_z, cmap="jet")
    cbar = plt.colorbar(cp)
    cbar.ax.set_ylabel('Accuracy %', rotation=90)
    plt.title('Layers/Neuron Tuning ')
    plt.xlabel('# Layers')
    plt.ylabel('# Neurons')
    plt.savefig(output)

    #finding the maximum accuracy set
    indx = np.argmax(data, axis=0)[2]
    layer_opt , neuron_opt = data[indx][0], data[indx][1]

    print("Plotting layer/neuron analysis is finished")
    print("Optimized parameters: Layer = {}, Neuron = {}".format(layer_opt, neuron_opt))


    return int(layer_opt), int(neuron_opt)


def plot_hyper_param(input_file, layer, neuron, skip = None, maxiter=1000, output="hyper_test.pdf"):

    '''
    Test analysis on learning rate and momentum hyperparameters

    :param input_file: file to read data
    :param layer: number of layers
    :param neuron: number of neurons
    :param skip: if not None, number of lines to skip in the input file
    :param maxiter: maximum iteration of the model NN classifier
    :param output: output file name
    :return: (optimal momentum, optimal learning rate)
    '''

    # total = 515345
    print("Hyper parameter optimzation routine is initiated")

    # reading the data from file
    if skip is not None:
        data = np.loadtxt(input_file, delimiter=',', skiprows=skip)
    else:
        data = np.loadtxt(input_file, delimiter=',')

    Y = data[:, 0]
    X = data[:, 1:]

    # transforming the years to decade
    for i in range(0, len(Y)):
        Y[i] = round_down(np.abs(Y[i]) % 100)

    #fitting the model
    X, y = StandardScaler().fit_transform(X), Y
    X_train, X_test, y_train, y_test = train_test_split(X, y)   #spliting training and testing values

    #range of search for momentum and learning rate
    eta = list(np.arange(0.05, 0.5, 0.05))
    alp = [0, 0.001, 0.01, 0.5, 0.9]

    #plotting the results
    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    lin = ['--', '-', ':']
    cycol = cycle('bgrkmyc')

    acc = []  # storing accuracy

    #loop over hyperparameters and on-the-fly plotting
    for al in tqdm(alp, desc="alpha loop"):
        for et in tqdm(eta, desc="eta loop"):
            cl = next(cycol)
            mlp = MLPClassifier(random_state=5, hidden_layer_sizes=tup(layer, neuron), alpha=al, learning_rate_init=et,
                                max_iter=maxiter)

            mlp.fit(X_train, y_train)
            predictions = mlp.predict(X_test)
            evaluate(predictions, y_test)
            acc.append((al,et, evaluate(y_test, predictions)))
            # acc.append(mlp.n_iter_)
        plt.plot(eta, [k for (i,j,k) in acc if i==al], label=r"$\alpha = $" + str(al), color=cl)
        plt.scatter(eta, [k for (i,j,k) in acc if i==al], color=cl)
    sh=[k for (i, j, k) in acc if i == al]
    print(sh)
    plt.xlabel(r"$\eta$")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(output)

    #finding the optimum hyperparameter values based on the maximum accuracy score
    m_max = max(acc, key=itemgetter(2))[2]
    indx = [acc.index(tupl) for tupl in acc if tupl[2] == m_max]
    indx = indx[0]
    alpha_opt, eta_opt = acc[indx][0], acc[indx][1]

    print("Hyper parameter optimzation routine is finished")
    print("Optimized parameters: Alpha = {}, eta = {}".format(alpha_opt, eta_opt))


    return alpha_opt, eta_opt

def do_predict(input_file, et, alp, layer, neuron, skip=None):


    '''
    Model predictions on actual dataset after parameter optimization
    :param input_file:
    :param et: learning rate
    :param alp: momentum
    :param layer: number of layers
    :param neuron: number of neurons
    :param skip: if not None, skip these numbers of lines in data file
    :return: accuracy score
    '''

    print("Prediction is initiated")

    #read the file
    if skip is not None:
        data = np.loadtxt(input_file, delimiter=',', skiprows=skip)
    else:
        data = np.loadtxt(input_file, delimiter=',')


    # pre-process the data
    Y = data[:, 0]
    X = data[:, 1:]

    # transforming the years to decade
    for i in range(0, len(Y)):
        Y[i] = round_down(np.abs(Y[i]) % 100)

    # fit the model
    X, y = StandardScaler().fit_transform(X), Y
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    mlp = MLPClassifier(random_state=5, hidden_layer_sizes=tup(layer, neuron), alpha=alp, learning_rate_init=et,
                        max_iter=10000)

    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)    # prediction tags
    acc = evaluate(y_pred, y_test)  # evaluate the accuracy
    print(classification_report(y_test, y_pred))    # print classification table


    print("Prediction is finished")
    print("Accuracy predicted as: {:.3f}%".format(acc))
    print('Accuracy score: %.3f' % accuracy_score(y_test, y_pred))

    return acc

def main (inp, sk=0):
    '''
    Main routine
    :param inp: input file
    :param sk: number of lines to skip if not zero
    :return:
    '''

    # start writing the log file
    old_stdout = sys.stdout
    log_file = open("NN.log", "w")
    sys.stdout = log_file
    print ("Outputs are to NN.log")

    start_time = time.time()        #set the time
    # out_layer = out_layer_neuron(inp, skip=sk) # Computationally expensive: Should be ran on supercomputer
    out_layer = "ss.npy" # Here we use pre-written data on the large dataset to skip the computationally intensive part
    layer_opt, neuron_opt = plot_layer_neuron(out_layer)    # plotting layer/neuron optimiztion
    alpha_opt, eta_opt = plot_hyper_param(inp, layer_opt, neuron_opt, skip=sk)  # plotting hyperparameter tuning optimization

    acc = do_predict(inp,eta_opt, alpha_opt, layer_opt, neuron_opt, skip=sk)    # Prediction process


    print('\nRunning time: %d min' % int((time.time() - start_time) / 60))  # print processing time

    # close the log file
    sys.stdout = old_stdout
    log_file.close()

if __name__ == '__main__':

    # total = 515344
    # main ("YearPredictionMSD.txt", 514829)
    # main ("reduced.txt")

    main(str(sys.argv[1]), int(sys.argv[2]))

    print("done")
