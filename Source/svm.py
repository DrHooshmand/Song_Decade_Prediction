#!/bin/python3

from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
import math
import numpy as np
import time
import sys
sys.path.insert(0, '../')

''' other useful resources:

    https://scikit-learn.org/stable/modules/svm.html
'''


def split(X, y, test_size=0.1, random_state=None):
    '''
    Splitting data to training and validation sets
    :param X: Feature space
    :param y: Tags
    :param test_size: size of the test set
    :param random_state: if not None, random state is defined
    :return: Training features, validation features, training tags, validation tags
    '''
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    return train_test_split(
        X, y,
        test_size=test_size,
        # None defaults to RandomState instance used by np.random
        random_state=random_state,
        # stratify=True
    )


def param_tuning(X_val, y_val):
    '''
    Script for parameter tuning of SVM classifier
    :param X_val: Feature space
    :param y_val: Tags
    :return: Best parameters
    '''
    ''' https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV '''
    C, max_iter = [math.pow(2, i) for i in range(-4, 8)], [1e6]
    param_grid = [{'loss': ['squared_hinge'], 'dual': [False],
                   'C': C, 'max_iter': max_iter},
                  {'loss': ['hinge'], 'dual': [True],
                   'C': C, 'max_iter': max_iter}]

    ''' https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC

        implements "one-vs-the-rest" multi-class strategy
        (preferred to "one-vs-one" because of significantly less runtime for
        similar results) '''
    clf = GridSearchCV(svm.LinearSVC(class_weight='balanced'),
                       param_grid,
                       scoring='accuracy',
                       iid=False,  # return average score across folds
                       cv=3)

    clf.fit(X_val, y_val)
    print('Best params set found on validation set:\n', clf.best_params_)

    print('\nGrid (mean accuracy) scores on validation set:\n')
    means = clf.cv_results_['mean_test_score']
    for mean, params in zip(means, clf.cv_results_['params']):
        print("%0.3f for %r" % (mean, params))

    return clf.best_params_


def train_and_test(X_train, y_train, X_test, y_test, best_params):
    '''
    Training and testing analysis
    :param X_train: training features
    :param y_train: training tags
    :param X_test:  testing features
    :param y_test:  testing tags
    :param best_params: best tuned parameters
    :return:
    '''
    clf = svm.LinearSVC(loss=best_params['loss'],
                        dual=best_params['dual'],
                        C=best_params['C'],
                        class_weight='balanced',
                        max_iter=best_params['max_iter'])
    clf.fit(X_train, y_train)
    print('\nDetailed classification report:\n')
    y_pred = clf.predict(X_test)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
    print(classification_report(y_test, y_pred))
    # https://scikit-learn.org/stable/modules/classes.html#classification-metrics
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))


def decade(year):
    '''
    script that turns the year to the decade
    :param year: input year
    :return: output decade
    '''
    return int(math.floor(year / 10) * 10)

def main(inp, sk=0, train_portion= 0.9, log="SVM.log"):

    '''
    Main script for SVM classifier
    :param inp: input file
    :param sk: if not None, skip these many rows
    :param train_portion: proportion of the data used for training
    :param log: name of the log file to output the results
    :return:
    '''

    # Writing the outputs to a log file
    old_stdout = sys.stdout
    log_file = open(log, "w")
    sys.stdout = log_file
    print (log)

    # start the counter for analysis
    start_time = time.time()
    data = np.loadtxt(inp, delimiter=',', skiprows=sk)  # read data from file
    X = StandardScaler().fit_transform(data[:, 1:]) # vectorize and clean the data
    y = np.vectorize(decade)(data[:, 0])

    N = np.shape(data)[0]
    N_train = math.floor(N*train_portion)
    N_test = N-N_train

    X_train, X_val, y_train, y_val = split(X[:N_train, :], y[:N_train])
    X_test, y_test = X[N_test:, :], y[N_test:]

    print(Counter(y_val))
    best_params = param_tuning(X_val, y_val)    # Parameter tuning
    train_and_test(X_train, y_train, X_test, y_test, best_params)

    print('\nRunning time: %d min' % int((time.time() - start_time) / 60))

    sys.stdout = old_stdout
    log_file.close()

if __name__ == '__main__':

    # main("reduced.txt")
    main(str(sys.argv[1]), int(sys.argv[2]))

    print("done")
