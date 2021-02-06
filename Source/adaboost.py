#!/bin/python3

from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import AdaBoostClassifier as ada
import math
import numpy as np
import time
import sys
sys.path.insert(0, '../')



def train_and_test(X_train, y_train, X_test, y_test):
    '''
    Script for performing adaboost analysis
    :param X_train: training features
    :param y_train: training tags
    :param X_test:  testing features
    :param y_test:  testing tags
    :return:
    '''

    ''' https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
        https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel
        https://stats.stackexchange.com/questions/43943/which-search-range-for-determining-svm-optimal-c-and-gamma-parameters
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV '''
    C, max_iter = [math.pow(2, i) for i in range(-4, 8)], [1e6]

    # Parameter tuning search scheme
    param_grid = [{"n_estimators": [10]}, {"n_estimators": [50]},{"n_estimators": [100]},
            {"n_estimators": [300]},{"n_estimators": [500]},{"n_estimators": [1000]}]

    ''' https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
        
        implements "one-vs-the-rest" multi-class strategy
        (preferred to "one-vs-one" because of significantly less runtime for
        similar results '''

    # Search for optimized parameters and fitting the model
    clf = GridSearchCV(ada(n_estimators=500),
                       param_grid,
                       scoring='accuracy',
                       iid=False,  # return average score across folds
                       cv=3)

    clf.fit(X_train, y_train)
    print('Best params set found on training set:\n', clf.best_params_) # Best parameters

    print('\nGrid (mean accuracy) scores on training set:\n')   # Print score
    means = clf.cv_results_['mean_test_score']
    for mean, params in zip(means, clf.cv_results_['params']):
        print("%0.3f for %r" % (mean, params))

    print('\nDetailed classification report:\n')
    y_pred = clf.predict(X_test)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
    print(classification_report(y_test, y_pred))
    # https://scikit-learn.org/stable/modules/classes.html#classification-metrics
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


def decade(year):
    '''
    script that turns the year to the decade
    :param year: input year
    :return: output decade
    '''
    return int(math.floor(year / 10) * 10)

def main(inp, sk=0, log= "Adaboost.log"):

    '''
    Main script for adaboost analyses
    :param inp: input file
    :param sk: if not None, number of lines to skip
    :param log: name of the log file to output the results
    :return:
    '''

    # Outputting the data to a log file

    old_stdout = sys.stdout
    log_file = open(log, "w")
    sys.stdout = log_file
    print(log)

    start_time = time.time()    # Set the timer on
    data = np.loadtxt(inp, delimiter=',', skiprows=sk)  # Load the data
    X = StandardScaler().fit_transform(data[:, 1:]) # Transform the data to machine readable format
    y = np.vectorize(decade)(data[:, 0])    # Vectorize the tags
    print(Counter(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    train_and_test(X_train, y_train, X_test, y_test)

    print('\nRunning time: %d min' % int((time.time() - start_time) / 60))
    
    sys.stdout = old_stdout
    log_file.close()
    
if __name__ == '__main__':

    # main("reduced.txt")
    main(str(sys.argv[1]), int(sys.argv[2]))

    print("done")
