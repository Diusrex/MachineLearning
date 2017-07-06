from sklearn import datasets

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../..")

from optimization_algorithms.gradient_descent import GradientDescent
from util.data_operation import accuracy
from util.data_manipulation import train_test_split
from util.graphing import class_estimation_graph

from supervised_learning.logistic_regression import LogisticRegression
from supervised_learning.one_vs_all import OneVsAllClassification

def CreateDefaultLogisticRegression():
    return LogisticRegression(GradientDescent())

def main():
    n_classes = 4
    # Just has one feature to make it easy to graph.
    X, y = datasets.make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0,
                                        n_clusters_per_class=1, flip_y=0.1, n_classes=n_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_proportion=0.2)
    
    logistic_reg = OneVsAllClassification(CreateDefaultLogisticRegression)
    logistic_reg.fit(X_train, y_train)
    
    y_pred = logistic_reg.predict(X_test)
    acc = accuracy(y_pred, y_test)
    
    class_estimation_graph(n_classes, X_test, y_test, y_pred,
                           "Logistic Regression %.2f%% Accuracy.\nShape is true class, color is estimate" % (acc*100))

if __name__ == "__main__":
    main()
