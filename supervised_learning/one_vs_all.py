import numpy as np
from sklearn import datasets

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")

from supervised_learning.logistic_regression import LogisticRegression
from optimization_algorithms.gradient_descent import GradientDescent
from util.data_operation import mean_square_error
from util.data_manipulation import train_test_split
from util.graphing import class_estimation_graph

class OneVsAllClassification(object):
    """
    Give a classifier that is ONLY able to predict how likely a binary classification is,
    predict which of N classes each element is in.
    
    Good example is the LogisticRegression classifier.
    
    Parameters
    --------
    classifier_constructor
        Function that, when called, returns a new instance of a learning algorithm
        with fit, predict, and copy functions.
    
    provide_likelihood : boolean
        In predict function, should the likelihood an example is part of the selected class
        be returned.
    
    Theory
    --------
    Will predict the likelihood for the N different classes, and select the class
    with the highest likelihood according to the classifier.
    
    Warning
    --------
    Do NOT use on classifiers like Neural Networks/Decision Trees which are
    natively able to handle multiple different classifications at once.
    """
    def __init__(self, classifier_constructor, provide_likelihood=False):
        self._classifier_constructor = classifier_constructor
        self._label_classifiers = None
        self._label_values = None
        self._provide_likelihood = provide_likelihood
        
    
    def fit(self, X, y):
        unique_classes = np.unique(y)
        self._label_classifiers = {sample_class: self._classifier_constructor()\
                                   for sample_class in unique_classes}
        
        for sample_class in self._label_classifiers:
            classifier = self._label_classifiers[sample_class]
            has_class = (y == sample_class)
            
            classifier.fit(X, has_class)
    
    def predict(self, X):
        """
        Will return the most likely class for each row in X.
        
        X must have the same size for n_features as the input this object was
        trained on.
        
        Parameters
        ---------
        
        X : array-like, shape [n_samples, n_features]
            Input array of features.
            
        Returns
        ---------
        Most likely class for each instance.
        
        If provide_likelihood was true, will also return the likelihood for each
        input row being part of its assigned class.
        """
        best_likelihood = np.zeros(X.shape[0])
        best_prediction = np.zeros(X.shape[0])
        for sample_class in self._label_classifiers:
            classifier = self._label_classifiers[sample_class]
            prob = classifier.predict(X)
            
            is_best = (prob > best_likelihood)
            best_likelihood[is_best] = prob[is_best]
            best_prediction[is_best] = sample_class
        
        if self._provide_likelihood:
            return (best_prediction, best_likelihood)
        
        return best_prediction
    
def CreateDefaultLogisticRegression():
    return LogisticRegression(GradientDescent())

if __name__ == "__main__":
    n_classes = 4
    # Just has one feature to make it easy to graph.
    X, y = datasets.make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0,
                                        n_clusters_per_class=1, flip_y=0.1, n_classes=n_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_proportion=0.2)
    
    logistic_reg = OneVsAllClassification(CreateDefaultLogisticRegression)
    logistic_reg.fit(X_train, y_train)
    
    y_pred = logistic_reg.predict(X_test)
    mse = mean_square_error(y_pred, y_test)
    
    class_estimation_graph(n_classes, X_test, y_test, y_pred,
                           "Logistic Regression %.2f MSE.\nShape is true class, color is estimate" % (mse))