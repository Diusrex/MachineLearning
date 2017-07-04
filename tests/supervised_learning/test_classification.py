import unittest
import numpy as np
from sklearn import datasets

# Actual tested algorithms
from supervised_learning.knn import KNN_Classification
from supervised_learning.logistic_regression import LogisticRegression
from supervised_learning.one_vs_all import OneVsAllClassification

from optimization_algorithms.gradient_descent import GradientDescent
from optimization_algorithms.util.numeric_gradient_checker import NumericGradientChecker

from util.data_operation import accuracy
from util.data_manipulation import train_test_split


# Initialize the data here, so all of the algorithms run on the exact same data.
def initializeDataMaps(X, y, X_map, y_map):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_proportion=0.2)
    X_map['train'] = X_train
    y_map['train'] = y_train
    X_map['test'] = X_test
    y_map['test'] = y_test

# Map from 'train' or 'test' to the data
binary_class_X = {}
binary_class_y = {}

# Super simple, both variables are informative
X, y = datasets.make_classification(
        n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, n_classes=2)

initializeDataMaps(X, y, binary_class_X, binary_class_y)

# Map from 'train' or 'test' to the data
multi_class_X = {}
multi_class_y = {}

# Super simple, both variables are informative
X, y = datasets.make_classification(
        n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, n_classes=4)

initializeDataMaps(X, y, multi_class_X, multi_class_y)

class ClassificationTester(unittest.TestCase):
    def runBinaryClassification(self, algorithm, expected_accuracy=0.95):
        """
        The algorithm should have been initialized with any additional checkers
        available for its learning method - like have the optimizer wrapped by
        NumericGradientChecker.
        
        Will be predicting for a single class - is it that class or not?
        """
        # Very simple dataset, only has 2 classes, 2 features, and no error
        X_train = binary_class_X['train']
        y_train = binary_class_y['train']
        
        algorithm.fit(X_train, y_train)
        
        X_test = binary_class_X['test']
        y_test = binary_class_y['test']
        
        # Round just in case the algorithm returns likelihoods
        y_pred = np.round(algorithm.predict(X_test))
        
        # Expect high accuracy due to no noise, simple data set, large # of samples
        self.assertGreater(accuracy(y_pred, y_test), expected_accuracy)
    
    def runMultiClassClassification(self, algorithm, expected_accuracy=0.90):
        """
        The algorithm should have been initialized with any additional checkers
        available for its learning method - like have the optimizer wrapped by
        NumericGradientChecker.
        
        The algorithm MUST be able to handle multi class inputs
        
        Will be predicting for a single class - is it that class or not?
        """
        # Very simple dataset, only has 4 classes, 2 features, and no error
        X_train = multi_class_X['train']
        y_train = multi_class_y['train']
        
        algorithm.fit(X_train, y_train)
        
        X_test = multi_class_X['test']
        y_test = multi_class_y['test']
        
        y_pred = algorithm.predict(X_test)
        
        # Expect high accuracy due to no noise, simple data set, large # of samples
        self.assertGreater(accuracy(y_pred, y_test), expected_accuracy)

class KNNClassifierTester(ClassificationTester):
    def testBinaryClassification1K(self):
        algorithm = KNN_Classification(1)
        
        # Will normally perform better than 0.7 (normally ~0.9), but this reduces the # of false failures
        self.runBinaryClassification(algorithm, expected_accuracy = 0.7)
    
    def testBinaryClassification3K(self):
        algorithm = KNN_Classification(3)
        
        # Will normally perform better than 0.8 (normally ~0.9), but this reduces the # of false failures
        self.runBinaryClassification(algorithm, expected_accuracy = 0.8)
    
    def testMultiClassClassification1K(self):
        algorithm = KNN_Classification(1)
        
        # Will normally perform better than 0.7 (normally ~0.9), but this reduces the # of false failures
        self.runMultiClassClassification(algorithm, expected_accuracy = 0.7)
    
    def testMultiClassClassification3K(self):
        algorithm = KNN_Classification(3)
        
        # Will normally perform better than 0.8 (normally ~0.9), but this reduces the # of false failures
        self.runMultiClassClassification(algorithm, expected_accuracy = 0.8)

class LogisticRegressionTester(ClassificationTester):
    def testBinaryClassification(self):
        algorithm = self._createLogisticRegression()
        
        # Will normally perform better than 0.8 (normally ~0.9), but this reduces the # of false failures
        self.runBinaryClassification(algorithm, expected_accuracy = 0.8)
    
    def testMultiClassClassificationUsingOneVsAll(self):
        one_vs_all_wrapper = OneVsAllClassification(self._createLogisticRegression)
        
        # Will normally perform better than 0.8 (normally ~0.9), but this reduces the # of false failures
        self.runMultiClassClassification(one_vs_all_wrapper, expected_accuracy = 0.75)
    
    def _createLogisticRegression(self):
        return LogisticRegression(NumericGradientChecker(GradientDescent()))
