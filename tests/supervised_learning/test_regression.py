import unittest
from sklearn import datasets

# Actual tested algorithms
from supervised_learning.knn import KNN_Regression
from supervised_learning.linear_regression import LinearRegression

from optimization_algorithms.gradient_descent import GradientDescent
from optimization_algorithms.util.numeric_gradient_checker import NumericGradientChecker

from util.data_operation import mean_square_error
from util.data_manipulation import train_test_split


# Initialize the data here, so all of the algorithms run on the exact same data.
def initializeDataMaps(X, y, X_map, y_map):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_proportion=0.2)
    X_map['train'] = X_train
    y_map['train'] = y_train
    X_map['test'] = X_test
    y_map['test'] = y_test

# Map from 'train' or 'test' to the data
simple_linear_X = {}
simple_linear_y = {}

# Super simple, both variables are informative and no noise
X, y = datasets.make_regression(n_samples=200, n_features=2, n_informative=2)
initializeDataMaps(X, y, simple_linear_X, simple_linear_y)


class RegressionTester(unittest.TestCase):
    def runSingleLinearRegression(self, algorithm, max_mse):
        """
        The algorithm should have been initialized with any additional checkers
        available for its learning method - like have the optimizer wrapped by
        NumericGradientChecker.
        
        Will be regressing with a single output variable per sample.
        """
        # Very simple dataset, only has 2 classes, 2 features, and no error
        X_train = simple_linear_X['train']
        y_train = simple_linear_y['train']
        
        algorithm.fit(X_train, y_train)
        
        X_test = simple_linear_X['test']
        y_test = simple_linear_y['test']
        
        # Round just in case the algorithm returns likelihoods
        y_pred = algorithm.predict(X_test)
        
        # Expect very good mse, since no noise
        self.assertGreater(max_mse, mean_square_error(y_pred, y_test))
        

class KNNRegressionTester(RegressionTester):
    
    def testSingleLinearRegression1K(self):
        algorithm = KNN_Regression(1)
        
        # Expect high amounts of error, although will very rarely be this high
        # Will sometimes reach error in the 1000s though
        self.runSingleLinearRegression(algorithm, max_mse = 2000)
    
    def testSingleLinearRegression3K(self):
        algorithm = KNN_Regression(3)
        
        # Expect high amounts of error, although will very rarely be this high
        # Will sometimes reach error in the 1000s though
        self.runSingleLinearRegression(algorithm, max_mse = 2000)


class LinearRegressionTester(RegressionTester):
    
    def testLinearRegressionOptimizer(self):
        algorithm = LinearRegression(optimizer=NumericGradientChecker(GradientDescent(learning_rate=0.1)))
        
        # Expect only some minor fp inaccuracy
        self.runSingleLinearRegression(algorithm, max_mse = 1e-8)
    
    def testLinearRegressionNormalEq(self):
        algorithm = LinearRegression()
        
        # Expect only some minor fp inaccuracy
        self.runSingleLinearRegression(algorithm, max_mse = 1e-8)