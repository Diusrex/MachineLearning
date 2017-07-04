import numpy as np

from sklearn import datasets

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../..")

from optimization_algorithms.util.numeric_gradient_checker import NumericGradientChecker
from supervised_learning.linear_regression import LinearRegression
from util.data_operation import mean_square_error



def estimator_function(X, theta):
    """
    Simple linear regression.
    """
    return np.dot(X, theta)

def correct_cost_function(X, pred, y):
    """
    Correctly computes cost and gradient for MSE
    """
    cost = mean_square_error(pred, y) / 2
    m = len(y)
    gradient = 1/m * np.dot(X.T, pred - y)
    return (cost, gradient)

def incorrect_cost_function_v1(X, pred, y):
    """
    Incorrecly computes cost and gradient for MSE, forgets to subtract y from
    the predicted when calculating the gradient.
    """
    cost = mean_square_error(pred, y) / 2
    m = len(y)
    gradient = 1/m * np.dot(X.T, pred)
    return (cost, gradient)

def incorrect_cost_function_v2(X, pred, y):
    """
    Incorrecly computes cost and gradient for MSE, returns the negative
    value of gradient, which would cause the algorithm to increase the error.
    """
    cost = mean_square_error(pred, y) / 2
    m = len(y)
    gradient = -1/m * np.dot(X.T, pred - y)
    return (cost, gradient)

def incorrect_cost_function_v3(X, pred, y):
    """
    Incorrecly computes cost and gradient for MSE, forgets to divide the MSE
    by 2.
    """
    cost = mean_square_error(pred, y)
    m = len(y)
    gradient = 1/m * np.dot(X.T, pred - y)
    return (cost, gradient)

def incorrect_cost_function_v4(X, pred, y):
    """
    Incorrectly computes the cost function with a major increase in cost
    """
    cost = mean_square_error(pred, y) / 2 + 100000
    m = len(y)
    gradient = 1/m * np.dot(X.T, pred - y)
    return (cost, gradient)

def incorrect_cost_function_v5(X, pred, y):
    """
    Incorrectly computes the cost function, has a minor decrease in gradient
    """
    cost = mean_square_error(pred, y) / 2
    m = len(y)
    gradient = 1/m * np.dot(X.T, pred - y)
    gradient[0] -= 1e-5
    return (cost, gradient)


if __name__ == '__main__':
    # Generate some data to run on
    X, y = datasets.make_regression(n_samples=2000, n_features=4, noise=4)
    
    # Don't want it printing out the gradients,since don't care about their values, just
    # if they were different
    gradient_checker = NumericGradientChecker(print_out_diff_gradient=False)
    
    # Example with a learning algorithm
    lin_reg = LinearRegression(gradient_checker)
    lin_reg.fit(X, y)
    
    # Will check the function cost function, and ensure it is correct.
    theta = np.zeros(X.shape[1])
    gradient_checker.optimize(X, y, theta, estimator_function, correct_cost_function)
    
    try:
        gradient_checker.optimize(X, y, theta, estimator_function, incorrect_cost_function_v1)
    except ValueError as e:
        print("Error for incorrect_cost_function_v1:", e)
    
    try:
        gradient_checker.optimize(X, y, theta, estimator_function, incorrect_cost_function_v2)
    except ValueError as e:
        print("Error for incorrect_cost_function_v2:", e)
    
    try:
        gradient_checker.optimize(X, y, theta, estimator_function, incorrect_cost_function_v3)
    except ValueError as e:
        print("Error for incorrect_cost_function_v3:", e)
    
    # Note: This will not be detected, because a constant different in cost function will be cancelled out.
    try:
        gradient_checker.optimize(X, y, theta, estimator_function, incorrect_cost_function_v4)
    except ValueError as e:
        print("Error for incorrect_cost_function_v4:", e)
    
    try:
        gradient_checker.optimize(X, y, theta, estimator_function, incorrect_cost_function_v5)
    except ValueError as e:
        print("Error for incorrect_cost_function_v5:", e)