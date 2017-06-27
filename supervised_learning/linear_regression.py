import numpy as np
from numpy import dot
# pinv works with singular matricies
from numpy.linalg import pinv

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")

from optimization_algorithms.optimizer import Optimizer
from util.data_operation import mean_square_error

class LinearRegression(object):
    """
    Standard Least Squares Linear predictor which can use least squares or
    gradient descent to fit provided data.
    
    Will add a bias term to X, and otherwise not alter the data at all.
    
    Parameters
    --------
    optimizer : optimization_algorithms.Optimizer
        Optional parameter. If provided, will calculate the weights using L2 cost
        function and this optimizer.
        Will otherwise calculate the weights using the normal equation.
    
    Theory
    --------
        - Highly dependent on output being predicted by a linear combination of \
        provided feature set (which may not be a linear combination of original \
        feature set).
        - Has low variance and high bias
            - More features it has, the more this shifts to high variance low bias.
            - To reduce overfitting it may be helpful to prune the feature set.
    
    Comments
    --------
        - There is a tradeoff between using an optimizer and the normal equation:
            
            - Don't need to fiddle around with learning rate/other paramaeters with normal equation
            - Normal equation is ~O(#features^3), so it can't really be used with a large (~1e4) # of features.
        
    """
    
    
    def __init__(self, optimizer=None):
        self._optimizer = optimizer
        self._weights = None
        self._intercept = None
        self._coeff = None
        
    def fit(self, X, y):
        """
        Fit internal parameters to minimize MSE on given X y dataset.
        
        Will add a bias term to X.
        
        Parameters
        ---------
        
        X : array-like, shape [n_samples, n_features]
            Input array of features.
            
        y : array-like, shape [n_samples,] or [n_samples,n_values]
            Input array of expected results. Can be 2 dimensional, if estimating
            multiple different values for each sample.
        """
        # Add bias columns as first column
        X = np.insert(X, 0, 1, axis=1)
        
        if self._optimizer is None:
            # Compute using normal equation
            X_tran = X.T
            # inverse(X^T * X) * X^T * y
            self._weights = dot(pinv(dot(X_tran, X)), dot(X_tran, y))
                
        else:
            num_features = np.shape(X)[1]
            self._weights = np.zeros((num_features, ))
            self._weights, status = self._optimizer.optimize(
                    X, y,
                    self._weights,
                    lambda X,theta : dot(X, theta),
                    LinearRegression._cost_function)
                    
            if (status != Optimizer.Status.CONVERGED):
                print("WARNING: Optimizer did not converge:", self._optimizer.converge_hints())
        
        self._intercept = self._weights[0]
        self._coeff = self._weights[1:]
        
    def predict(self, X):
        """
        Predict the value(s) associated with each row in X.
        
        X must have the same size for n_features as the input this instance was
        trained on.
        
        Parameters
        ---------
        
        X : array-like, shape [n_samples, n_features]
            Input array of features.
        """
        # Add bias columns as first column
        X = np.insert(X, 0, 1, axis=1)
        
        return dot(X, self._weights)
    
    def get_feature_params(self):
        return self._coeff

    def _cost_function(X, pred, y):
        cost = mean_square_error(pred, y) / 2
        m = len(y)
        gradient = 1/m * dot(X.T, pred - y)
        return (cost, gradient)
