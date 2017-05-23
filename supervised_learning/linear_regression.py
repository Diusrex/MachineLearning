import random
import numpy as np
from numpy import dot
# pinv works with singular matricies
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from sklearn import datasets

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")

from util.data_operation import mean_square_error
from util.data_manipulation import train_test_split

class LinearRegression(object):
    """
    Standard Least Squares Linear predictor which can use least squares or
    gradient descent to fit provided data.
    
    Will use all data as is, without any transformations.
    
    Parameters
    --------
    graident_descent_options : GradientDescentOptions
        Optional parameter. If provided, will calculate the weights with
        gradient descent using the options.
    
    Theory
    --------
        - Highly dependent on output being predicted by a linear combination of \
        provided feature set (which may not be a linear combination of original \
        feature set).
        - Has low variance
        - Has high bias
        - To reduce overfitting it may be helpful to prune the feature set.
    
    Comments
    --------
        - This just uses standard gradient descent which is slow for very large \
        input sets. In those cases, a better option would probably be stochastic \
        gradient descent.
        
    """
    class GradientDescentOptions(object):
        """
        Makes it easier to specify the different gradient descent options.
        If provided to LinearRegression, then gradient descent will be used.
        
        Parameters
        ---------
        
        num_iterations : integer
            Specifies the number of iterations of gradient descent to be performed.
        
        learning_rate : numeric
            Determines what speed the gradient descent will update the weights.
            A too high or too low value may cause the gradient descent to not
            converge.
        
        """
        def __init__(self, num_iterations=20, learning_rate=0.001):
            self._num_iterations = num_iterations
            self._learning_rate = learning_rate
    
    def __init__(self, graident_descent_options=None, print_out=False):
        self._gradient_descent_options = graident_descent_options
        self._print_out = print_out
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
            
        y : array-like, shape [n_samples,]
            Input array of expected results. Can be 2 dimensional, if estimating
            multiple different values for each sample.
        """
        
        # Add bias columns as first column
        X = np.insert(X, 0, 1, axis=1)
        
        if self._gradient_descent_options == None:
            # Compute using normal equation
            X_tran = X.T
            # inverse(X^T * X) * X^T * y
            self._weights = dot(pinv(dot(X_tran, X)), dot(X_tran, y))
                
        else:
            num_features = np.shape(X)[1]
            self._weights = np.zeros((num_features, ))
            for _ in range(self._gradient_descent_options._num_iterations):
                # XT * (y_estimate - y)
                w_delta = -dot(X.T, dot(X, self._weights) - y)
                
                self._weights += self._gradient_descent_options._learning_rate * w_delta
        
        self._intercept = self._weights[0]
        self._coeff = self._weights[1:]
        
    def predict(self, X):
        """
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


if __name__ == "__main__":
    # Just has one feature to make it easy to graph.
    X, y = datasets.make_regression(n_samples=200, n_features=1,
                                    bias=random.uniform(-10, 10), noise=5)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_proportion=0.2)
    
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    y_pred = linear_reg.predict(X_test)
    mse = mean_square_error(y_pred, y_test)
    
    plt.scatter(X_test, y_test, color="Black", label="Actual")
    plt.plot(X_test, y_pred, label="Estimate")
    plt.legend(loc='lower right', fontsize=8)
    plt.title("Linear Regression %.2f MSE)" % (mse))
    plt.show()