import numpy as np

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../..")

from optimization_algorithms.gradient_descent import GradientDescent
from optimization_algorithms.optimizer import Optimizer

class NumericGradientChecker(Optimizer):
    """
    Wraps around an optimizer, and will ensure that the cost calculation and
    gradient calculation for the ML algorithm are correct.
    
    Compares the returned gradient with the numeric gradient, which is approximated
    for element i in the features as being ~ (cost(theta + eps) - cost(theta -eps)) / 2eps,
    where feature i is the only element changed by eps.
    
    For an example, see numeric_gradient_checker_example in optimization_algorithms.examples
    
    Parameters
    ---------
    
    optimizer : Optimizer
        Optimizer to run and graph the cost values over time.
        If not provided, will default to GradientDescent.
    
    eps : numeric
        Value of eps used in the numeric gradient calculation, should be sufficiently small
    
    acceptable_diff : numeric
        Total absolute difference between gradient and numeric gradient that is acceptable.
        The default value is quite high for the value of eps, since most algorithms would differ
        by <= 1e-9.
    
    print_out_diff_gradient : boolean
        If the difference between gradient and numeric gradient isn't acceptable, should the
        offending gradient and numeric_gradient be printed out.
    
    WARNING
    --------
    
    This checker is quite expensive, especially the more complicated the cost_functions
    calculations are.
    
    It should ONLY be used to ensure a ML algorithms cost_function is correct.
    """
    
    def __init__(self, optimizer=None, eps=1e-3, acceptable_diff=1e-6, print_out_diff_gradient=True):
        if optimizer is None:
            optimizer = GradientDescent()
        
        self._optimizer = optimizer
        self._print_out_diff_gradient = print_out_diff_gradient
        self._eps = eps
        self._acceptable_diff = acceptable_diff
                
    def optimize(self, X, y, initial_theta, cost_function):
        """
        See Optimizer class for a complete description of what this function is for.
        """
        return self._optimizer.optimize(
                X, y, initial_theta,
                self._check_numberic_gradient(cost_function))
    
    def _check_numberic_gradient(self, cost_function):
        """
        Returns a function that wraps around the cost function for a ML algorithm.
        
        Function will check, at each step, that the numerical_gradient and
        gradient from the cost function are consistent
        """
        
        def check_numberic_gradient(X, theta, y):
            cost, gradient = cost_function(X, theta, y)
            
            num_features = gradient.shape[0]
            numerical_gradient = np.zeros((num_features,))
            
            nudge = np.zeros((num_features,))
            for feature in range(num_features):
                nudge[feature] = self._eps
                above_ret = cost_function(X, theta + nudge, y)
                below_ret = cost_function(X, theta - nudge, y)
                
                numerical_gradient[feature] = (above_ret[0] - below_ret[0]) / (2 * self._eps)
                
                nudge[feature] = 0
            
            total_diff = sum(abs(numerical_gradient - gradient))
            if total_diff > self._acceptable_diff:
                if self._print_out_diff_gradient:
                    print("Generated gradient:", gradient)
                    print("Numerical gradient:", numerical_gradient)
                
                raise ValueError(
                        "Gradient and numerical gradient differ by %f (expect < %f), bad implementation?" %
                        (total_diff, self._acceptable_diff))
            
            return (cost, gradient)
        return check_numberic_gradient
            
    def converge_hints(self):
        """
        Returns a string containing hints for the user on how to achieve
        convergence with the used optimizer.
        """
        return self._optimizer.converge_hints()