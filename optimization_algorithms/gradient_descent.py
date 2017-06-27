import numpy as np

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")

from optimization_algorithms.optimizer import Optimizer

class GradientDescent(Optimizer):
    """
    Optimizes a class using the gradient of the function
    
    Parameters
    ---------
    
    num_iterations : integer
        Specifies the number of iterations of gradient descent to be performed.
    
    learning_rate : numeric
        Determines what speed the gradient descent will update the weights.
        A too high or too low value may cause the gradient descent to not
        converge.
    
    convergence_graident : numeric
        Minimum gradient magnitude for the optimization to not have yet converged.
    
    debug_gradient : boolean
       Should the gradient be checked every iteration against the numeric gradient.
       If the different is > 1e-6, will raise a ValueError.
       Good way to test the cost_function implementation, to ensure it is correct.
       
       WARNING: Heavy performance penalty, should only be used to test algorithms
       after initially implemented
    """
    def __init__(self, num_iterations=200, learning_rate=0.01, convergence_threshold=1e-6,
                 debug_gradient=False):
        self._num_iterations = num_iterations
        self._learning_rate = learning_rate
        self._convergence_threshold = convergence_threshold
        self._debug_gradient = debug_gradient
                
    def optimize(self, X, y, initial_theta, estimator_function, cost_function):
        theta = initial_theta
        for _ in range(self._num_iterations):
            estimates = estimator_function(X, theta)
            _, gradient = cost_function(X, estimates, y)
            
            if (self._debug_gradient):
                self._check_gradient_results(gradient, X, y, theta, estimator_function, cost_function)
            
            if sum(abs(gradient)) < self._convergence_threshold:
                return (theta, Optimizer.Status.CONVERGED)
            
            # Go in the opposite direction of the gradient
            theta += self._learning_rate * -gradient
        
        return (theta, Optimizer.Status.OUT_OF_ITERATIONS)
    
    
    def _check_gradient_results(self, gradient, X, y, theta, estimator_function, cost_function):
        """
        Will check the gradient, by numerically calculating the gradient using
        the estimator function
        
        Approximates the gradient for element i as being ~ (cost(theta + eps) - cost(theta -eps)) / 2eps,
        where theta at i is the only element increased by eps.4
        """
        num_features = gradient.shape[0]
        numerical_gradient = np.zeros((num_features,))
        
        eps = 0.0001
        # Relatively large difference, normally will be < 1e-8
        acceptable_diff = 1e-6
        
        nudge = np.zeros((num_features,))
        for feature in range(num_features):
            nudge[feature] = eps
            above, _ = cost_function(X, estimator_function(X, theta + nudge), y)
            below, _ = cost_function(X, estimator_function(X, theta - nudge), y)
            
            numerical_gradient[feature] = (above - below) / (2 * eps)
            
            nudge[feature] = 0
        
        total_diff = sum(abs(numerical_gradient - gradient))
        if total_diff > acceptable_diff:
            print("Generated gradient:", gradient)
            print("Numerical gradient:", numerical_gradient)
            
            raise ValueError(
                    "Gradient and numerical gradient differ by %f (expect < 1e-6), bad implementation?" %
                    (total_diff))
        
    def converge_hints(self):
        """
        Returns a string containing hints for the user on how to achieve
        convergence with the used optimizer.
        """
        return "Try increasing the num_iterations or increasing the learning rate."