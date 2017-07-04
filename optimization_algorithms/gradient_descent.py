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
    """
    def __init__(self, num_iterations=200, learning_rate=0.01, convergence_threshold=1e-6):
        self._num_iterations = num_iterations
        self._learning_rate = learning_rate
        self._convergence_threshold = convergence_threshold
                
    def optimize(self, X, y, initial_theta, cost_function):
        theta = initial_theta
        for _ in range(self._num_iterations):
            _, gradient = cost_function(X, theta, y)
            
            if np.sum(np.abs(gradient)) < self._convergence_threshold:
                return (theta, Optimizer.Status.CONVERGED)
            
            # Go in the opposite direction of the gradient
            theta += self._learning_rate * -gradient
        
        return (theta, Optimizer.Status.OUT_OF_ITERATIONS)
        
    def converge_hints(self):
        """
        Returns a string containing hints for the user on how to achieve
        convergence with the used optimizer.
        """
        return "Try increasing the num_iterations or increasing the learning rate."
