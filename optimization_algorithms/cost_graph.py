# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")

from optimization_algorithms.optimizer import Optimizer
from util.dynamic_graph import DynamicGraph

class OptimizerCostGraph(Optimizer):
    """
    Wraps around an optimizer, and will print out a graph of the changing cost
    while the optimizer is running.
    
    Parameters
    ---------
    
    optimizer : Optimizer
        Optimizer to run and graph the cost values over time.
        
    WARNING
    --------
    
    This utility will slow down the learning speed by a decent amount if there are a large number
    of iterations relative to iterations_per_point - will regraph all points each iterations_per_update.
    
    So updating iterations_per_point and iterations_per_update is vital to have good efficiency.
    """
    
    def __init__(self, optimizer, iterations_per_point=10, iterations_per_update=20):
        if iterations_per_point > iterations_per_update:
            raise ValueError("iterations_per_point should be smaller than iterations_per_update,"
                             " otherwise some updates would force a redraw without anything in the graph changing.")
        
        self._optimizer = optimizer
        self._iterations_per_update = iterations_per_update
        self._iterations_per_point = iterations_per_point
        
        self._num_iterations = 0
        self._costs_over_time = []
        self._time_steps = []
        self._dynamic_graph = DynamicGraph(xlabel="Number Iterations", ylabel="Total Cost")
                
    def optimize(self, X, y, initial_theta, estimator_function, cost_function):
        """
        See Optimizer class for a complete description of what this function is for.
        """
        final_theta, status =\
            self._optimizer.optimize(X, y, initial_theta, estimator_function,
                                     self._cost_function(cost_function))
        
        self._dynamic_graph.final_update(self._time_steps, self._costs_over_time)
        
        return (final_theta, status)
    
    def _cost_function(self, problem_cost_function):
        
        def update_cost_and_return(X, pred, y):
            cost, gradient= problem_cost_function(X, pred, y)
            # Check for mod then increment to ensure have a point for first run.
            if self._num_iterations % self._iterations_per_point == 0:
                
                self._costs_over_time.append(cost)
                self._time_steps.append(self._num_iterations + 1)
            
            
            if self._num_iterations % self._iterations_per_update == 0:
                self._dynamic_graph.redraw(self._time_steps, self._costs_over_time)
                
            self._num_iterations += 1
            
            return (cost, gradient)
        return update_cost_and_return
            
    def converge_hints(self):
        """
        Returns a string containing hints for the user on how to achieve
        convergence with the used optimizer.
        """
        return self._optimizer.converge_hints()