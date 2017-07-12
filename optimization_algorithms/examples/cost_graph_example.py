from sklearn import datasets

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../..")

from optimization_algorithms.util.cost_graph import OptimizerCostGraph
from optimization_algorithms.gradient_descent import GradientDescent
from util.data_manipulation import train_test_split

from supervised_learning.logistic_regression import LogisticRegression

# This example uses OptimizerCostGraph on GradientDescent to plot the error over time

# Have the default be very small, but if file is ran as main, will run for much longer
def main(num_iterations=200, iterations_per_update=20):
    # Just has one feature to make it easy to graph.
    X, y = datasets.make_classification(n_samples=200, n_features=1, n_informative=1, n_redundant=0,
                                        n_clusters_per_class=1, flip_y=0.1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_proportion=0.2)
    
    logistic_reg = LogisticRegression(
            optimizer=OptimizerCostGraph(
                    GradientDescent(num_iterations=num_iterations),
                    iterations_per_update=iterations_per_update))
    logistic_reg.fit(X_train, y_train)
    

if __name__ == "__main__":
    main(num_iterations=20000, iterations_per_update=500)
