import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")

from optimization_algorithms.gradient_descent import GradientDescent
from util.data_operation import mean_square_error
from util.data_manipulation import train_test_split

from supervised_learning.logistic_regression import LogisticRegression

if __name__ == "__main__":
    # Just has one feature to make it easy to graph.
    X, y = datasets.make_classification(n_samples=200, n_features=1, n_informative=1, n_redundant=0,
                                        n_clusters_per_class=1, flip_y=0.1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_proportion=0.2)
    
    logistic_reg = LogisticRegression(optimizer=GradientDescent(num_iterations=20000))
    logistic_reg.fit(X_train, y_train)
    
    y_pred = logistic_reg.predict(X_test)
    mse = mean_square_error(y_pred, y_test)
    
    y_pred_rounded = np.round(y_pred)
    mse_rounded = mean_square_error(y_pred_rounded, y_test)
    
    plt.figure()
    plt.scatter(X_test, y_test, color="Black", label="Actual")
    plt.scatter(X_test, y_pred, color="Red", label="Prediction")
    plt.scatter(X_test, y_pred_rounded, color="Blue", label="Rounded Prediction")
    plt.legend(loc='center right', fontsize=8)
    plt.title("Logistic Regression %.2f MSE, %.2f MSE with Rounding)" % (mse, mse_rounded))
    plt.show()
