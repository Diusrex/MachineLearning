import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../..")

from util.data_operation import mean_square_error
from util.data_manipulation import train_test_split

from supervised_learning.linear_regression_tf import LinearRegressionTF


def main(_=None):
    # Just has one feature to make it easy to graph.
    X, y = datasets.make_regression(n_samples=200, n_features=1,
                                    bias=random.uniform(-10, 10), noise=5)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_proportion=0.2)
    
    linear_reg = LinearRegressionTF()
    linear_reg.fit(X_train, y_train)
    y_pred = linear_reg.predict(X_test)
    y_pred = np.squeeze(y_pred)
    mse = mean_square_error(y_pred, y_test)
    
    plt.figure()
    plt.scatter(X_test, y_test, color="Black", label="Actual")
    plt.plot(X_test, y_pred, label="Estimate")
    plt.legend(loc='lower right', fontsize=8)
    plt.title("Linear Regression %.2f MSE)" % (mse))
    plt.show()

if __name__ == "__main__":
    tf.app.run()
