import random
import matplotlib.pyplot as plt
from sklearn import datasets

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../..")

from optimization_algorithms.gradient_descent import GradientDescent
from util.data_operation import mean_square_error
from util.data_manipulation import train_test_split

from supervised_learning.linear_regression import LinearRegression

def main():
    # Just has one feature to make it easy to graph.
    X, y = datasets.make_regression(n_samples=200, n_features=1,
                                    bias=random.uniform(-10, 10), noise=5)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_proportion=0.2)
    
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    y_pred = linear_reg.predict(X_test)
    mse = mean_square_error(y_pred, y_test)
    
    linear_reg_w_grad_desc = LinearRegression(optimizer=GradientDescent(num_iterations=2500))
    linear_reg_w_grad_desc.fit(X_train, y_train)
    y_pred_w_grad_desc = linear_reg_w_grad_desc.predict(X_test)
    mse_w_grad_desc = mean_square_error(y_pred_w_grad_desc, y_test)
    
    plt.figure()
    plt.scatter(X_test, y_test, color="Black", label="Actual")
    plt.plot(X_test, y_pred, label="Estimate")
    plt.plot(X_test, y_pred_w_grad_desc, label="Estimate using Optimizer")
    plt.legend(loc='lower right', fontsize=8)
    plt.title("Linear Regression %.2f MSE Normal Eq, %.2f MSE Gradient Descent)" % (mse, mse_w_grad_desc))
    plt.show()

if __name__ == "__main__":
    main()
