import matplotlib.pyplot as plt

from sklearn import datasets

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")

from util.data_operation import mean_square_error
from util.data_manipulation import train_test_split

from supervised_learning.knn import KNN_Regression

if __name__ == "__main__":
    # Just using one feature to make it graphable
    X, y = datasets.make_regression(n_samples=200, n_features=1, bias=150, noise=4)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_proportion=0.2)
    
    reg = KNN_Regression(k=4)
    
    reg.fit(X_train, y_train)
    
    y_pred = reg.predict(X_test)
    
    mse = mean_square_error(y_pred, y_test)
    
    plt.scatter(X_test, y_test, color="Black", label="Actual")
    plt.scatter(X_test, y_pred, color="Red", label="Prediction")
    plt.legend(loc='lower right', fontsize=8)
    plt.title("KNN Regression (%.2f MSE)" % mse)
    plt.show()
