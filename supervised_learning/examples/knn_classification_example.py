from sklearn import datasets

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")

from optimization_algorithms.gradient_descent import GradientDescent
from util.data_operation import mean_square_error
from util.data_manipulation import train_test_split
from util.graphing import class_estimation_graph

from supervised_learning.knn import KNN_Classification

if __name__ == "__main__":
    n_classes = 4
    # Just has one feature to make it easy to graph.
    X, y = datasets.make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0,
                                        n_clusters_per_class=1, flip_y=0.1, n_classes=n_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_proportion=0.2)
    
    k=4
    classifier = KNN_Classification(k=k)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    mse = mean_square_error(y_pred, y_test)
    
    class_estimation_graph(n_classes, X_test, y_test, y_pred,
                           "KNN (k=%d) %.2f MSE.\nShape is true class, color is estimate" % (k, mse))