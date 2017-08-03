
from sklearn import datasets

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../..")

from optimization_algorithms.gradient_descent import GradientDescent
from util.data_manipulation import train_test_split
from util.graphing import decision_boundary_graph

from supervised_learning.svm import SVM, Kernel, svm_able_to_run
from supervised_learning.knn import KNN_Classification
from supervised_learning.logistic_regression import LogisticRegression

def main(num_samples=50, points_per_dimension=20):
    X, y = datasets.make_classification(n_samples=num_samples, n_features=2, n_informative=2, n_redundant=0,
                                        n_clusters_per_class=2, flip_y=0.1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_proportion=0.2)
    
    logistic_reg = LogisticRegression(optimizer=GradientDescent(num_iterations=20000))
    logistic_reg.fit(X_train, y_train)
    decision_boundary_graph(X_test, y_test, logistic_reg, "Logistic Regression",
                            points_per_dimension=points_per_dimension)
    
    if svm_able_to_run:
        logistic_reg = SVM(Kernel.linear_kernel(), C=1)
        logistic_reg.fit(X_train, y_train)
        decision_boundary_graph(X_test, y_test, logistic_reg, "SVM - Linear Kernel",
                                points_per_dimension=points_per_dimension)
        
        logistic_reg = SVM(Kernel.gaussian_kernel(sigma=2), C=1)
        logistic_reg.fit(X_train, y_train)
        decision_boundary_graph(X_test, y_test, logistic_reg, "SVM - Gaussian Kernel",
                                points_per_dimension=points_per_dimension)
    else:
        print("WARNING: cvxopt not installed, SVM will not work.")
    
    logistic_reg = KNN_Classification(k=1)
    logistic_reg.fit(X, y)
    logistic_reg2 = KNN_Classification(k=3)
    logistic_reg2.fit(X, y)
    
    decision_boundary_graph(X_test, y_test, logistic_reg, "KNN K=1",
                            points_per_dimension=points_per_dimension)
    decision_boundary_graph(X_test, y_test, logistic_reg2, "KKN K=3",
                            points_per_dimension=points_per_dimension)

if __name__ == "__main__":
    main(num_samples=200, points_per_dimension=100)