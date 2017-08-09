# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../..")

from util.data_generation import create_1d_categorical_feature_regression
from util.data_manipulation import train_test_split
from util.data_operation import mean_square_error
from util.graphing import display_2d_regression

from supervised_learning.decision_tree import DecisionTreeRegression

def train_and_run_dtree(decision_tree, X_train, X_test,
                        y_train, y_test, format_title, should_print_tree):
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)
    
    if should_print_tree:
        decision_tree.print_tree()
    
    mse = mean_square_error(y_pred, y_test)
    
    display_2d_regression(X_test[:, 0], X_test[:, 1], y_pred, y_test,
                          format_title.format(mse))

def main(should_print_tree=False):
    X, y = create_1d_categorical_feature_regression()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_proportion=0.2)
    
    # Can't run CART with regression when using categorical variables
    train_and_run_dtree(DecisionTreeRegression(algorithm_to_use='ID3'), X_train, X_test,
                  y_train, y_test, 'Decision Tree ID3 (MSE {:.2f})',
                  should_print_tree)

if __name__ == "__main__":
    should_print_tree = len(sys.argv) > 1
    main(should_print_tree)