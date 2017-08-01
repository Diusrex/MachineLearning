# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../..")

from util.data_generation import create_2d_categorical_feature_two_class
from util.data_manipulation import train_test_split
from util.data_operation import accuracy
from util.graphing import class_estimation_graph

from supervised_learning.decision_tree import DecisionTree

def main(print_tree=False):
    X, y = create_2d_categorical_feature_two_class()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_proportion=0.2)
    
    decision_tree = DecisionTree()
    
    decision_tree.fit(X_train, y_train)
    # Calculate accurracy on test set
    y_pred = decision_tree.predict(X_test)
    acc = accuracy(y_pred, y_test)
    
    if print_tree:
        decision_tree.print_tree()
    class_estimation_graph(2, X_test, y_test, y_pred, "Decision Tree categorical classification (Accuracy of %.1f%%)" % (100 *acc))
    
if __name__ == "__main__":
    main(print_tree=True)