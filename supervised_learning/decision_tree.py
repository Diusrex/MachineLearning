import numpy as np

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")

from util.data_operation import entropy

# TODO: Later will have categorical="none", then can be changed to "all" or
# a 1d array
# TODO: Add regularization - max depth, few observations in node, entropy change is small,
# cross-validation entropy starts to increase.
class DecisionTree(object):
    """
    Classifies given examples using the provided categorical data.
    Currently assumes only given categorical data.
    
    Determines feature to split on by minimizing entropy.
    
    Theory
    --------
        - Decision Trees are helpful when need to understand how different\
        values will change the class.
        - Only examine one feature at a time, so create rectangular classification boxes.
        - Very high variance, structure may heavily change due to sampling.
        - Splitting is locally greedy due to minimizing entropy at current step.
        - Can be unwieldy, due to it being possible for relatively simple data\
        creating a large tree.
        - It is very likely that this decision tree will overfit due to not having\
        any kind of regularization implemented.
        - No way to get a confidence measurement.
    """
    class LeafNode(object):
        def __init__(self, estimate):
            self._estimate = estimate
            
        def predict(self, row):
            return self._estimate
        
        def print_tree(self, offset):
            """
            Parameters
            --------
            
            offset : string
                All spaces that should be printed before printing out any content
                from the node.
            """
            print(offset + "Class:", str(self._estimate))
    
    class CategoricalNode(object):
        """
        Splits the tree based on the different values possible for the given feature
        """
        def __init__(self, default_class, class_and_counts, feature_index_split_on, children_dict):
            self._default_class = default_class
            self._class_and_counts = class_and_counts
            self._feature_index_split_on = feature_index_split_on
            self._children_dict = children_dict
            
        def predict(self, row):
            feature_val = row[self._feature_index_split_on]
            if feature_val in self._children_dict:
                return self._children_dict[feature_val].predict(row)
            else:
                # The feature value wasn't encountered, so go with default class
                return self._default_class
        
        def print_tree(self, offset):
            """
            Parameters
            --------
            
            offset : string
                All spaces that should be printed before printing out any content
                from the node.
            """
            print(offset + "Split on", self._feature_index_split_on,
                  "had class dist", list(self._class_and_counts),
                  "and default", self._default_class)
            value_offset = offset + "  "
            child_offset = offset + "    "
            for feature_value in self._children_dict:
                print(value_offset + "value", str(feature_value))
                self._children_dict[feature_value].print_tree(child_offset)
    
    def fit(self, X, y):
        """
        Fit internal parameters to minimize MSE on given X y dataset.
        
        Parameters
        ---------
        
        X : array-like, shape [n_samples, n_features]
            Input array of features.
            
        y : array-like, shape [n_samples,]
            Input array of expected results.
        """
        available_features = [i for i in range(0, X.shape[1])]
        
        features_values = [np.unique(column) for column in X.T]
        examples = np.append(X, np.reshape(y, (np.shape(X)[0], 1)), axis=1)
        
        class_values, class_counts = self._get_unique_class_vals_and_counts(examples)
        most_common_class = class_values[np.argmax(class_counts)]
        
        self._base_node = self._fit_node(examples, available_features, features_values, most_common_class)
    
    def _fit_node(self, examples, available_features, features_values, default_class):
        """
        Returns the sub-tree generated by 
        
        Parameters
        ---------
        examples : array-like, shape [n_samples, n_features + 1]
            Input array of samples with features and class. class identifier
            must have been appended to end of rows.
        
        available_features : array-like, shape [n_available_features]
            The index for each feature that has not yet been used. Element at index
            i corresponds to features_values at index i.
        
        features_values : array-like, shape [n_available_features]
            All possible values for each feature that has not yet been used.
            Element at index i corresponds to features_values at index i.
        
        default_class : numeric
            From the parent node, what was the most common class
        """
        if len(examples) == 0:
            return DecisionTree.LeafNode(default_class)
        
        class_values, class_counts = self._get_unique_class_vals_and_counts(examples)
        most_common_class = class_values[np.argmax(class_counts)]
        
        # If only one value or no other features, just pick most common
        if len(class_values) == 1 or len(available_features) == 0:
            return DecisionTree.LeafNode(most_common_class)
        
        # Otherwise calculate the best feature, which will minimize entropy.
        best_uncertainty = self._uncertainty(examples, available_features[0], features_values[0])
        best_feature_index = 0
        
        for feature in range(1, len(available_features)):
            uncertainty = self._uncertainty(examples,
                                            available_features[feature],
                                            features_values[feature])
            
            if uncertainty < best_uncertainty:
                best_uncertainty = uncertainty
                best_feature_index = feature
        
        chosen_feature = available_features[best_feature_index]
        chosen_feature_values = features_values[best_feature_index]
        
        # Remove the chosen feature from available features and their possible values.
        next_available_features =\
            available_features[:best_feature_index] + available_features[(best_feature_index +1):]
        next_features_values =\
            features_values[:best_feature_index] + features_values[(best_feature_index +1):]
        
        # Calculate for the children
        children_dict = {}
        for value in chosen_feature_values:
            examples_with_value = examples[examples[:, chosen_feature] == value, :]
            children_dict[value] = self._fit_node(examples_with_value, next_available_features, next_features_values, most_common_class)
            
        return DecisionTree.CategoricalNode(most_common_class, zip(class_values, class_counts), chosen_feature, children_dict)
    
    def _uncertainty(self, examples, feature_index, features_values):
        """
        Given a feature to split on, will calculate the resulting entropy by splitting
        the samples on the features different values.
        
        Assumes the feature is categorical.
        
        Parameters
        ---------
        examples : array-like, shape [n_samples, n_features + 1]
            Input array of samples with features and class. class identifier
            must have been appended to end of rows.
            
        feature_index : numeric
            Index for the feature for the samples.
        
        features_values:
            All of the unique values the feature can take.
        
        Returns
        -------
        Non-negative value for the entropy that would result from splitting the data
        on the given split.
        """
        penalty = 0
        
        for attrib_val in features_values:
            examples_with_val = examples[examples[:, feature_index] == attrib_val, :]
            if examples_with_val.size == 0:
                continue
            
            _, counts = self._get_unique_class_vals_and_counts(examples_with_val)
            
            penalty += entropy(counts)
        
        return penalty
    
    def predict(self, X):
        """
        Predict the value(s) associated with each row in X.
        
        X must have the same size for n_features as the input this instance was
        trained on.
        
        Parameters
        ---------
        
        X : array-like, shape [n_samples, n_features]
            Input array of features.
            
        Returns
        ------
        array-like, shape [n_samples]
            Class predicted by tree for each sample.
        """
        return np.apply_along_axis(self._base_node.predict,
                                   axis=1, arr=X)
    
    def print_tree(self):
        self._base_node.print_tree("")
        
    def _get_unique_class_vals_and_counts(self, examples):
        """
        Given a set of rows, will return all of the unique classes and their counts.
        
        Parameters
        ---------
        examples : array-like, shape [n_samples, n_features + 1]
            Input array of samples with features and class. class identifier
            must have been appended to end of rows.
        
        Returns
        ----------
        (unique_values, counts) where counts[i] is number of occurrences of
        unique_values[i].
        """
        return np.unique(examples[:, -1], return_counts=True)
    
    def get_feature_params(self):
        pass

