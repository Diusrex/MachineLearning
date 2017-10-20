import numpy as np

from abc import ABC, abstractmethod

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")

from supervised_learning.internal.decision_tree_split_algorithms import _ID3_Algorithm, _CART_Algorithm
from util.data_operation import entropy, mean_square_error

# TODO: Later will have categorical="none", then can be changed to "all" or
# a 1d array
# TODO: Add regularization - max depth, few observations in node, entropy change is small,
# cross-validation entropy starts to increase.
class DecisionTree(ABC):
    """
    Generates a decision tree to predict future data using the provided examples
    with categorical data.
    
    Determines feature to split on by minimizing uncertainty.
    
    Parameters
    ---------
    algorithm_to_use : string
        What decision tree generation algorithm should be used. Currently only
        ID3 and CART are supported.
    
    max_depth : numeric
        If provided, will limit the depth that the tree can reach. May help
        prevent overfitting.
        
    min_node_samples : numeric
        If provided, any nodes that have less samples than the min will
        be forced to be a leaf.
    
    tuning_param
        How should the cost-complexity pruning proceed. If 'find', 
    
    Theory
    --------
        - Decision Trees are helpful when need to understand how different\
        values will change the expected result.
        - Only examine one feature at a time, so creates rectangular estimation boxes.
        - Very high variance, structure may heavily change due to sampling.
        - Splitting is locally greedy due to minimizing uncertainty at current step, and\
        is likely to reach a local optimum.
        - Due to the greedy splitting, will prefer shorter trees, but will often\
        not pick the shortest possible tree.
        - No way to get a confidence measurement (although this can be added when using\
        GINI index)
    
    References
    --------
           T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.
    """
    
    def __init__(self, algorithm_to_use='ID3', max_depth=None, min_node_samples=None):
        if algorithm_to_use == 'ID3':
            self._split_algorithm = _ID3_Algorithm()
        elif algorithm_to_use == 'CART':
            self._split_algorithm = _CART_Algorithm()
        else:
            raise ValueError("algorithm_to_use value '{}' is not valid, only ID3 is supported".format(algorithm_to_use))
        
        self._max_depth = max_depth
        self._min_node_samples = min_node_samples
    
    class LeafNode(object):
        def __init__(self, estimate, values_and_counts):
            self._estimate = estimate
            self._values_and_counts = values_and_counts
            
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
            print(offset + "Estimate:", str(self._estimate), "had value-count dist", list(self._values_and_counts))
    
    class CategoricalNode(object):
        """
        Splits the tree based on the different values possible for the given feature
        """
        def __init__(self, default_estimate, values_and_counts, feature_index_split_on, feature_value_to_group_map,
                     group_split_explanation, groups_dict):
            self._default_estimate = default_estimate
            self._values_and_counts = values_and_counts
            self._feature_index_split_on = feature_index_split_on
            self._feature_value_to_group_map = feature_value_to_group_map
            self._group_split_explanation = group_split_explanation
            self._groups_dict = groups_dict
            
        def predict(self, row):
            feature_val = row[self._feature_index_split_on]
            group = self._feature_value_to_group_map(feature_val)
            if group in self._groups_dict:
                return self._groups_dict[group].predict(row)
            else:
                # The group wasn't encountered, so go with default.
                return self._default_estimate
        
        def print_tree(self, offset):
            """
            Parameters
            --------
            offset : string
                All spaces that should be printed before printing out any content
                from the node.
            """
            print(offset + "Split on feature", self._feature_index_split_on,
                  self._group_split_explanation,
                  "had value-count dist", list(self._values_and_counts),
                  "and default", self._default_estimate)
            value_offset = offset + "  "
            child_offset = offset + "    "
            for feature_value in self._groups_dict:
                print(value_offset + "group", str(feature_value) + ":")
                self._groups_dict[feature_value].print_tree(child_offset)
    
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
        self._split_algorithm.check_data_is_valid(X, y)
        
        available_features = [i for i in range(0, X.shape[1])]
        
        features_values = [np.unique(column) for column in X.T]
        examples = np.append(X, np.reshape(y, (np.shape(X)[0], 1)), axis=1)
        
        self._base_node = self._fit_node(examples, available_features, features_values,
                                         1) # First will have depth of 1

    def _fit_node(self, examples, available_features, features_values, depth):
        """
        Returns the sub-tree generated by fitting the examples using some (or all)
        of the available features.
        
        Parameters
        ---------
        examples : array-like, shape [n_samples, n_features + 1]
            Input array of samples with features and expected value. The expected value
            must have been appended to end of rows.
        
        available_features : array-like, shape [n_available_features]
            The index for each feature in examples that can still be used. Element
            at index i corresponds to features_values at index i.
        
        features_values : array-like, shape [n_available_features]
            All possible values for each feature that can still be used.
            Element at index i corresponds to features_values at index i.
        
        depth : numeric
            How many nodes (including node about to be created) are there to root
            of tree.
        """
        # There should always be at least one example.
        assert(len(examples) > 0)
        unique_values, values_counts = self._get_unique_values_and_counts(examples)
        best_estimate = self._get_best_estimate(unique_values, values_counts)
        
        # Force as leaf if only 1 value, no features, hit max depth, or doesn't have enough samples.
        if len(unique_values) == 1 or len(available_features) == 0 or\
            self._reached_max_depth(depth) or\
            not self._has_enough_samples_for_node(examples.shape[0]):
            return DecisionTree.LeafNode(best_estimate, zip(unique_values, values_counts))
        
        best_uncertainty = None
        
        best_feature_index_in_available = None
        best_feature_value_to_group_map = None
        best_group_split_explanation = None
        best_examples_by_group = None
        
        for feature_index_in_available in range(len(available_features)):
            for feature_value_to_group_map, examples_by_group, group_split_explanation in\
                self._split_algorithm.create_splits_from_feature_values(
                        examples, available_features[feature_index_in_available],
                        features_values[feature_index_in_available]):
                
                uncertainty = self._uncertainty(examples_by_group)
                
                if best_uncertainty is None or best_uncertainty > uncertainty:
                    best_uncertainty = uncertainty
                    best_feature_index_in_available = feature_index_in_available
                    best_examples_by_group = examples_by_group
                    best_feature_value_to_group_map = feature_value_to_group_map
                    best_group_split_explanation = group_split_explanation
        
        
        chosen_feature_index_in_examples = available_features[best_feature_index_in_available]
        
        # Calculate for the groups
        groups_dict = {}
        for group in best_examples_by_group:
            examples_in_group = best_examples_by_group[group]
            
            next_available_features = available_features
            next_features_values = features_values
            
            feature_values_orig = next_features_values[best_feature_index_in_available]
            
            # Only bother to update the feature values group if it won't be immediately
            # removed
            if not self._split_algorithm.should_remove_feature_after_use():
                next_features_values[best_feature_index_in_available] =\
                    np.unique(examples_in_group[:, best_feature_index_in_available])
            
            # Should not be used, or only has one value left.
            if self._split_algorithm.should_remove_feature_after_use() or\
                len(next_features_values[best_feature_index_in_available]) == 1:
                # Remove the chosen feature from available features and their possible values
                # if the split algorithm says that is fine.
                next_available_features =\
                    available_features[:best_feature_index_in_available] +\
                    available_features[(best_feature_index_in_available +1):]
                next_features_values =\
                    features_values[:best_feature_index_in_available] +\
                    features_values[(best_feature_index_in_available +1):]
            
            groups_dict[group] = self._fit_node(
                    examples_in_group, next_available_features,
                    next_features_values, depth + 1)
            
            # Restore the original features_values
            if not self._split_algorithm.should_remove_feature_after_use():
                features_values[best_feature_index_in_available] = feature_values_orig
        
        return DecisionTree.CategoricalNode(best_estimate, zip(unique_values, values_counts),
                                            chosen_feature_index_in_examples, best_feature_value_to_group_map,
                                            best_group_split_explanation,
                                            groups_dict)
    
    @abstractmethod
    def _get_best_estimate(self, unique_values, values_counts):
        """
        Given all of the different unique values and their count, return
        the estimate that will minimize the error.
        
        Parameters
        ---------
        unique_values : array-like, shape[num_unique_values]
            All of the different unique values at the current node.
            
        values_counts : array-like, shape[num_unique_values]
            The number of occurrences for each value in unique_values.
        """
        pass
    
    def _reached_max_depth(self, depth):
        return self._max_depth is not None and depth >= self._max_depth
    
    def _has_enough_samples_for_node(self, num_samples):
        return self._min_node_samples is None or num_samples >= self._min_node_samples
    
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
            Estimate predicted by tree for each sample.
        """
        return np.apply_along_axis(self._base_node.predict,
                                   axis=1, arr=X)
    
    @abstractmethod
    def _uncertainty(self, examples_split_by_groups):
        """
        Given how the examples are separated into different groups, will
        calculate the uncertainty for the groups distribution.
        
        Currently only entropy is used as the measure of uncertainty.
        
        Parameters
        ---------
        examples_split_by_groups : map from group to examples in group
            Contains all of the samples
        
        Returns
        -------
        Non-negative value for the entropy that would result from splitting the data
        on the given split.
        """
        pass
    
    def print_tree(self):
        self._base_node.print_tree("")
        
    def _get_unique_values_and_counts(self, examples):
        """
        Given a set of rows, will return all of the unique values and their counts.
        
        Parameters
        ---------
        examples : array-like, shape [n_samples, n_features + 1]
            Input array of samples with features and expected value. The expected value
            must have been appended to end of rows.
        
        Returns
        ----------
        (unique_values, counts) where counts[i] is number of occurrences of
        unique_values[i].
        """
        return np.unique(examples[:, -1], return_counts=True)
    
    def get_feature_params(self):
        pass

class DecisionTreeClassifier(DecisionTree):
    """
    Generates a decision tree to predict the class of future data using provided
    features.
    
    Determines feature to split on by minimizing uncertainty (currently just entropy).
    
    Parameters
    ---------
    algorithm_to_use : string
        What decision tree generation algorithm should be used. Currently only
        ID3 and CART are supported.
    
    max_depth : numeric
        If provided, will limit the depth that the tree can reach. May help
        prevent overfitting.
        
    min_node_samples : numeric
        If provided, any nodes that have less samples than the min will
        be forced to be a leaf.
    """
    def __init__(self, algorithm_to_use='ID3', max_depth=None, min_node_samples=None):
        super().__init__(algorithm_to_use=algorithm_to_use, max_depth=max_depth,
             min_node_samples=min_node_samples)
        
    def _get_best_estimate(self,
                           unique_values, values_counts):
        # Just return the most common class
        return unique_values[np.argmax(values_counts)]
        
    
    def _uncertainty(self, examples_split_by_groups):
        """
        Given how the examples are separated into different groups, will
        calculate the uncertainty for the groups distribution.
        
        Currently only entropy is used as the measure of uncertainty.
        
        Parameters
        ---------
        examples_split_by_groups : map from group to examples in group
            Contains all of the samples
        
        Returns
        -------
        Non-negative value for the entropy that would result from splitting the data
        on the given split.
        """
        penalty = 0
        for group in examples_split_by_groups:
            examples_in_group = examples_split_by_groups[group]
            _, counts = self._get_unique_values_and_counts(examples_in_group)
            # TODO: Be able to customize the penalty used.
            # Note that I should also update the comments to reflect this.
            penalty += entropy(counts)
        
        return penalty


class DecisionTreeRegression(DecisionTree):
    """
    Generates a decision tree to predict the value of future data using provided
    features.
    
    Determines feature to split on by minimizing uncertainty (currently just MSE).
    
    Parameters
    ---------
    algorithm_to_use : string
        What decision tree generation algorithm should be used. Currently only
        ID3 and CART are supported.
    
    max_depth : numeric
        If provided, will limit the depth that the tree can reach. May help
        prevent overfitting.
        
    min_node_samples : numeric
        If provided, any nodes that have less samples than the min will
        be forced to be a leaf.
    """
    def __init__(self, algorithm_to_use='ID3', max_depth=None, min_node_samples=None):
        super().__init__(algorithm_to_use=algorithm_to_use, max_depth=max_depth,
             min_node_samples=min_node_samples)
        
    def _get_best_estimate(self,
                           unique_values, values_counts):
        # Select (weighted) average
        return np.average(unique_values, weights=values_counts)
    
    def _uncertainty(self, examples_split_by_groups):
        """
        Given how the examples are separated into different groups, will
        calculate the uncertainty for the groups distribution.
        
        The uncertainty within a group is the MSE from the best estimate (mean of all
        samples) to each samples value.
        
        Parameters
        ---------
        examples_split_by_groups : map from group to examples in group
            Contains all of the samples
        
        Returns
        -------
        Sum of the MSE for each group.
        """
        penalty = 0
        for group in examples_split_by_groups:
            examples_in_group = examples_split_by_groups[group]
            all_y_values = examples_in_group[:, -1]
            # For each group, best value for estimate would be the mean
            best_estimate = np.mean(all_y_values)
            
            penalty += mean_square_error(all_y_values, best_estimate)
        
        return penalty

