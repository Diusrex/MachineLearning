import numpy as np

from abc import ABC, abstractmethod

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
    Generates a decision tree to classify future data using the
    provided examples with categorical data.
    
    Currently assumes only given categorical data.
    
    Determines feature to split on by minimizing uncertainty (currently just entropy).
    
    Parameters
    ---------
    algorithm_to_use : string
        What decision tree generation algorithm should be used. Currently only
        ID3 is supported.
    
    max_depth : numeric
        If provided, will limit the depth that the tree can reach. May help
        prevent overfitting.
        
    min_node_samples : numeric
        If provided, any nodes that have less samples than the min will
        be forced to be a leaf.
    
    Theory
    --------
        - Decision Trees are helpful when need to understand how different\
        values will change the class.
        - Only examine one feature at a time, so create rectangular classification boxes.
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
        else:
            raise ValueError("algorithm_to_use value '{}' is not valid, only ID3 is supported".format(algorithm_to_use))
        
        self._max_depth = max_depth
        self._min_node_samples = min_node_samples
    
    class LeafNode(object):
        def __init__(self, estimate, class_and_counts):
            self._estimate = estimate
            self._class_and_counts = class_and_counts
            
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
            print(offset + "Class:", str(self._estimate), "had class-count dist", list(self._class_and_counts))
    
    class CategoricalNode(object):
        """
        Splits the tree based on the different values possible for the given feature
        """
        def __init__(self, default_class, class_and_counts, feature_index_split_on, feature_value_to_group_map,
                     groups_dict):
            self._default_class = default_class
            self._class_and_counts = class_and_counts
            self._feature_index_split_on = feature_index_split_on
            self._feature_value_to_group_map = feature_value_to_group_map
            self._groups_dict = groups_dict
            
        def predict(self, row):
            feature_val = row[self._feature_index_split_on]
            group = self._feature_value_to_group_map(feature_val)
            if group in self._groups_dict:
                return self._groups_dict[group].predict(row)
            else:
                # The group wasn't encountered, so go with default class
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
                  "had class-count dist", list(self._class_and_counts),
                  "and default", self._default_class)
            value_offset = offset + "  "
            child_offset = offset + "    "
            for feature_value in self._groups_dict:
                print(value_offset + "value", str(feature_value))
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
            Input array of samples with features and class. class identifier
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
        
        class_values, class_counts = self._get_unique_class_vals_and_counts(examples)
        most_common_class = class_values[np.argmax(class_counts)]
        
        # Force as leaf if only 1 class, no features, hit max depth, or doesn't have enough samples.
        if len(class_values) == 1 or len(available_features) == 0 or\
            self._reached_max_depth(depth) or\
            not self._has_enough_samples_for_node(examples.shape[0]):
            return DecisionTree.LeafNode(most_common_class, zip(class_values, class_counts))
        
        best_uncertainty = None
        
        best_feature_index_in_available = None
        best_feature_value_to_group_map = None
        best_examples_by_group = None
        
        for feature_index_in_available in range(len(available_features)):
            for feature_value_to_group_map, examples_by_group in\
                self._split_algorithm.create_splits_from_feature_values(
                        examples, available_features[feature_index_in_available],
                        features_values[feature_index_in_available]):
                
                uncertainty = self._uncertainty(examples_by_group)
                
                if best_uncertainty is None or best_uncertainty > uncertainty:
                    best_uncertainty = uncertainty
                    best_feature_index_in_available = feature_index_in_available
                    best_examples_by_group = examples_by_group
                    best_feature_value_to_group_map = feature_value_to_group_map
        
        
        chosen_feature_index_in_examples = available_features[best_feature_index_in_available]
        
        # Calculate for the groups
        groups_dict = {}
        for group in best_examples_by_group:
            examples_in_group = best_examples_by_group[group]
            
            next_available_features = available_features
            next_features_values = features_values
            
            feature_values_orig = next_features_values[best_feature_index_in_available]
            
            # Only bother to set the feature values group if it won't actally be immediately
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
        
        return DecisionTree.CategoricalNode(most_common_class, zip(class_values, class_counts),
                                            chosen_feature_index_in_examples, best_feature_value_to_group_map,
                                            groups_dict)
    
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
            Class predicted by tree for each sample.
        """
        return np.apply_along_axis(self._base_node.predict,
                                   axis=1, arr=X)
    
    
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
            _, counts = self._get_unique_class_vals_and_counts(examples_in_group)
            # TODO: Be able to customize the penalty used.
            # Note that I should also update the comments to reflect this.
            penalty += entropy(counts)
        
        return penalty
    
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


class _SplitAlgorithm(ABC):
    """
    Algorithm that will describe how the values that occur for the feature should
    be split and how the feature should be handled after use.
    
    Currently only supports categorical variables.
    """
    @abstractmethod
    def create_splits_from_feature_values(self, examples, feature_index, feature_values):
        """
        A function to create a SplitWithFeatureValues instance given the feature we
        care about and all the examples.
        
        Iterating through the returned object will yield each
        [feature_value_to_group_map, groups]
        for every possible split.
        
        For example, can cause it to have a child node for each feature value by
        mapping each value to itself. Or can force a binary tree by mapping each
        value to 0 or 1.
        
        Parameters
        ---------
        examples : array-like, shape [n_samples, n_features + 1]
            Input array of samples with features and class. class identifier
            must have been appended to end of rows.
        
        available_features : array-like, shape [n_available_features]
            The index for each feature that can still be used. Element at index
            i corresponds to features_values at index i.
        
        features_values : array-like, shape [n_available_features]
            All possible values for each feature that can still be used.
            Element at index i corresponds to features_values at index i.
        
        Returns
        ---------
        Array-like of all splits. Split will contain [feature_value_to_group_map, groups]
        pairs.
        """
        pass
    
    def _separate_examples_into_groups(self, examples, feature_index, feature_values, feature_value_to_group_map):
        """
        Given the examples, what feature to split on and the feature val to group map,
        will return a map from group to all of the samples that are contained in that
        group.
        
        Parameters
        ---------
        examples : array-like, shape [n_samples, n_features + 1]
            Input array of samples with features and class. class identifier
            must have been appended to end of rows.
            
        feature_index : numeric
            Index for the feature for the samples.
        
        feature_values
            All possible values for the current feature.
        
        feature_value_to_group_map : function
            Given the feature, return what group the feature is a part of.
        """
        examples_by_group = {}
        for feature_val in feature_values:
            examples_with_val = examples[examples[:, feature_index] == feature_val, :]
            # Don't bother when doesn't have any samples.
            if examples_with_val.size == 0:
                continue
            
            group_val = feature_value_to_group_map(feature_val)
            if group_val not in examples_by_group:
                examples_by_group[group_val] = examples_with_val
            else:
                examples_by_group[group_val] = np.vstack(
                        (examples_by_group[group_val], examples_with_val))
        
        return examples_by_group
    
    
    @abstractmethod
    def should_remove_feature_after_use(self):
        """
        Return true iff a categorical feature can be removed after splitting
        on it.
        """
        pass

class _ID3_Algorithm(_SplitAlgorithm):
    # TODO: More theory!
    """
    Implements the ID3 (Iterative Dichotomiser 3) Algorithm invented by Ross Quinlan.
    Currently only supports categorical data.
    
    For categorical data, will split it so there is a group for each value possible.
    Of course, this means that categorical data will only be used once.
    
    Normally doesn't include pruning, but I will look into add that later. Pruning
    will be done after creating the tree, to allow interactions between categories,
    which may not show up when creating the tree.
    
    See _SplitAlgorithm for descriptions of the functions.
    
    Theory
    -------
        - Creates a very broad tree, so can run out of data quickly when categorical\
        variables have many possible values.
        - Data can be overfitted if small sample is used.
        - Can be unwieldy, due to chance of relatively simple data creating a large tree.
        - In general, C4.5 is better due tonC4.5 natively handling continous +\
        discrete, incomplete data points, pruning, and adding different weights.\
        However, I haven't been able to find a good resource to explain the\
        theory behind C4.5.
    """
    
    def create_splits_from_feature_values(self, examples, feature_index, feature_values):
        """
        See _SplitAlgorithm for descriptions of the function
        """
        # Very simple, just map each feature_value to itself - create a child
        # for each node.
        m = {}
        for value in feature_values:
            m[value] = value
        
        def feature_map(value):
            if value in m:
                return m[value]
            # Something that should never be in the values
            return "Some ridiculous string that should never show up so will not be recognized"
        
        groups = self._separate_examples_into_groups(examples, feature_index, feature_values,
                                                     feature_map)
        # Only creates one split
        split = (feature_map, groups,)
        return (split,)
        
    def should_remove_feature_after_use(self):
        """
        See _SplitAlgorithm for descriptions of the function
        """
        # For categorical features, will split the different categories completely
        # apart, so won't split on them again.
        return True
