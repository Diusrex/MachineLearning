from abc import ABC, abstractmethod
import numpy as np

class _SplitAlgorithm(ABC):
    """
    Algorithm that will describe how the values that occur for the feature should
    be split and how the feature should be handled after use.
    
    Currently only supports categorical variables.
    """
    @abstractmethod
    def check_data_is_valid(self, X, y):
        """
        Ensures that all X + y data given is correct. Currently assumes is provided with
        categorical data.
        
        If the data doesn't meet all requirements, then will raise an exception.
        
        Parameters
        ---------
        X : array-like, shape [n_samples, n_features]
            Input array of features.
            
        y : array-like, shape [n_samples,]
            Input array of expected results.
        """
        pass
    
    @abstractmethod
    def create_splits_from_feature_values(self, examples, feature_index, feature_values):
        """
        A function to create a SplitWithFeatureValues instance given the feature we
        care about and all the examples.
        
        Iterating through the returned object will yield each
        [feature_value_to_group_map, groups, explanation]
        for every possible split.
        
        For example, can cause it to have a child node for each feature value by
        mapping each value to itself. Or can force a binary tree by mapping each
        value to 0 or 1.
        
        Parameters
        ---------
        examples : array-like, shape [n_samples, n_features + 1]
            Input array of samples with features and expected value. The expected value
            must have been appended to end of rows.
        
        available_features : array-like, shape [n_available_features]
            The index for each feature that can still be used. Element at index
            i corresponds to features_values at index i.
        
        features_values : array-like, shape [n_available_features]
            All possible values for each feature that can still be used.
            Element at index i corresponds to features_values at index i.
        
        Returns
        ---------
        Array-like of all splits. Split will contain
        [feature_value_to_group_map, groups, explanation]
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
            Input array of samples with features and expected value. The expected value
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
        - In general, C4.5 is better due to C4.5 natively handling continous +\
        discrete, incomplete data points, pruning, and adding different weights.\
        However, I haven't been able to find a good resource to explain the\
        theory behind C4.5.
    """
    def check_data_is_valid(self, X, y):
        """
        See _SplitAlgorithm for descriptions of the function
        """
        # Doesn't need to do any checks, so long as all inputs are categorical
        # will run fine.
        pass
    
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
        split = (feature_map, groups, "different group per value")
        return (split,)
        
    def should_remove_feature_after_use(self):
        """
        See _SplitAlgorithm for descriptions of the function
        """
        # For categorical features, will split the different categories completely
        # apart, so won't split on them again.
        return True

class _CART_Algorithm(_SplitAlgorithm):
    """
    Implements the CART (Categorical and Regression Tree) Algorithm. Will create
    a binary tree.
    Currently only supports categorical data with two possible categories. The
    value of the categories doesn't matter.
    
    For categorical data, will order the data based on proportion of 1 label
    for each value of categorical feature. Will then try splitting at each value,
    where each value with a lower proportion goes to one group, rest goes to the other.
    
    Normally does include cost-complexity pruning, but will add that later.
    Pruning is done after creating the tree.
    
    See _SplitAlgorithm for descriptions of the functions.
    
    Theory
    -------
        - Compared to ID3 will create a more compressed (but possibly far taller)\
        tree due to only splitting into two different groups.
        - Can be more generalizable than ID3 due to only splitting into two different\
        groups.
        - Should normally be run with cost-complexity pruning, due to this algorithm\
        being able to continously split the data.
        - Treat categorical variables with many different values carefully - they\
        can easily be overfit, due to having ~ 2^(#diff values) different possible\
        splits.
    """
    def check_data_is_valid(self, X, y):
        """
        See _SplitAlgorithm for descriptions of the function
        """
        unique_values = np.unique(y)
        if len(unique_values) != 2:
            raise ValueError("CART requires binary data for categorical classification. " +
                             "> 2 classes or regression will not work.")
    
    def create_splits_from_feature_values(self, examples, feature_index, feature_values):
        """
        See _SplitAlgorithm for descriptions of the function
        """
        # First, split the data up into different values.
        m = {}
        for value in feature_values:
            m[value] = value
        examples_by_value = self._separate_examples_into_groups(
                examples, feature_index, feature_values, lambda val: m[val])
        
        # Which category will be used for the proportion.
        category_for_proportion = np.unique(examples[:, -1])[0]
        
        # Now, order the values by proportion.
        # Do this by grouping the value with proportion, then sorting
        proportion_for_value = []
        for value in examples_by_value:
            examples = examples_by_value[value]
            proportion = np.mean(examples[:, -1] == category_for_proportion)
            proportion_for_value.append((proportion, value))
        
        # Order from most to least.
        proportion_for_value.sort()
        
        index_to_value = {}
        value_to_index = {}
        
        # Will be transfering values from group 1 to group 0.
        group_0_elements = np.zeros((0, examples.shape[1]))
        group_1_elements = np.zeros((0, examples.shape[1]))
        # Generate mapping from value to index and index to value.
        # Also put all values into group_1 in order of index.
        for idx, proportion_and_value in enumerate(proportion_for_value):
            value = proportion_and_value[1]
            index_to_value[idx] = value
            value_to_index[value] = idx
            
            group_1_elements = np.vstack((group_1_elements, examples_by_value[value]))
        
        splits = []
        # Separator is the first element in group 1. All indicies before it
        # will be in group 0.
        for separator in range(1, len(index_to_value)):
            # Samples with value index just before separator need to be transferred
            # from group 1 to group 0.
            value_previous = index_to_value[separator - 1]
            examples_for_prev_value = examples_by_value[value_previous]
            group_0_elements = np.vstack((group_0_elements, examples_for_prev_value))
            # Remove the elements
            group_1_elements = group_1_elements[examples_for_prev_value.shape[0]:,:]
            
            feature_map = _CART_Algorithm._Splitter(value_to_index, separator)
            groups = {False: group_0_elements, True: group_1_elements}
            
            values_in_groups = {False: [], True: []}
            for value in value_to_index:
                group = feature_map(value)
                values_in_groups[group].append(value)
            
            explanation = "group splits: " + str(values_in_groups)
            splits.append((feature_map, groups, explanation))
        
        return splits
        
        
    def should_remove_feature_after_use(self):
        """
        See _SplitAlgorithm for descriptions of the function
        """
        # For categorical or discrete variables, can perform further binary splits.
        return False

    class _Splitter(object):
        """
        Class to act as feature_value_to_group_map. Wraps around the value to index map
        and separator, ensuring that later iterations changes to separator value
        don't change earlier iterations.
        """
        def __init__(self, value_to_index, separator):
            self._value_to_index = value_to_index
            self._separator = separator
            
        def __call__(self, value):
            """
            Return which group the value belongs to. Values that are recognized
            will go to groups False (group 0) or True (group 1).
            """
            if value in self._value_to_index:
                return self._value_to_index[value] >= self._separator
            # Only groups created are True and False
            return -1
