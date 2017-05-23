import numpy as np
from sklearn import datasets

# Add base directory of project to path.
import matplotlib.pyplot as plt
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")

from supervised_learning import linear_regression
from util.data_manipulation import train_test_split
from util.data_operation import mean_square_error


class FeaturePruner(object):
    """
    Note that this approach works better when the features have the same scale,
    since it will just dumbly remove features that have a small magnitude weight.
    Otherwise, a feature with a huge scale would naturally have a small
    magnitude weight.
       
    Parameters
    --------
    estimator
        Class that should support fit(X, y) and get_feature_params().
        get_feature_params() should return the weight placed on each feature in order.
    
    features_to_reduce_to : int
        Maximum number of features after pruning.
        
    step : int
        Number of features to remove each iteration
       
    Warning
    --------
    Since this depends on the weight provided to each feature, the features
    should probably be rescaled. Otherwise, features with a larger weight
    would be more likely to be pruned.
    """
    
    def __init__(self, estimator, features_to_reduce_to, step=1):
        self._estimator = estimator
        self._features_to_reduce_to = features_to_reduce_to
        self._step = step
        self._including_feature = None
        
    def fit(self, X, y):
        """
        Fit FeaturePruner to X.
        
        Parameters
        ---------
        
        X : array-like, shape [n_samples, n_features]
            Input array of features.
            
        y : array-like, shape [n_samples,]
            Input array of expected results. Can be 2 dimensional, if estimating
            multiple different values for each sample.
        """
        num_features = X.shape[1]
        
        # Will be used to index X to remove unwanted features.
        self._including_feature = np.ones((num_features,), dtype=bool)
        
        # If current_features_index[i] = j, then the ith feature in
        # X at this step corresponds to the jth feature in X before
        # pruning any features.
        index_for_current_features = np.arange(num_features)
        
        while len(index_for_current_features) > self._features_to_reduce_to:
            self._estimator.fit(X[:, self._including_feature], y)
            weights = self._estimator.get_feature_params()
            
            # Find the step# smallest weighted features.
            current_features_removed = np.argsort(abs(weights))[:self._step]
            
            features_to_remove = index_for_current_features[current_features_removed]
            self._including_feature[features_to_remove] = False
            
            index_for_current_features = np.delete(index_for_current_features,
                                               current_features_removed)
    
    def transform(self, X):
        """
        Return X after reducing number of features down to features_to_reduce_to
        
        """
        return X[:, self._including_feature]
    
    def fit_transform(self, X, y):
        """
        Fit to X,y then return X after reducing number of features down to
        features_to_reduce_to
        
        """
        self.fit(X, y)
        return self.transform(X)
    
if __name__ == "__main__":
    # Just include one relevant feature, which we will graph upon.
    # Given far too many features with not enough samples, so will often
    # overfit when not pruning.
    X, y = datasets.make_regression(n_samples=100, n_features=30, n_informative=1,
                                    noise=5)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_proportion=0.2)
    
    # Without any pruning
    reg_orig = linear_regression.LinearRegression()
    reg_orig.fit(X_train, y_train)
    y_pred_orig = reg_orig.predict(X_test)
    orig_mse = mean_square_error(y_pred_orig, y_test)
    
    # Setup pruner and prune the # of feautres features down to 1
    pruner = FeaturePruner(linear_regression.LinearRegression(), 1)
    X_train_pruned = pruner.fit_transform(X_train, y_train)
    X_test_pruned = pruner.transform(X_test)
    
    reg_pruned = linear_regression.LinearRegression()
    reg_pruned.fit(X_train_pruned, y_train)
    y_pred_pruned = reg_pruned.predict(X_test_pruned)
    pruned_mse = mean_square_error(y_pred_pruned, y_test)
    
    actual = plt.scatter(X_test_pruned, y_test, color="Black", label="Actual Value")
    pruned = plt.scatter(X_test_pruned, y_pred_pruned, color="Red", label="Pruned Features")
    orig = plt.scatter(X_test_pruned, y_pred_orig, color="Pink", label="Original")
    plt.legend(loc='lower right', fontsize=8)
    plt.title("Linear Regression - Orig: (%.2f MSE) vs Pruned Features: (%.2f MSE)" % (orig_mse, pruned_mse))
    plt.show()