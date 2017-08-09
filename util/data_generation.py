from sklearn import datasets
import numpy as np

def create_linearly_separable_two_class():
    """
    Generates a linearly separable dataset, with classes 0 or 1.
    
    Note that the dataset is not generated randomly.
    
    Returns
    --------
    
    X : array-like, shape [n_samples,n_features]
        All available input data.

    y : array-like, shape [n_samples,]
        All available expected output data. Will be 0 or 1
    """
    # TODO: Would like to randomly generate instead of using this dataset.
    iris = datasets.load_iris()
    # Only use first two features
    X = iris.data[:, :2]
    y = iris.target
    # Only keep first classes 0 and 1
    wanted_data = y < 2
    
    X = X[wanted_data]
    y = y[wanted_data]
    
    return X, y

def create_2d_categorical_feature_two_class():
    """
    
    Note that the dataset is not generated randomly and is x-or of both x1,
    x2 being odd.
    
    """
    # Note: Toy problem for now, should probably be made more complex...
    # We care about a mix of X1, X2
    X = np.array([[x1, x2] for x1 in range(10) for x2 in range(10)])
    y = X[:, 0] > X[:, 1]
    return X, y

def create_1d_categorical_feature_regression():
    """
    Creates a 1 categorical feature data set, which will have
    multiple different values at each category
    """
    X = np.array([[x1, x2] for x1 in range(10) for x2 in range(10)])
    
    # Have 2 values for each category - slightly above and slightly below x1 * x2
    base = X[:,0] + X[:, 1]
    below = base - 0.5
    above = base + 0.5
    
    X = np.vstack([X, X])
    y = np.append(below, above)
    
    return X, y