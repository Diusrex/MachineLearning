from sklearn import datasets

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
