import numpy as np

def cross_validation(algorithm, X, y, error_function, num_folds=5, reorder_data=True):
    """
    Will run k-fold cross validation using num_folds, returning the num_folds
    different errors from the error_function when predicting for each fold.
    
    Parameters
    --------
    algorithm
        Ml algorithm that supports fit(X, y) and predict(X).
    
    X : array-like, shape [n_samples, n_features]
        Input array of features. Will be shuffled around.
        
    y : array-like, shape [n_samples,]
        Input array of expected results.
    
    error_function : function(y_pred, y_true)
        Returns the wanted error_function given the predicted and actual y values
    
    folds : numeric
        Number of folds to use in CV
        
    reordered_data : boolean
        Should the data be reordered when running the folds.
    """
    return cross_validation_with_functions(algorithm.fit, algorithm.predict, X, y,
                                           error_function, num_folds, reorder_data=reorder_data)
    
def cross_validation_with_functions(train_func, pred_func, X, y, error_function, num_folds=5,
                                    reorder_data=True):
    """
    Will run k-fold cross validation using num_folds, returning the num_folds
    different errors from the error_function when predicting for each fold.
    
    Provides more control over what functions should be used to train and predict.
    For general case, use cross_validation.
    
    Parameters
    --------
    train_func : function(X, y)
        Function that will train the ML algorithm on the provided data
    
    pred_fun : function(X)
        Function that will predict using the ML algorithm after training using train_func.
    
    X : array-like, shape [n_samples, n_features]
        Input array of features. Will be shuffled around.
        
    y : array-like, shape [n_samples,]
        Input array of expected results.
    
    error_function : function(y_pred, y_true)
        Returns the wanted error_function given the predicted and actual y values
    """
    num_points = len(y)
    # First shuffle the data.
    if reorder_data:
        perm = np.random.permutation(num_points)
        X = X[perm]
        y = y[perm]
    
    data_per_fold = int(np.ceil(num_points / num_folds))
    X_folds = []
    y_folds = []
    for fold in range(num_folds):
        start = data_per_fold * fold
        end = min(data_per_fold * (fold + 1), num_points)
        X_folds.append(X[start:end, :])
        y_folds.append(y[start:end])
    
    # Run prediction for the folds
    errors = []
    for pred_fold in range(num_folds):
        [X_folds[fold] for fold in range(num_folds)]
        X_train = np.vstack([X_folds[fold] for fold in range(num_folds) if
                             fold != pred_fold])
    
        y_train = np.concatenate(
                [y_folds[fold] for fold in range(num_folds) if fold != pred_fold])
        
        train_func(X_train, y_train)
        
        error = error_function(
                pred_func(X_folds[pred_fold]),
                y_train[pred_fold])
        errors.append(error)
    
    return errors
