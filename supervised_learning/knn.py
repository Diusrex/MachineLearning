import heapq
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn import datasets

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")

from util.data_operation import mean_square_error
from util.data_manipulation import train_test_split


class KNN(object):
    """
    Parameters
    --------
    
    k : integer
        Number of closest examples to aggregate over.
    
    aggregation : function
        Combines the K closest y values together.
        This can be mean, median, or any other function.
        
    distance_function : function
        Function to calculate the distance between two points.
        First argument will be current row, second will be row comparing against.
    
    Theory
    --------
        - Dependent on output being predicted by the closest elements using \
        the distance function provided.
        - Has high variance, especially when k is small.
        - Has low bias
        - If the scale of different features is not important to prediction, they\
        should be normalized to avoid overemphasizing certain features.
        - May perform badly when given more and more features - the examples \
        become more and more spread out.
        
    """
    def __init__(self, k, aggregation, distance_function):
        self._k = k
        self._aggregation = aggregation
        self._distance_function = distance_function
        self._data = None
        
    def fit(self, X, y):
        """
        Store and possibly process the given inputs.
        
        Parameters
        ---------
        
        X : array-like, shape [n_samples, n_features]
            Input array of features.
            
        y : array-like, shape [n_samples,] or [n_samples,n_values]
            Input array of expected results. Can be 2 dimensional, if estimating
            multiple different values for each sample.
        """
        if np.shape(X)[0] < self._k:
            raise ValueError('Not enough X points, have %d but want k of %d' % (np.shape(X)[0], self._k))
        
        self._data = np.append(X, np.reshape(y, (np.shape(X)[0], 1)), axis=1)
    
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
            Aggregation among k nearest neighbors for each row provided in X.
        
        """
        return np.apply_along_axis(self._compute_and_aggregate_knn, axis=1, arr=X)
    
    def _compute_and_aggregate_knn(self, row):
        """
        Calculate the k nearest neighbors using self._distance_function,
        then determine the estimate for the row using self._aggregate.
        
        Parameters
        ---------
        
        row : array-like, shape [n_features]
            Single row containing all the required features.
        """
        def get_distance(data_row):
            return self._distance_function(row, data_row[:-1])
        # Note: This isn't the most efficient solution, but does work well.
        # Would probably be better to store the data according to differing
        # regions in input space, then look at close regions when given a
        # new element.
        closest_k = heapq.nsmallest(self._k, self._data, key=get_distance)
        return self._aggregate(closest_k)
    
    def _aggregate(self, closest_k):
        """
        Combines the y value among the k cloesest neighbors using self._aggregation.
        
        Parameters
        ---------
        
        closest_k : array-like, shape [k, n_features + 1]
            Closest k elements from training dataset. Must have the y value as
            last element for each row.
        """
        return self._aggregation([row[-1] for row in closest_k])



def _euclidian_distance(a, b):
    return np.linalg.norm(a - b)

class KNN_Regression(KNN):
    """
    Will run KNN and aggregate using the average of the y values selected.
    
    Parameters
    --------
    
    k : integer
        Number of closest examples to aggregate over.
    
    aggregation : function
        Combines the K closest y values together.
        This can be mean, median, or any other function.
        Defaults to the mean of the values.
        
    distance_function : function
        Function to calculate the distance between two points.
        First argument will be current row, second will be row comparing against.
        Defaults to euclidian distance.
    
    """
    def __init__(self, k, aggregation=np.average, distance_function=_euclidian_distance):
        super().__init__(k=k, aggregation=aggregation, distance_function=distance_function)
        

class KNN_Classification(KNN):
    """
    Will run KNN and aggregate using the mode.
    
    Parameters
    --------
    
    k : integer
        Number of closest examples to aggregate over.
    
    aggregation : function
        Combines the K closest y values together.
        This can be mean, median, or any other function.
        Defaults to the most common value.
        
    distance_function : function
        Function to calculate the distance between two points.
        First argument will be current row, second will be row comparing against.
        Defaults to euclidian distance.
    """
    def __init__(self, k, aggregation=None, distance_function=_euclidian_distance):
        if aggregation is None:
            aggregation = KNN_Classification._most_common
        
        super().__init__(k=k, aggregation=aggregation, distance_function=distance_function)
    
    @staticmethod
    def _most_common(values):
        
        # Only want the value of first element provided
        return Counter(values).most_common(n=1)[0][0]


def run_regression_example():
    # Just using one feature to make it graphable
    X, y = datasets.make_regression(n_samples=200, n_features=1, bias=150, noise=4)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_proportion=0.2)
    
    reg = KNN_Regression(k=4)
    
    reg.fit(X_train, y_train)
    
    y_pred = reg.predict(X_test)
    
    mse = mean_square_error(y_pred, y_test)
    
    plt.scatter(X_test, y_test, color="Black", label="Actual")
    plt.scatter(X_test, y_pred, color="Red", label="Prediction")
    plt.legend(loc='lower right', fontsize=8)
    plt.title("KNN Regression (%.2f MSE)" % mse)
    plt.show()


def run_classification_example():
    # Just using one feature to make it graphable
    X, y = datasets.make_classification(n_samples=200, n_features=1, n_informative=1, n_redundant=0,
                                        n_clusters_per_class=1, flip_y=0.1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_proportion=0.2)
    
    reg = KNN_Classification(k=4)
    
    reg.fit(X_train, y_train)
    
    y_pred = reg.predict(X_test)
    
    mse = mean_square_error(y_pred, y_test)
    
    plt.scatter(X_test, y_test, color="Black", label="Actual")
    plt.scatter(X_test, y_pred, color="Red", label="Prediction")
    plt.legend(loc='lower right', fontsize=8)
    plt.title("KNN Regression (%.2f MSE)" % mse)
    plt.show()

if __name__ == "__main__":
    run_regression_example()
    
    run_classification_example()