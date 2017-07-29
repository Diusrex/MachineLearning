import heapq
import numpy as np

from collections import Counter

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

    def classification_weight(self, X):
        """
        Rill return a weight in range -inf to inf of how sure the ML algorithm
        is that the sample was class 1 (positive) or class 0 (negative).
        
        Parameters
        ---------
        
        X : array-like, shape [n_samples, n_features]
            Input array of features.
            
        Returns
        ---------
        Values ranging from -inf to +inf. If a sample is negative, would be classified
        as class 0, and positive means would be classified as class 1. The greater the
        magnitude, the more confident the ml would be.
        """
        old_aggregation = self._aggregation
        
        self._aggregation = np.sum
        values = self.predict(X)
        self._aggregation = old_aggregation
        
        # Difference between values and k is number of 
        return (2 * values - self._k) / self._k