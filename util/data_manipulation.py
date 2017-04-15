import numpy as np

from abc import ABC, abstractmethod

class BaseScaler(ABC):
    """
    Performs some form of normalization on every provided column, with scaling of the form
        x' = (x - constant_adjustment) / factor_adjustment
    """
    def __init__(self):
        self._constant_reduction = None
        self._factor_divisor = None
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def fit(self, X):
        ncols = X.shape[1]
        
        self._constant_reduction = np.zeros(ncols)
        self._factor_divisor = np.zeros(ncols)
        for index, column in enumerate(X.T):
            self._constant_reduction[index] = self._calculate_constant_reduction(column)
            self._factor_divisor[index] = self._calculate_factor_divisor(column)
        
    @abstractmethod
    def _calculate_constant_reduction(self, column):
        """
        Column is reduced by this factor before doing factor adjustment.
        
        Returns
        --------
            Value to subtract column by when rescaling.
        """
        pass
    
    @abstractmethod
    def _calculate_factor_divisor(self, column):
        """
        Column is divided by this factor after being subtracted by constant adjustment.
        
        Returns
        --------
            Value to divide column by when rescaling.
        """
        pass
    
    def transform(self, X):
        return np.divide(X - self._constant_reduction, self._factor_divisor)
    
    def restore(self, X):
        return np.multiply(X, self._factor_divisor) + self._constant_reduction
    
    
class StandardizationScaler(BaseScaler):
    """
    Will normalize all provided columns to have mean 0 and normal standard deviation
    
    Calculation is:
        x' = (x - mean) / standard_deviation
    
    """
        
    def _calculate_constant_reduction(self, column):
        return np.mean(column)
    
    def _calculate_factor_divisor(self, column):
        return np.std(column)


class RangeScaler(BaseScaler):
    """
    Will transform all variables to be in the range [range_min, range_max].
    
    Calculation is:
        x' = (x - min(x) + range_min) / (max(x) - min(x)) * (range_max - range_min)
    
    """
    def __init__(self, range_min=0, range_max=1):
        assert(range_min < range_max)
        self._range_min = range_min
        self._range_max = range_max
    
    def _calculate_constant_reduction(self, column):
        return np.min(column) - self._range_min * self._calculate_factor_divisor(column)
    
    def _calculate_factor_divisor(self, column):
        return (np.max(column) - np.min(column)) / (self._range_max - self._range_min)

class UnitLengthScaler(BaseScaler):
    """
    Will normalize all provided columns to be unit length.
        Or, ||x|| = 1
    
    Calculation is:
        x' = x / ||x||
    
    """
    def _calculate_constant_reduction(self, column):
        # Only divide column, not subtracting anything
        return 0
    
    def _calculate_factor_adjustment(self, column):
        return np.linalg.norm(column)