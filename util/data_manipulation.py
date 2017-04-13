import numpy as np

from abc import ABC, abstractmethod

class BaseScaler(ABC):
    """
    Performs some form of normalization on every provided column, with scaling of the form
        x' = (x - constant_adjustment) / factor_adjustment
    """
    def __init__(self):
        self._constant_adjustment = None
        self._factor_adjustment = None
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def fit(self, X):
        ncols = X.shape[1]
        
        self._constant_adjustment = np.zeros(ncols)
        self._factor_adjustment = np.zeros(ncols)
        for column, index in zip(X.T, range(ncols)):
            self._constant_adjustment[index] = self._calculate_constant_adjustment(column)
            self._factor_adjustment[index] = self._calculate_factor_adjustment(column)
        
    @abstractmethod
    def _calculate_constant_adjustment(self, column):
        """
        Column is reduced by this factor before doing factor adjustment.
        
        Returns
        --------
            Value to subtract column by when rescaling.
        """
        pass
    
    @abstractmethod
    def _calculate_factor_adjustment(self, column):
        """
        Column is divided by this factor after being subtracted by constant adjustment.
        
        Returns
        --------
            Value to divide column by when rescaling.
        """
        pass
    
    def transform(self, X):
        return np.divide(X - self._constant_adjustment, self._factor_adjustment)
    
    def restore(self, X):
        return np.multiply(X, self._factor_adjustment) + self._constant_adjustment
    
    
class StandardizationScaler(BaseScaler):
    """
    Will normalize all provided columns to have mean 0 and normal standard deviation
    
    Calculation is:
        x' = (x - mean) / standard_deviation
    
    """
        
    def _calculate_constant_adjustment(self, column):
        return np.mean(column)
    
    def _calculate_factor_adjustment(self, column):
        return np.std(column)


class RangeScaler(BaseScaler):
    """
    Will transform all variables to be in the range [0, 1].
    
    Calculation is:
        x' = (x - min(x)) / (max(x) - min(x))
    
    """
    def _calculate_constant_adjustment(self, column):
        return np.min(column)
    
    def _calculate_factor_adjustment(self, column):
        return np.max(column) - np.min(column)

class UnitLengthScaler(BaseScaler):
    """
    Will normalize all provided columns to be unit length.
        Or, ||x|| = 1
    
    Calculation is:
        x' = x / ||x||
    
    """
    def _calculate_constant_adjustment(self, column):
        # Only divide column, not subtracting anything
        return 0
    
    def _calculate_factor_adjustment(self, column):
        return np.linalg.norm(column)