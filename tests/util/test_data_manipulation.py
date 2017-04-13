import unittest
import numpy as np
import numpy.testing as nptest

from math import sqrt
from util.data_manipulation import RangeScaler, StandardizationScaler, UnitLengthScaler

# Ensure it transforms the matrix correctly
class StandardizationScalerTests(unittest.TestCase):
    def testFitAndRestore(self):
        scaler = StandardizationScaler()
        X = np.matrix('1 2; 3 4')
        X_trans = scaler.fit_transform(X)
        
        nptest.assert_allclose(X_trans, np.matrix('-1 -1; 1 1'))
        
        X_restored = scaler.restore(X_trans)
        nptest.assert_allclose(X_restored, X)
    
    def testFitMultipleAndRestore(self):
        scaler = StandardizationScaler()
        X = np.matrix('1; 3')
        scaler.fit(X)
        
        nptest.assert_allclose(scaler.transform(X), np.matrix('-1; 1'))
        
        # Mean of 5.5 -> 6.5; -2.5; -5.5; 1.5
        # Standard Deviation of 4.5
        X_new = np.matrix('12; 3; 0; 7')
        X_new_trans = scaler.fit_transform(X_new)
        
        nptest.assert_allclose(X_new_trans, np.matrix([[13/9], [-5/9], [-11/9], [1/3]]))
        
        # After fitting on a different array, gives a different answer
        nptest.assert_allclose(scaler.transform(X), np.matrix([[-1], [-5/9]]))


# Ensure it transforms the matrix correctly
class RangeScalerTests(unittest.TestCase):
    def testFitAndRestore(self):
        scaler = RangeScaler()
        X = np.matrix('1 2; 3 4; 5 10')
        X_trans = scaler.fit_transform(X)
            
        nptest.assert_allclose(X_trans, np.matrix('0 0; 0.5 0.25; 1 1'))
        
        X_restored = scaler.restore(X_trans)
        nptest.assert_allclose(X_restored, X)
    
    def testFitMultipleAndRestore(self):
        scaler = RangeScaler()
        # Min = 1, Max = 5
        X = np.matrix('1; 3; 5')
        scaler.fit(X)
        
        nptest.assert_allclose(scaler.transform(X), np.matrix('0; 0.5; 1'))
        
        # Min = -2, Max = 6
        X_new = np.matrix('-2; 0; 3; 6')
        X_new_trans = scaler.fit_transform(X_new)
        
        nptest.assert_allclose(X_new_trans, np.matrix('0; 0.25; 0.625; 1'))
        
        # After fitting on a different array, gives a different answer
        nptest.assert_allclose(scaler.transform(X), np.matrix('0.375; 0.625; 0.875'))

# Ensure it transforms the matrix correctly
class UnitLengthScalerTests(unittest.TestCase):
    def testFitAndRestore(self):
        scaler = UnitLengthScaler()
        X = np.matrix('-1 1; 1 2')
        X_trans = scaler.fit_transform(X)
        
        nptest.assert_allclose(
                X_trans, np.matrix([[-1 / sqrt(2), 1 / sqrt(5)],
                                    [ 1 / sqrt(2), 2 / sqrt(5)]]))
        
        X_restored = scaler.restore(X_trans)
        nptest.assert_allclose(X_restored, X)
    
    def testFitMultipleAndRestore(self):
        scaler = UnitLengthScaler()
        # ||X|| = 2
        X = np.matrix('-1; 1')
        scaler.fit(X)
        
        nptest.assert_allclose(
                scaler.transform(X), np.matrix([[-1/sqrt(2)], [1/sqrt(2)]]))
        
        # ||X|| = 5
        X_new = np.matrix('3; 4; 0')
        X_new_trans = scaler.fit_transform(X_new)
        
        nptest.assert_allclose(X_new_trans, np.matrix('0.6; 0.8; 0'))
        
        # After fitting on a different array, gives a different answer
        nptest.assert_allclose(scaler.transform(X), np.matrix('-0.2; 0.2'))

