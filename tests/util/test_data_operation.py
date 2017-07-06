import unittest
import numpy as np
import numpy.testing as nptest

from util.data_operation import mean_square_error, logistic_function

class BasicOperationsTests(unittest.TestCase):
    
    def testMSE(self):
        X = np.fromstring('1 2 3 4', dtype=int, sep=" ")
        Y = np.fromstring('2 5 3 7', dtype=int, sep=" ")
        # Difference: 1 3 0 2
        
        self.assertEqual(mean_square_error(X, X), 0)
        self.assertEqual(mean_square_error(X, Y),
                         19/4)

    def testLogisticFunction(self):
        self.assertEqual(logistic_function(0), 0.5)
        
        nptest.assert_allclose(np.array([0, 0.5, 1]),
                               logistic_function(np.array([-100, 0, 100])),
                               atol=1e-9)
