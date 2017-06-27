import unittest
import numpy as np

from util.data_operation import mean_square_error

class BasicOperationsTests(unittest.TestCase):
    
    def testMSE(self):
        X = np.fromstring('1 2 3 4', dtype=int, sep=" ")
        Y = np.fromstring('2 5 3 7', dtype=int, sep=" ")
        # Difference: 1 3 0 2
        
        self.assertEqual(mean_square_error(X, X), 0)
        self.assertEqual(mean_square_error(X, Y),
                         19/4)
