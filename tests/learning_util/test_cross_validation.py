import unittest
import numpy as np
import numpy.testing as nptest

from learning_util.cross_validation import cross_validation


def generate_data(num_samples):
    """
    X data is 2-d, in form: [row_num, 0]
    
    y data has form [row_num] for each entry.
    
    This ensures that all X and y entries are unique
    
    Returns
    --------
    (X y)
        Data with num_samples rows.
    """
    X = np.array([[i, 0] for i in range(num_samples)])
    y = np.array([i for i in range(num_samples)])
    
    return X, y
        
class MockMLAlgorithm(object):
    def __init__(self, unit_test_class, all_X_data, all_y_data,
                 num_folds, had_reordering=True):
        self._unit_test_class = unit_test_class
        self._num_samples = len(all_y_data)
        self._all_X_data = all_X_data
        self._all_y_data = all_y_data
        self._num_folds = num_folds
        self._had_reordering = had_reordering
        self._fold_base_size = int(np.ceil(self._num_samples / self._num_folds))
        
        self._test_fold_index = 0
        
        self._expected_fold_X_data = None
        
        
    def fit(self, X, y):
        """
        Will ensure that the X and y data provided correspond, and update expectations for
        what data will be provided in expected fold X data
        """
        missing_data = np.ones((self._num_samples,), dtype=bool)
        # Must have same # rows.
        self.assertEqual(X.shape[0], y.shape[0])
        
        test_fold_start = self._getTestFoldStart()
        
        for i in range(len(y)):
            found_match = np.where(np.all(self._all_X_data==X[i, :], axis=1))
            # Returns an array of arrays.
            index = found_match[0][0]
            
            if not self._had_reordering:
                # Ensure that it was placed into where we would expect with no shuffle.
                expected_location = index
                if index >= test_fold_start:
                    expected_location -= self._fold_base_size
                
                self.assertEqual(i, expected_location)
            
            # Ensure the y data matches
            self.assertEqual(y[i],
                             self._all_y_data[index])
            
            missing_data[index] = False
        
        self._expected_fold_X_data = self._all_X_data[missing_data]
    
    def predict(self, X):
        # Check has correct dimensions
        nptest.assert_array_equal(self._expected_fold_X_data.shape,
                               X.shape)
        
        missing = np.ones(X.shape[0], dtype=bool)
        
        # Check that missing expected fold X data is provided.
        for i in range(X.shape[0]):
            found_match = np.where(np.all(self._expected_fold_X_data==X[i, :], axis=1))
            # Returns an array of arrays
            index = found_match[0][0]
            
            missing[index] = False
        
        nptest.assert_array_equal(missing,
                                  np.zeros(X.shape[0], dtype=bool))
        
        self._test_fold_index += 1
        
    def _getTestFoldStart(self):
        return self._fold_base_size * self._test_fold_index
    
    def assertCorrectNumFolds(self):
        self.assertEqual(self._test_fold_index, self._num_folds)
        
    def assertEqual(self, actual, expected):
        self._unit_test_class.assertEqual(actual, expected)

    def error_function(self, pred, actual):
        # Will have been incremented by predict.
        return self._test_fold_index - 1
        
class CrossValidationTests(unittest.TestCase):
    def testSanity(self):
        num_samples = 200
        num_folds = 5
        X, y = generate_data(num_samples)
        
        ml_algorithm = MockMLAlgorithm(self, X, y, num_folds)
        
        errors = cross_validation(ml_algorithm, X, y,
                                 ml_algorithm.error_function,
                                 num_folds=num_folds)
        
        nptest.assert_array_equal(errors,
                                  [0, 1, 2, 3, 4])
        
        ml_algorithm.assertCorrectNumFolds()

    def testWorksCorrectlyUnevenData(self):
        # num_samples not divisible by number of folds.
        num_samples = 200
        num_folds = 7
        X, y = generate_data(num_samples)
        
        ml_algorithm = MockMLAlgorithm(self, X, y, num_folds)
        
        errors = cross_validation(ml_algorithm, X, y,
                                 ml_algorithm.error_function,
                                 num_folds=num_folds)
        
        nptest.assert_array_equal(errors,
                                  [0, 1, 2, 3, 4, 5, 6])
        
        ml_algorithm.assertCorrectNumFolds()
        
    def testNoShuffle(self):
        num_samples = 200
        num_folds = 5
        X, y = generate_data(num_samples)
        
        ml_algorithm = MockMLAlgorithm(self, X, y, num_folds, had_reordering=False)
        
        cross_validation(ml_algorithm, X, y,
                                 ml_algorithm.error_function,
                                 num_folds=num_folds,
                         reorder_data=False)
        
        ml_algorithm.assertCorrectNumFolds()
