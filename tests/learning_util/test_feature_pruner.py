# -*- coding: utf-8 -*-
"""
Created on Tue May  9 21:35:19 2017

@author: Morgan
"""
import unittest
import numpy as np
import numpy.testing as nptest

from learning_util.feature_pruner import FeaturePruner


class FeaturePrunerTests(unittest.TestCase):
    def testPositiveOnly(self):
        trainer = FeaturePrunerTests.SimpleTrainer([
                np.array([1, 0, 1]), # Remove index 1
                np.array([0.5, 0.75]), # Remove index 0
                ])
        pruner = FeaturePruner(trainer, 1) # features_to_reduce_to
        
        # Actual values provided don't matter
        pruner.fit(np.matrix([[0, 1, 2]]), None)
        
        kept = pruner.transform(np.matrix([0, 1, 2]))
        
        # It should only keep index 2
        nptest.assert_array_equal([[2]],
                                  kept)
    
    def testPositiveAndNedgative(self):
        trainer = FeaturePrunerTests.SimpleTrainer([
                np.array([1, 0, 1, -1, -3]), # Remove index 1
                np.array([0.5, 0.75, -2, -0.25]), # Remove index 4
                ])
        pruner = FeaturePruner(trainer, 3) # features_to_reduce_to
        
        # Actual values provided don't matter
        pruner.fit(np.matrix([[0, 1, 2, 3, 4]]), None)
        
        kept = pruner.transform(np.matrix([0, 1, 2, 3, 4]))
        
        # It should keep indicies 0, 2, 3
        nptest.assert_array_equal([[0, 2, 3]],
                                  kept)
    
    class SimpleTrainer():
        """
        Each time fit is called, will advance one index through the provided \
        weights.
        """
        def __init__(self, weights):
            self._index = -1 # Since the FeaturePruner starts off by first training
            self._weights = weights
            
        def fit(self, _, __):
            self._index += 1
            
        def get_feature_params(self):
            return self._weights[self._index]