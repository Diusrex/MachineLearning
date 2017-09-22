# Could use better name!

import unittest
import random
import numpy as np
import numpy.testing as nptest

import reinforcement_learning.multi_armed_bandits.bandits as bandits

class MultiArmedBanditsTester(unittest.TestCase):
    # Check simple requirements - like the different bandits all have restart + reset work properly
    
    # TODO: Only need 1 of each bandit type once have more than one type.
    multi_armed_bandit = bandits.MultiArmedBandit((bandits.PercentageBandit(0.5),
                                                   bandits.PercentageBandit(0.2)))
    
    def testAllBanditsResetProperly(self):
        multi_armed_bandit = MultiArmedBanditsTester.multi_armed_bandit
        
        # Get about 1000 steps. Then reset, and ensure is identical
        num_steps = 1000
        results = []
        multi_armed_bandit.reset()
        for i in multi_armed_bandit.actions():
            result = [multi_armed_bandit.pull_bandit(i) for _ in range(num_steps)]
            results.append(result)
        
        multi_armed_bandit.reset()
        
        # Changed in place
        actions = multi_armed_bandit.actions()
        random.shuffle(actions)
        # Run them in a different order
        for i in actions:
            expected = results[i]
            actual = [multi_armed_bandit.pull_bandit(i) for _ in range(num_steps)]
            
            nptest.assert_allclose(actual, expected)
                                                  
    
    def testAllBanditsRegenerateProperly(self):
        multi_armed_bandit = MultiArmedBanditsTester.multi_armed_bandit
        
        # Get about 1000 steps. Then regenerate, and ensure has changed
        num_steps = 1000
        results = []
        multi_armed_bandit.reset()
        for i in multi_armed_bandit.actions():
            result = [multi_armed_bandit.pull_bandit(i) for _ in range(num_steps)]
            results.append(result)
            
        multi_armed_bandit.regenerate()
        
        # Changed in place
        actions = multi_armed_bandit.actions()
        random.shuffle(actions)
        # Run them in a different order
        for i in actions:
            expected = results[i]
            actual = [multi_armed_bandit.pull_bandit(i) for _ in range(num_steps)]
            
            self.assertFalse(np.allclose(expected, actual))