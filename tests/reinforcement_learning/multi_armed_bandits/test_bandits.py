# Could use better name!

import unittest
import random
import numpy as np
import numpy.testing as nptest

import reinforcement_learning.multi_armed_bandits.bandits as bandits

class MultiArmedBanditsTester(unittest.TestCase):
    # Check simple requirements - like the different bandits all have restart + reset work properly
    
    multi_armed_bandit = bandits.MultiArmedBandit([
            bandits.PercentageBandit(0.5), bandits.PercentageBandit(0.2),
            bandits.NormalDistributionBandit(0, 1),
            bandits.NormalDistributionBandit(5, 2),
            bandits.SequenceBandit([(0, 1), (2, 5), (bandits.PercentageBandit(0.2), 2)], True),
            bandits.SequenceBandit([(bandits.SequenceBandit([(0, 1), (1, 0)], True), 1), # Take once from this sequence bandit
                                    (1, 5),
                                    (bandits.PercentageBandit(0.2), 3)], False),
    ])
    
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
    
    def testAllFunctionsDontCrash(self):
        multi_armed_bandit = MultiArmedBanditsTester.multi_armed_bandit
        
        # Don't care about the values, just that it doesn't crash
        #multi_armed_bandit.expected_value_per_bandit()
        
        for bandit in multi_armed_bandit._bandits:
            # Also, just care that it returns some string
            self.assertTrue(isinstance(bandit.describe(), str))


# Class is a little more complex, so needs to have its expected_value tested.
class SequenceBanditTester(unittest.TestCase):
    def testExpectedValueRepeated(self):
        bandit = bandits.SequenceBandit([(0, 1), (3, 5), (bandits.PercentageBandit(0.2), 10)], False)
        
        # 0 once, 3 5 times, then expected 0.2 10 times
        expected = (0 * 1 + 3 * 5 + 0.2 * 10) / (1 + 5 + 10)
        
        self.assertAlmostEqual(bandit.expected_value(), expected)
    
    def testExpectedValueFinal(self):
        bandit = bandits.SequenceBandit([(0, 1), (3, 5), (bandits.PercentageBandit(0.2), 10)], True)
        
        # 0 once, 3 5 times, then expected 0.2 almost infinite times
        expected = (0 * 1 + 3 * 5 + 0.2 * bandits.SequenceBandit._final_weight) / (1 + 5 + bandits.SequenceBandit._final_weight)
        
        self.assertAlmostEqual(bandit.expected_value(), expected)
    
    def testExpectedValueLimitation(self):
        bandit = bandits.SequenceBandit(
            [(bandits.SequenceBandit([(0, 1), (1, 0)], True), 1), # Take once from this sequence bandit
             (3, 5),
             (bandits.PercentageBandit(0.2), 10)],
            False)
        
        # We would actually want: 0 * 1 + 1 * 5 + .2 * 3. But the inner sequence bandit isn't that smart
        
        # 0 once, 3 5 times, then expected 0.2 10 times
        expected = (1 * 1 + 3 * 5 + 0.2 * 10) / (1 + 5 + 10)
        
        self.assertAlmostEqual(bandit.expected_value(), expected)
