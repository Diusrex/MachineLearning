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
            bandits.SequenceBandit([(0, 1), (2, 5), (bandits.PercentageBandit(0.2), 2)],
                                   True, expected_value_is_current=True),
            bandits.SequenceBandit([(0, 1), (2, 5), (bandits.PercentageBandit(0.2), 2)],
                                   True, expected_value_is_current=False),
            bandits.SequenceBandit([(bandits.SequenceBandit([(0, 1), (1, 0)], True), 1), # Take once from this sequence bandit
                                    (1, 5),
                                    (bandits.PercentageBandit(0.2), 3)],
                                   False, expected_value_is_current=True),
            bandits.SequenceBandit([(bandits.SequenceBandit([(0, 1), (1, 0)], True), 1), # Take once from this sequence bandit
                                    (1, 5),
                                    (bandits.PercentageBandit(0.2), 3)],
                                   False, expected_value_is_current=False),
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
    def testReturnsExpectedRewardsRepeat(self):
        # Stays on 2 for some time
        bandit = bandits.SequenceBandit([(0, 1), (3, 5), (bandits.PercentageBandit(1), 10)], repeat=True)
        
        # Run the full sequence 5 times.
        for sequence in range(5):
            # One pull of 0
            self.assertEqual(bandit.pull(), 0, "Sequence {}, step 0".format(sequence))
            # 5 pulls of 3
            for step in range(5):
                self.assertEqual(bandit.pull(), 3, "Sequence {}, step {}".format(sequence, step))
            
            # Now 10 pulls of 1
            for step in range(10):
                self.assertEqual(bandit.pull(), 1, "Sequence {}, step {}".format(sequence, step))
                
    def testReturnsExpectedRewardsNoRepeat(self):
        # Stays on 2 for some time
        bandit = bandits.SequenceBandit([(0, 1), (3, 5), (bandits.PercentageBandit(1), 10)], repeat=False)
        
        # One pull of 0
        self.assertEqual(bandit.pull(), 0, "step 0")
        # 5 pulls of 3
        for step in range(5):
            self.assertEqual(bandit.pull(), 3, "step {}".format(step))
        
        # Now large # of pulls for last item - ensure it doesn't cycle back.
        for step in range(10000):
            self.assertEqual(bandit.pull(), 1, "step {}".format(step))
        
        
    def testExpectedValueRepeatedUseAll(self):
        bandit = bandits.SequenceBandit([(0, 1), (3, 5), (bandits.PercentageBandit(0.2), 10)],
                                        True,
                                        expected_value_is_current=False)
        
        # 0 once, 3 5 times, then expected 0.2 10 times
        expected = (0 * 1 + 3 * 5 + 0.2 * 10) / (1 + 5 + 10)
        
        self.assertAlmostEqual(bandit.expected_value(), expected)
    
    def testExpectedValueFinalUseAll(self):
        bandit = bandits.SequenceBandit([(0, 1), (3, 5), (bandits.PercentageBandit(0.2), 10)],
                                        False,
                                        expected_value_is_current=False)
        
        # 0 once, 3 5 times, then expected 0.2 almost infinite times
        expected = (0 * 1 + 3 * 5 + 0.2 * bandits.SequenceBandit._final_weight) / (1 + 5 + bandits.SequenceBandit._final_weight)
        
        self.assertAlmostEqual(bandit.expected_value(), expected)
    
    
    def testExpectedValueFinalUseCurrentOnly(self):
        bandit = bandits.SequenceBandit([(0, 1), (3, 5), (bandits.PercentageBandit(0.2), 10)],
                                        repeat=True,
                                        expected_value_is_current=True)
        # Ensure can run through multiple times and get the same expected values
        for sequence in range(5):
            self.assertEqual(bandit.expected_value(), 0)
            # Now pull once to get onto second element
            bandit.pull()
            self.assertEqual(bandit.expected_value(), 3)
            # Now pull 5 times
            for _ in range(5):
                bandit.pull()
            
            self.assertEqual(bandit.expected_value(), 0.2)
            
            # Cycle back to the start
            for _ in range(10):
                bandit.pull()
    
    def testExpectedValueFinalUseCurrentOnlyReset(self):
        bandit = bandits.SequenceBandit([(0, 1), (3, 5), (bandits.PercentageBandit(0.2), 10)],
                                        False,
                                        expected_value_is_current=True)
        
        # Pull once to get onto the second element, then reset.
        bandit.pull()
        bandit.reset()
        
        self.assertEqual(bandit.expected_value(), 0)
        # Now pull once to get onto second element
        bandit.pull()
        self.assertEqual(bandit.expected_value(), 3)
        
    
    def testExpectedValueFinalUseCurrentOnlyRegenerate(self):
        bandit = bandits.SequenceBandit([(0, 1), (3, 5), (bandits.PercentageBandit(0.2), 10)],
                                        False,
                                        expected_value_is_current=True)
        
        # Pull once to get onto the second element, then regenerate.
        bandit.pull()
        bandit.regenerate()
        
        self.assertEqual(bandit.expected_value(), 0)
        # Now pull once to get onto second element
        bandit.pull()
        self.assertEqual(bandit.expected_value(), 3)
        
    
    def testExpectedValueLimitationUseAll(self):
        bandit = bandits.SequenceBandit(
            # Take once from this sequence bandit.
            [(bandits.SequenceBandit([(0, 1), (1, 0)], False, expected_value_is_current=False), 1),
             (3, 5),
             (bandits.PercentageBandit(0.2), 10)],
            True,
            expected_value_is_current=False)
        
        # We would actually want: 0 * 1 + 1 * 5 + .2 * 3. But the inner sequence bandit isn't that smart
        
        # 0 once, 3 5 times, then expected 0.2 10 times
        expected = (1 * 1 + 3 * 5 + 0.2 * 10) / (1 + 5 + 10)
        
        self.assertAlmostEqual(bandit.expected_value(), expected)
