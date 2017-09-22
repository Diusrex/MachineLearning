import unittest

import reinforcement_learning.multi_armed_bandits.examples as examples

from tests.run_examples import run_all_examples_in_module

class MultiArmedBanditsExamplesTester(unittest.TestCase):
    def testAllExamples(self):
        run_all_examples_in_module(examples)
