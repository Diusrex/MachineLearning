import unittest

import optimization_algorithms.examples as examples

from tests.run_examples import run_all_examples_in_module

class LearningUtilExamplesTester(unittest.TestCase):
    def testAllExamples(self):
        run_all_examples_in_module(examples)
