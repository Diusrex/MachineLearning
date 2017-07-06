import unittest

import supervised_learning.examples as examples

from tests.run_examples import run_all_examples_in_module

class SupervisedLearningExamplesTester(unittest.TestCase):
    def testAllExamples(self):
        run_all_examples_in_module(examples)
