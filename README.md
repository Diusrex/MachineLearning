# Machine Learning Implementation

I am using this repository to practice implementing different machine learning algorithms and improve my understanding of these algorithms.

These implementation should NOT be used in an actual project, since they are not optimized for speed or handling a large number of features.

## Installation

Requires Python 3, with modules:
- numpy
- matplotlib
- sklearn (for datasets)
- [cvxopt](http://cvxopt.org/) - convex optimization library (only required for SVM, tests will still pass if not included)

## Running tests

Run all tests using `python -m unittest discover` in base directory of repository.

Do note that many of the tests will create graphs, so it can be fairly ugly to watch...

### Algorithm Philosopy
* Keep them as simple as possible!
   * Any additional data transformation should be handled by external classes (like RangeScaler).
   * Regularization should be handled by the optimizer (when possible).
    
* Try to setup everything as building blocks as much as possible.
   * Lots of decorator pattern, since that is an easyish way to compose the algorithms
       * E.g: [Optimizer](https://github.com/Diusrex/Machine-Learning-Implementations/blob/master/optimization_algorithms/optimizer.py)
       can be wrapped with a [cost graph] to view the error over training iterations.
            
### Examples

Examples are in many different folders to show how you can use different algorithms or other classes.

All examples are constantly checked for correctness by running them in unit tests.

Each example file must include a main function, and the default implementation should be
very fast while not print out any unnecessary output.
    
### Documentation
Documentation should be developed according to the guidlines of [Numpydoc](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt).
    
