import numpy as np

def mean_square_error(pred, actual):
    diff = pred - actual
    return np.dot(diff, diff) / diff.size

def accuracy(pred, actual):
    return np.mean(pred == actual)
