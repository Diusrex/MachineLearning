import numpy as np
import math

def mean_square_error(pred, actual):
    diff = pred - actual
    return np.dot(diff, diff) / diff.size

def accuracy(pred, actual):
    return np.mean(pred == actual)

def logistic_function(val):
    return 1 / (1 + np.exp(-val))

def entropy(group_counts):
    """
    Returns the total non-negative information theory entropy,
    give the number of observations for each different group
    """
    total = sum(group_counts)
    entro = 0
    for item_count in group_counts:
        entro += item_entropy(item_count, total)
    return entro

def item_entropy(item_count, total_count):
    """
    Returns the non-negative information theory entropy for a single item,
    give the number of observations of this item and the total number of
    observations.
    """
    # Two cases where the entropy is 0
    if item_count == total_count or item_count == 0:
        return 0
    
    item_prob = 1.0 * item_count / total_count
    return -item_prob * math.log(item_prob)
