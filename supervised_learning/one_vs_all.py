import numpy as np

class OneVsAllClassification(object):
    """
    Give a classifier that is ONLY able to predict how likely a binary classification is,
    predict which of N classes each element is in.
    
    Good example is the LogisticRegression classifier.
    
    Parameters
    --------
    classifier_constructor
        Function that, when called, returns a new instance of a learning algorithm
        with fit, predict, and copy functions.
    
    provide_likelihood : boolean
        In predict function, should the likelihood an example is part of the selected class
        be returned.
    
    Theory
    --------
    Will predict the likelihood for the N different classes, and select the class
    with the highest likelihood according to the classifier.
    
    Warning
    --------
    Do NOT use on classifiers like Neural Networks/Decision Trees which are
    natively able to handle multiple different classifications at once.
    """
    def __init__(self, classifier_constructor, provide_likelihood=False):
        self._classifier_constructor = classifier_constructor
        self._label_classifiers = None
        self._label_values = None
        self._provide_likelihood = provide_likelihood
        
    
    def fit(self, X, y):
        """
        Fit # unique y value classifiers (created using classifier_constructor)
        to the y data.
        
        Will not add a bias term to the X data, but many classifiers do or can.
        
        Parameters
        ---------
        
        X : array-like, shape [n_samples, n_features]
            Input array of features.
            
        y : array-like, shape [n_samples,] or [n_samples,n_values]
            Input array of expected results. Must be binary (0 or 1)
        """
        unique_classes = np.unique(y)
        self._label_classifiers = {sample_class: self._classifier_constructor()\
                                   for sample_class in unique_classes}
        
        for sample_class in self._label_classifiers:
            classifier = self._label_classifiers[sample_class]
            has_class = (y == sample_class)
            
            classifier.fit(X, has_class)
    
    def predict(self, X):
        """
        Will return the most likely class for each row in X.
        
        X must have the same size for n_features as the input this object was
        trained on.
        
        Parameters
        ---------
        
        X : array-like, shape [n_samples, n_features]
            Input array of features.
            
        Returns
        ---------
        Most likely class for each instance.
        
        If provide_likelihood was true, will also return the likelihood for each
        input row being part of its assigned class.
        """
        best_likelihood = np.zeros(X.shape[0])
        best_prediction = np.zeros(X.shape[0])
        for sample_class in self._label_classifiers:
            classifier = self._label_classifiers[sample_class]
            prob = classifier.predict(X)
            
            is_best = (prob > best_likelihood)
            best_likelihood[is_best] = prob[is_best]
            best_prediction[is_best] = sample_class
        
        if self._provide_likelihood:
            return (best_prediction, best_likelihood)
        
        return best_prediction
