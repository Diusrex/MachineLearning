import numpy as np
import tensorflow as tf

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")

from optimization_algorithms.optimizer import Optimizer
from util.data_operation import logistic_function

class LogisticRegressionTF(object):
    """
    Standard Logistic Regression classifier using maximum likelihood cost function.
    
    Parameters
    --------
    num_iterations : integer
        Specifies the number of iterations of gradient descent to be performed.
    
    learning_rate : numeric
        Determines what speed the gradient descent will update the weights.
        A too high or too low value may cause the gradient descent to not
        converge.
    
    classification_boundary : numeric or None
        If provided, predict will classify samples given the boundary -
        so points with probability >= classification_boundary will be class 1,
        everything else will be class 0.
        If None, predict will return the probability of the sample being class 1.
        
    Theory
    --------
        - Highly dependent on decision boundary being linear combination of \
        provided feature set (which may not be a linear combination of original \
        feature set).
            - Decision boundary is where dot(x, theta) = 0.
        - Has low variance and high bias
            - More features it has, the more this shifts to high variance low bias.
            - To reduce overfitting it may be helpful to prune the feature set or use\
            regularization.
        - Easy to understand the effect of a single or set of features on output
            - Look at their weight. If >0, then increases likelihood, otherwise decreases
        - Just supports binary features (0 or 1).
            - To have multiple classification, create + train a LogisticRegressionTF \
            for each class, determining how likely it is to occur. Then, for each new \
            input, class is the most likely class.
    """
    def __init__(self, num_iterations=2500, learning_rate=0.001, classification_boundary = None):
        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self._num_iterations = num_iterations
        self._classification_boundary = classification_boundary
        self._sess = None
        
    def set_classification_boundary(self, classification_boundary):
        """
        If classification_boundary is numeric, predict will classify samples
        given the boundary - so points with
        probability >= classification_boundary will be class 1,
        everything else will be class 0.
        If classification_boundary is None, predict will return the probability
        of the sample being class 1.
        """
        self._classification_boundary = classification_boundary
        
    def fit(self, X, y):
        """
        Fit internal parameters to minimize MSE on given X y dataset.
        
        Parameters
        ---------
        
        X : array-like, shape [n_samples, n_features]
            Input array of features.
            
        y : array-like, shape [n_samples,] or [n_samples,n_values]
            Input array of expected results. Must be binary (0 or 1)
        """
        if len(y.shape) == 1:
            y = y.reshape((y.size, 1))
        
        train = self._create_train_graph(X.shape[1], y.shape[1])
        
        # Clean up the previous session if it existed.
        if self._sess is not None:
            self._sess.close()
        
        self._sess = tf.Session(graph=self._graph)
        with self._graph.as_default():
            init = tf.global_variables_initializer()
            self._sess.run(init)
            
            summary_writer = tf.summary.FileWriter('train', self._sess.graph)
            summaries_tensor = tf.summary.merge_all()
            for step in range(self._num_iterations):
                summary_value, _ = self._sess.run([summaries_tensor, train], {self._X:X, self._y:y})
                summary_writer.add_summary(summary_value, global_step=step)
        
    def predict(self, X):
        """
        Predict the value(s) associated with each row in X.
        
        X must have the same size for n_features as the input this instance was
        trained on.
        
        Parameters
        ---------
        
        X : array-like, shape [n_samples, n_features]
            Input array of features.
            
        Returns
        ---------
        Values in range [0, 1] indicating how sure the predictor is that a value
        is 1. To choose most likely, use np.round.
        """
        values = self._sess.run(self._logistic_model, {self._X:X})
        
        if self._classification_boundary is None:
            return values
        
        return values >= self._classification_boundary
    
    def classification_weight(self, X):
        """
        Rill return a weight in range -inf to inf of how sure the ML algorithm
        is that the sample was class 1 (positive) or class 0 (negative).
        
        Parameters
        ---------
        
        X : array-like, shape [n_samples, n_features]
            Input array of features.
            
        Returns
        ---------
        Values ranging from -inf to +inf. If a sample is negative, would be classified
        as class 0, and positive means would be classified as class 1. The greater the
        magnitude, the more confident the ml would be.
        """
        return self._sess.run(self._class_weight, {self._X:X})
    
    def _create_train_graph(self, num_features, num_outputs_per_sample):
        """
        Create the training graph, given the number of features and the number of
        wanted outputs per sample.
        
        Will also setup all summary creation.
        
        Parameters
        ---------
        num_features : numeric
            Number of features each sample should contain.
        
        num_outputs_per_sample : numeric
            How many different outputs are wanted per sample.
        """
        self._graph = tf.Graph()
        with self._graph.as_default():
            
            self._X = tf.placeholder(tf.float32, shape=[None, num_features], name="X")
            self._w = tf.Variable(tf.zeros([num_features, num_outputs_per_sample]),
                                  dtype=tf.float32, name="w")
            self._b = tf.Variable(0, dtype=tf.float32, name="b")
            
            self._y = tf.placeholder(tf.float32, shape=[None, num_outputs_per_sample], name="y")
            
            self._class_weight = tf.add(tf.matmul(self._X, self._w), self._b, name="class_weight")
            self._logistic_model = tf.sigmoid(self._class_weight, name="inference")
            
            # Use reduce_mean since want to divide by # of items
            #regularization = tf.nn.l2_loss(self._w, name="l2_reg")
            self._loss = -tf.reduce_mean(tf.multiply(self._y, tf.log(self._logistic_model)) +
                                     tf.multiply(1 - self._y, tf.log(1 - self._logistic_model)))
            
            self._setup_summaries()
            
            return self._optimizer.minimize(self._loss)
    
    def _setup_summaries(self):
        """
        Some simple summary. The usefulness can vary, but I do believe they are
        a good start.
        """
        with tf.name_scope('summaries'):
            tf.summary.scalar("loss", self._loss)
            
            tf.summary.scalar('weight_mean', tf.reduce_mean(self._w))
            tf.summary.scalar('max_weight', tf.reduce_max(self._w))
            tf.summary.scalar('min_Weight', tf.reduce_min(self._w))
            tf.summary.histogram('weights_histogram', self._w)
    
    def __del__(self):
        self._sess.close()
