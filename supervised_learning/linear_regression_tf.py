import tensorflow as tf
import numpy as np

# TODO: Parameters
# TODO: Copy comments from LR
# TODO: Allow normal equation?
class LinearRegressionTF(object):
    """
    Standard Least Squares Linear predictor implemented using tensor flow which
    will use tensorflows GradientDescentOptimizer to fit the provided data.
    
    
    Will add a bias term to X, and otherwise not alter the data at all.
    
    Parameters
    --------
    num_iterations : integer
        Specifies the number of iterations of gradient descent to be performed.
    
    learning_rate : numeric
        Determines what speed the gradient descent will update the weights.
        A too high or too low value may cause the gradient descent to not
        converge.
    
    Theory
    --------
        - Highly dependent on output being predicted by a linear combination of \
        provided feature set (which may not be a linear combination of original \
        feature set).
        - Has low variance and high bias
        - But, with a large enough set of features can become high variance and overfit.
        - To reduce overfitting it may be helpful to prune the feature set or use\
        regression.
    """
    def __init__(self, num_iterations=2500, learning_rate=0.001):
        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self._num_iterations = num_iterations
        self._sess = None
    
    def fit(self, X, y):
        """
        Fit internal parameters to minimize MSE on given X y dataset.
        
        Parameters
        ---------
        X : array-like, shape [n_samples, n_features]
            Input array of features.
            
        y : array-like, shape [n_samples,] or [n_samples,n_values]
            Input array of expected results. Can be 2 dimensional, if estimating
            multiple different values for each sample.
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
        """
        return self._sess.run(self._linear_model, {self._X:X})
    
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
            print(num_outputs_per_sample)
            self._w = tf.Variable(tf.ones([num_features, num_outputs_per_sample]),
                                  dtype=tf.float32, name="w")
            self._b = tf.Variable(0, dtype=tf.float32, name="b")
            
            self._y = tf.placeholder(tf.float32, shape=[None, num_outputs_per_sample], name="y")
            
            self._linear_model = tf.add(tf.matmul(self._X, self._w), self._b,
                                        name="inference")
            
            self._loss = tf.reduce_sum(tf.squared_difference(self._linear_model, self._y),
                                       name="loss")
            
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
