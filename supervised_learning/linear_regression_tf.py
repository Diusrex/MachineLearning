import tensorflow as tf

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")

from supervised_learning.internal.base_model_tf import BaseTFModel, vocab, BaseTFModelOptions

class LinearRegressionTF(BaseTFModel):
    """
    Standard Least Squares Linear predictor implemented using tensor flow which
    will use tensorflows GradientDescentOptimizer to fit the provided data.
    
    
    Will add a bias term to X, and otherwise not alter the data at all.
    
    Parameters
    --------
    
    options: BaseTFModelOptions
        All of the options that should be used. If not provided, all of the options
        will be specified by the flags in internal.base_model_tf.
    
    Theory
    --------
        - Highly dependent on output being predicted by a linear combination of \
        provided feature set (which may not be a linear combination of original \
        feature set).
        - Has low variance and high bias
        - But, with a large enough set of features can become high variance and overfit.
        - To reduce overfitting it may be helpful to prune the feature set, use\
        regularization, or use additional inputs.
    """
    def __init__(self, classification_boundary = None, options = None):
        super().__init__(options=options)
    
    def _create_inference_graph(self, num_features, num_outputs_per_sample):
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
        X = tf.placeholder(tf.float32, shape=[None, num_features], name=vocab.x)
        w = tf.Variable(tf.zeros([num_features, num_outputs_per_sample]),
                              dtype=tf.float32, name="w")
        self._setup_weight_summary(w)
        
        b = tf.Variable(0, dtype=tf.float32, name="b")
        
        tf.add(tf.matmul(X, w), b,
               name=vocab.inference)
    
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
