import tensorflow as tf

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")

from supervised_learning.internal.base_model_tf import BaseTFModel, vocab

class LogisticRegressionTF(BaseTFModel):
    """
    Standard Logistic Regression classifier using maximum likelihood cost function.
    
    Parameters
    --------
    classification_boundary : numeric or None
        If provided, predict will classify samples given the boundary -
        so points with probability >= classification_boundary will be class 1,
        everything else will be class 0.
        If None, predict will return the probability of the sample being class 1.
    
    options: BaseTFModeOptions
        All of the options that should be used. If not provided, all of the options
        will be specified by the flags in internal.base_model_tf.
        
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
    def __init__(self, classification_boundary = None, options = None):
        super().__init__(options=options)
        self._classification_boundary = classification_boundary
        
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
        values = super().predict(X)
        
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
        return self._sess.run(
                self._get_tensor_by_name("class_weight"),
                {self._get_tensor_by_name(vocab.x):X})
    
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
        
        class_weight = tf.add(tf.matmul(X, w), b, name="class_weight")
        tf.sigmoid(class_weight, name=vocab.inference)
        
    def _create_loss_graph(self, inference, num_outputs_per_sample):
        """
        Create the model training graph, given the inference graph for the model.
        Will be called in the correct tf.Graph context.
        
        Must setup vocab.y, vocab.loss.
        
        Defaults to L2 but can be overriden by child class.
        """
        y = tf.placeholder(tf.float32, shape=[None, num_outputs_per_sample], name=vocab.y)
        
        return tf.reduce_mean(-tf.multiply(y, tf.log(inference)) -
                                 tf.multiply(1 - y, tf.log(1 - inference)),
                                 name=vocab.loss)
        
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
