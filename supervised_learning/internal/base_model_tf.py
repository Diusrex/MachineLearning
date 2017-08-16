# This class should be inherited by all tf models.
# It provides a standard way to handle training, as well as a range of other
# options.
import tensorflow as tf

from abc import ABC, abstractmethod
from collections import namedtuple


# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../..")

from util.tensorflow_util import reset_tensorflow_flags


# Should name standard tf operations using vocal.<name>.
Vocab = namedtuple('Vocab', ['x', 'y', 'loss', 'inference', 'learning_rate', 'optimizer'])
vocab = Vocab('x', 'y', 'loss', 'inference', 'learning_rate', 'optimizer')

# Due to the IDE I use, need to reset flags before running anything. Otherwise,
# flags will still be redefined and will crash...
reset_tensorflow_flags()

# Some of the standard flags for a model
flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.01,
                   'Determines the rate that weights will be updated when training.')
flags.DEFINE_integer('num_iterations', 2000, 'Max number of iterations for training.')
flags.DEFINE_string('training_summary_dir', 'train',
                    'Directory training summary data is written to. Relative path.')
flags.DEFINE_integer('save_summary_every', 10,
                     'How many training steps should be done before summaries are saved again. '
                     'If value is 0, will not do any updates.')

FLAGS = flags.FLAGS

class BaseTFModelOptions(object):
    """
    All of the standard options for a tensorflow model. If not provided, will
    default to flags instead. The flags all have the exact same name as the
    parameters to this class.
    
    This should be imported in every file when BaseTFModel is imported
    
    Parameters
    --------
    learning_rate : numeric
        Determines the rate that weights will be updated when training.
        
    num_iterations : integer
        Max number of iterations for training.
        
    training_summary_dir : relative path
        Directory training summary data is written to.
    
    save_summary_every : integer
        How many training steps should be done before summaries are saved again.
        If value is 0, will not do any updates
    """
    def __init__(self, learning_rate=None, num_iterations=None, training_summary_dir=None,
                 save_summary_every=None):
        self.learning_rate = self._option_value(learning_rate, FLAGS.learning_rate)
        self.num_iterations = self._option_value(num_iterations, FLAGS.num_iterations)
        self.save_summary_every = self._option_value(save_summary_every,
                                                       FLAGS.save_summary_every)
        self.training_summary_dir = self._option_value(training_summary_dir,
                                                       FLAGS.training_summary_dir)
    
    def _option_value(self, passed_in, flag_value):
        """
        If there was a custom value provided in constructor, use that. Otherwise
        default to the flag.
        """
        if passed_in is None:
            return flag_value
        return passed_in

class BaseTFModel(ABC):
    """
    Base Tensorflow Model.
    
    For a model inheriting from this class, it must pass the BaseTFModelOptions
    along.
    
    Currently just defaults to GradientDescentOptimizer, will expand this
    in the future.
    
    Parameters
    --------
    options: BaseTFModelOptions
        All of the options that should be used. If not provided, all of the options
        will be specified by the flags.
    """
    def __init__(self, options):
        if options is None:
            options = BaseTFModelOptions()
        optimizer = tf.train.GradientDescentOptimizer(options.learning_rate)
        self._options  = options
        self._optimizer = optimizer
        self._sess = None
        
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
        
        # Clean up the previous session if it existed.
        if self._sess is not None:
            self._sess.close()
        
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._create_inference_graph(X.shape[1], y.shape[1])
            self._create_loss_graph(
                    self._get_tensor_by_op_name(vocab.inference),
                    y.shape[1])
            
            self._optimizer.minimize(self._get_tensor_by_op_name(vocab.loss), name=vocab.optimizer)
            
            self._setup_general_summaries()
        
        self._sess = tf.Session(graph=self._graph)
        with self._graph.as_default():
            init = tf.global_variables_initializer()
            self._sess.run(init)
            
            summary_writer = tf.summary.FileWriter(self._options.training_summary_dir,
                                                   self._sess.graph)
            summaries_tensor = tf.summary.merge_all()
            for step in range(self._options.num_iterations):
                self._sess.run(
                            self._get_operation_by_name(vocab.optimizer),
                            {self._get_tensor_by_op_name(vocab.x):X,
                             self._get_tensor_by_op_name(vocab.y):y})
                
                if self._options.save_summary_every > 0 and\
                    step % self._options.save_summary_every == 0:
                    summary_value = self._sess.run(
                            summaries_tensor,
                            {self._get_tensor_by_op_name(vocab.x):X,
                             self._get_tensor_by_op_name(vocab.y):y})
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
        Estimated class for each sample
        """
        return self._sess.run(
                self._get_tensor_by_op_name(vocab.inference),
                {self._get_tensor_by_op_name(vocab.x):X})
        
    
    @abstractmethod
    def _create_inference_graph(self, num_features, num_outputs_per_sample):
        """
        Create the model inference graph, given the number of features and the
        number of wanted outputs per sample. Will be called in the correct
        tf.Graph context.
        
        Can call _setup_weight_summary to ensure summaries for the weights
        are displayed.
        
        Should NOT create any loss or similar.
        
        Must setup: vocab.x, vocab.weights, vocab.inference
        
        Parameters
        ---------
        num_features : numeric
            Number of features each sample should contain.
        
        num_outputs_per_sample : numeric
            How many different outputs are wanted per sample.
        """
        pass
    
    def _setup_weight_summary(self, weight_tensor, weight_identifier='weight'):
        with tf.name_scope(weight_identifier + '_summaries'):
            tf.summary.scalar('mean', tf.reduce_mean(weight_tensor))
            tf.summary.scalar('max', tf.reduce_max(weight_tensor))
            tf.summary.scalar('min', tf.reduce_min(weight_tensor))
            tf.summary.histogram('histogram', weight_tensor)
    
    def _create_loss_graph(self, inference, num_outputs_per_sample):
        """
        Create the model training graph, given the inference graph for the model.
        Will be called in the correct tf.Graph context.
        
        Must setup vocab.y, vocab.loss.
        
        Defaults to L2 but can be overriden by child class.
        """
        y = tf.placeholder(tf.float32, shape=[None, num_outputs_per_sample], name="y")
        # TODO: Add regularization.
        return tf.reduce_mean(tf.squared_difference(self._get_tensor_by_op_name(vocab.inference),
                                                    y),
                              name=vocab.loss)
    
    def _setup_general_summaries(self):
        """
        Some simple summary. The usefulness can vary, but I do believe they are
        a good start.
        """
        with tf.name_scope('summaries'):
            # TODO: Graph the scalars using standard names
            tf.summary.scalar("loss", self._get_tensor_by_op_name(vocab.loss))
    
    def _get_tensor_by_op_name(self, tensor_name):
        return self._get_operation_by_name(tensor_name).outputs[0]
    
    def _get_operation_by_name(self, operation_name):
        return self._graph.get_operation_by_name(operation_name)
    
    def __del__(self):
        self._sess.close()
