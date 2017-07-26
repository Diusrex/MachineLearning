import cvxopt
import math
import numpy as np

# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")

# These two functions are just for stopping cvxopt from printing output.
# From https://stackoverflow.com/a/8391735.
stout_backup = sys.stdout
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = stout_backup

class SVM(object):
    """
    Using the kernel_function to possibly add additional features, will calculate
    the classification between two different classes using support vectors along the
    maximum margin.
    
    This implementation uses cvxopt, so you will need to have that installed to be able to run any of the samples.
    
    Parameters
    --------
    kernel_function : function(row, row)
        Some defaults are provided in Kernel class
        A valid kernel function for a SVM. Calling on any X dataset should generate
        a symmetric positive semi-definite matrix.
    
    C : None or numeric
        The value of the tradeoff paramater C. If it is none, solution is required
        to correctly separate the classes. The lower C is, the more emphasis is placed
        on the margin.
        Note that C must be > support_vector_min_weight
    
    support_vector_min_weight : numeric
        Required value assigned to an input row by the quadratic programming sovler
        for the row to be considered a support vector.
        Shouldn't be 0 due to fp errors (everything would be selected as a support vector) which
        would elimitate some of the advantages of SVM (sparsity).
    
    return_raw_values : boolean
        Should the predict function return raw values, where the value for a row being > 1 is definitely
        positive, < -1 is definitely negative, and in-between is less certain.
    
    Theory
    --------
        - Tries to find the maximum margin, such that the points are all seprated correctly.
        - Use C to trade-off between having using a large margin (low C) and correctly\
        labelling all of the points (high C).
        - Using the more complex kernel functions, able to re-map the inputs into very\
        high dimensionality without a signficiant increase in computation time.
        - Uses cvxopt to solve the equivalent quadratic programming problem.
        - Solution is sparse due to selecting a relatively limited number of rows to be the\
        support vectors + find the maximum margin.
        - Due to the sparsity aspect, overfitting is more limited than would be expected\
        when mapping into such large dimension spaces.
    
    Resources
    --------
        - Russell Greiner's CMPUT 466 SVM notes
            Core understanding of SVM
        - http://tullo.ch/articles/svm-py/
            Some clarifications and pointed out a good solver.
        - http://www.cs.cmu.edu/~guestrin/Class/10701-S07/Slides/kernels.pdf
            How to calculate the Kernel.
    """
    def __init__(self, kernel_function, C = None, support_vector_min_weight = 5e-7, return_raw_values = False):
        if C is not None and C < support_vector_min_weight:
            raise ValueError("C must be >= support_vector_min_weight, otherwise no support vectors will be selected.")
        
        self._kernel_function = kernel_function
        self._C = C
        self._support_vector_min_weight = support_vector_min_weight
        self._return_raw_values = return_raw_values
        
    def fit(self, X, y):
        """
        Fit internal parameters to minimize MSE on given X y dataset.
        
        Will add a bias term to X.
        
        If a solution was not found (generally means C = None while classes were
        non-separable) will raise an exception.
        
        Parameters
        ---------
        
        X : array-like, shape [n_samples, n_features]
            Input array of features.
            
        y : array-like, shape [n_samples,]
            Input array of expected results. Should only contain 0 or 1
        """
        # Transform y - so will be 1 or -1
        y = y * 2.0 - 1.0
        
        solver_ret = self._run_solver(X, y)
        
        # Didn't complete
        if solver_ret['status'] != 'optimal':
            raise AssertionError(
                    "Solver status was {0}, probably fixed by setting C to an integer".format(
                            solver_ret['status']))
        
        self._setup_for_prediction(X, y, solver_ret)
        
        
    def _run_solver(self, X, y):
        """
        Will setup the inputs to the solver, then run it.
        
        Note that normally the lambdas are referred to as alpha for SVM.
        
        Parameters
        ---------
        
        X : array-like, shape [n_samples, n_features]
            Input array of features.
            
        y : array-like, shape [n_samples,]
            Input array of expected results. Should only contain -1 or 1
        
        Returns
        ---------
        Result map from the cvxopt solver. Mostly care about 'x' (which is the lambdas)
        and 'status' values.
        """
        kernel_matrix = self._create_kernel_matrix(X)
        
        num_rows = X.shape[0]
        
        # P = kernel_matrix * y along columns * y along rows
        P = np.multiply(np.outer(y, y), kernel_matrix)
        P = cvxopt.matrix(P[:], (num_rows, num_rows))
        
        # Weight all the li equally
        q = cvxopt.matrix(-1.0, (num_rows, 1))
        
        # Sum of all li must be = 0
        A = cvxopt.matrix(y, (1, num_rows))
        b = cvxopt.matrix(0.0, (1, 1))
        
        # li >= 0
        G = -1 * np.identity(num_rows)
        h = np.zeros((num_rows, 1))
        
        if self._C is not None:
            # li <= C
            G_lt_C = np.identity(num_rows)
            h_lt_C = self._C * np.ones((num_rows, 1))
        
            G = np.vstack((G, G_lt_C))
            h = np.vstack((h, h_lt_C))
        
        
        G = cvxopt.matrix(G[:], G.shape)
        h = cvxopt.matrix(h[:], h.shape)
        
        # Block printing since the solver will output extra results.
        blockPrint()
        res = cvxopt.solvers.qp(P, q, G, h, A, b)
        enablePrint()
        
        return res
    
    def _create_kernel_matrix(self, X):
        """
        Generates a m * m matrix (m being number of samples in X), where 
        matrix at ij is the value of the kernel function called on samples i and j.
        
        Parameters
        ---------
        
        X : array-like, shape [n_samples, n_features]
            Input array of features.
        
        Returns
        ---------
        
        array-like, shape [n_samples, n_samples]
            Value of kernel function called on corresponding rows in X.
        """
        
        def kern_wrapper(index_1, index_2):
            """
            Given two indicies, will call the kernel function with the indicies.
            """
            return self._kernel_function(X[index_1,:], X[index_2,:])

        num_rows = X.shape[0]
        return np.fromfunction(np.vectorize(kern_wrapper), (num_rows, num_rows), dtype=int)
        
    def _setup_for_prediction(self, X, y, solver_ret):
        """
        Given the return from cvxopt will generate the support vectors and b.
        
        Parameters
        ---------
        
        X : array-like, shape [n_samples, n_features]
            Input array of features.
            
        y : array-like, shape [n_samples,]
            Input array of expected results. Should only contain -1 or 1
            
        solver_ret
            Return value from calling cvxopt to optimize the dual form.
        
        """
        # We only care about the support vectors
        lambdas = np.ravel(solver_ret['x'])
        
        support_vectors = lambdas > self._support_vector_min_weight
        
        self._support_vectors_X = X[support_vectors, :]
        self._support_vectors_y = y[support_vectors]
        self._support_vectors_lambdas = lambdas[support_vectors]
        
        self._num_support_vectors = self._support_vectors_X.shape[0]
        
        # Calculate b using the mean of the differences. Performs far better than calculating
        # from a single support vector.
        vals = []
        for x, y in zip(self._support_vectors_X, self._support_vectors_y):
            kernel_of_vec_and_w = self._calculate_support_vectors_times_row(x)
            vals.append(y - kernel_of_vec_and_w)
        self._b = np.mean(vals)
        
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
        
        # Will calculate the value given to each row. If > 0, row is positive class, otherwise is negative class
        internal_results =\
            np.apply_along_axis(self._calculate_support_vectors_times_row, axis=1, arr=X) + self._b
        
        if not self._return_raw_values:
            internal_results = np.sign(internal_results)
            # Transform from -1 or 1 to 0 or 1
            return (internal_results + 1) / 2
        
        # Leave in form where > 0 is positive, < 0 is negative, and < abs(1) means is not completely sure of result.
        return internal_results
    
    def _calculate_support_vectors_times_row(self, row):
        def multiply_by_support_vector(support_vector_index):
            return self._support_vectors_lambdas[support_vector_index] * self._support_vectors_y[support_vector_index] *\
                self._kernel_function(self._support_vectors_X[support_vector_index,:], row)
        
        res = np.fromfunction(np.vectorize(multiply_by_support_vector), (self._num_support_vectors,), dtype=int)
        
        return np.sum(res)
        
    def get_feature_params(self):
        
        return None

class Kernel(object):
    @staticmethod
    def linear_kernel():
        """
        Simple kernel, doesn't do any remapping into different feature spaces
        so will generate a linear boundary.
        
        Will cause the SVM to act like logistic regression, but still find the
        largest margin.
        """
        
        def calculate(u, v):
            return np.dot(u, v)
    
        return calculate
    
    @staticmethod
    def polynomial_d_only(d):
        """
        Remaps the input into polynomials of degree d.
        
        Parameters
        ---------
        
        d : numeric
            Degree to remap inputs to.
        """
        def calculate(u, v):
            return math.pow(np.dot(u, v), d)
    
    @staticmethod
    def polynomial_up_to_d(d):
        """
        Remaps the input into polynomials up to degreee d.
        
        Parameters
        ---------
        
        d : numeric
            Max degree to map inputs to.
        """
        def calculate(u, v):
            return math.pow(np.dot(u, v) + 1, d)
    
    @staticmethod
    def gaussian_kernel(sigma):
        """
        Essentially is a similarity metric which has an infinite number of dimensions!
        
        Each input point has its own gaussian 'blob' around it, which decreases
        with distance. As sigma is increased, the blob becomes smaller.
        
        Kinda acts like a nearest neighbor, but will train to get stronger/weaker
        influence by different points.
        
        Parameters
        ---------
        
        sigma : numeric
            How contained should the influence from each point be. Larger sigma is,
            more contained the influence is. Note that with a larger sigma it is
            possible to overfit.
        """
            
        def calculate(u, v):
            diff = u - v
            return np.exp(- np.dot(diff, diff) / (2 * sigma * sigma))
        
        return calculate
