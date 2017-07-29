import matplotlib.pyplot as plt
import numpy as np

def class_estimation_graph(num_class, X, y, y_pred, formatted_title):
    """
    Given 2-d grid, as well as the true + estimated class for each sample in grid,
    create a plot to make it easy to see how good the estimation was.
    
    Parameters
    --------
    num_class : integer
        Number of classes. Assumes classes are labelled from 0 to num_class - 1
    
    X : array-like, shape [n_samples, 2]
        x1 x2 coordinates for each sample.
        
    y : array-like, shape [n_samples,]
        Expected class for each sample.
    
    y_pred : array-like, shape [n_samples,]
        Predicted class for each sample.
    
    formatted_title
        Title to display for plot.
    """
    true_class_shapes = ['o', 's', '*', 'v', '8']
    estimate_class_color = ['Blue', 'Green', 'Pink', 'Grey', 'Red']
    if num_class > len(true_class_shapes):
        raise ValueError("Only able to handle up to %d different classes" % (len(true_class_shapes)))
    
    if X.shape[1] != 2:
        raise ValueError("Only able to handle dimension 2 data.")
    
    plt.figure()
    
    for true_class in range(num_class):
        true_class_indicies = (y == true_class)
        for estimated_class in range(num_class):
            estimated_class_indicies = (y_pred == estimated_class)
            
            X_indicies = X[true_class_indicies & estimated_class_indicies]
            
            label = None
            if true_class == estimated_class:
                label = "Correct shape + color class " + str(true_class)
            
            if len(X_indicies) > 0:
                plt.scatter(X_indicies[:, 0], X_indicies[:, 1], marker=true_class_shapes[true_class],
                            color=estimate_class_color[estimated_class], label = label)
    
    plt.legend(fontsize=8)
    plt.title(formatted_title)
    plt.show()


def decision_boundary_graph(X, y, ml_algorithm, formatted_ml_info, points_per_dimension=100):
    """
    Given the 2-D X data, will output the decision boundary.
    
    Parameters
    ----------
    X : array-like, shape [n_samples, 2]
        x1 x2 coordinates for each sample.
        
    y : array-like, shape [n_samples,]
        Expected class for each sample.
        
    ml_algorithm
        Class that supports predict and classification_weight. Predict should return
        0 or 1, while classification_weight should return a positive value for class 1
        and negative value for class 0.
        
    formatted_ml_info : string
        Title to describe what the ml algorithm is, as well as any other wanted details.
    
    """
    true_class_shapes = ['o', 'v']
    class_grid_color = ['Red', 'Blue']
    
    # Give a little extra space for the min, just to give a better view
    min_dim = np.floor(np.min(X, axis=0)) - 1
    max_dim = np.ceil(np.max(X, axis=0)) + 1
    
    f, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim([min_dim[0],max_dim[0]])
    ax.set_ylim([min_dim[1],max_dim[1]])
    
    x_vec = np.linspace(min_dim[0], max_dim[0], num=points_per_dimension)
    y_vec = np.linspace(min_dim[1], max_dim[1], num=points_per_dimension)
    
    # Generate the Z graph from x1 x2 pairs.
    Z = ml_algorithm.classification_weight(
            np.array([[x1, x2] for x2 in y_vec for x1 in x_vec]))
    
    # Need to reshape it to be 2D instead of 1D
    Z.shape = (100, 100)
    
    ax.contour(x_vec, y_vec,
               Z,
               levels=[0], cmap="Greys_r")
    
    # Outside of [-2, 2] most algorithms will be confident
    Z = np.maximum(Z, -2)
    Z = np.minimum(Z, 2)
    ax.contourf(x_vec, y_vec,
               Z, levels=[-2, -0.99, -0.5, 0, 0.5, 0.99, 2],
               cmap="RdBu")
    
    y_pred = ml_algorithm.predict(X)
    
    
    for true_class in range(2):
        true_class_indicies = (y == true_class)
        for estimated_class in range(2):
            estimated_class_indicies = (y_pred == estimated_class)
            
            X_indicies = X[true_class_indicies & estimated_class_indicies]
            
            label = None
            if true_class == estimated_class:
                label = "Correct shape for class " + str(true_class) +\
                    " (class color " + class_grid_color[true_class] + ")"
            
            if len(X_indicies) > 0:
                plt.scatter(X_indicies[:, 0], X_indicies[:, 1], marker=true_class_shapes[estimated_class],
                            color=class_grid_color[true_class], label = label)
    
    plt.legend(fontsize=8)
    plt.title(formatted_ml_info + "\nColor is the true class. Shape is estimated class.")
    plt.show()
