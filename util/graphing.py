import matplotlib.pyplot as plt

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
    
    plt.figure()
    
    for true_class in range(num_class):
        true_class_indicies = (y == true_class)
        for estimated_class in range(num_class):
            estimated_class_indicies = (y_pred == estimated_class)
            
            X_indicies = X[true_class_indicies & estimated_class_indicies]
            
            label = None
            if true_class == estimated_class:
                label = "Correct shape + color class " + str(true_class)
            plt.scatter(X_indicies[:, 0], X_indicies[:, 1], marker=true_class_shapes[true_class],
                        color=estimate_class_color[estimated_class], label = label)
    
    plt.legend(fontsize=8)
    plt.title(formatted_title)
    plt.show()
