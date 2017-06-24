import matplotlib.pyplot as plt

# TODO: Would be nice to be able to have an alternative graphing which would
# just draw all of the data at the end when final_update is called - more
# efficient version! And to be able to handle inline graphing solution

plt.ion()

class DynamicGraph():
    """
    Simple utility to allow updating a graph as the values to the graph change over time
    - changing the number of data points, or their values.
    
    
    Parameters
    ------
    
    graph_title
        If provided, will be used as the title for the plot
        
    xlabel
        Optional label for the x-axis
    
    ylabel
        Optional label for the y-axis
        
    Notes
    ------
    
    If the graph becomes overlaid by later figures, make sure to use plt.figure()
    before calling any other plt functions.
    
    WARNING
    --------
    
    When adding new elements, the data will be duplicated so it can get quite slow with huge arrays.
    Best to not graph every point, but only every few points
    """
    def __init__(self, graph_title=None, xlabel=None, ylabel=None):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([],[], 'o', markersize=3)
        
        if graph_title is not None:
            self.ax.set_title(graph_title)
        
        if xlabel is not None:
            self.ax.set_xlabel(xlabel)
            
        if ylabel is not None:
            self.ax.set_ylabel(ylabel)
        
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_autoscalex_on(True)
        self.ax.grid()
        
        # Now check to see if it has the wanted functionality.
        try:
            self.figure.canvas.flush_events()
        except NotImplementedError as e:
            raise NotImplementedError("Warning, you are using an inline graphing solution, so this approach will not work.", )
            print()

    def redraw(self, xdata, ydata):
        """
        Regenerate the graph using the new data. Will rescale the axis as necessary.
            
        Parameters
        ------
        
        xdata : array-like, shape [n_samples]
            x-data values for ALL of the wanted data points
        
        ydata : array-like, shape [n_samples]
            y-data values for ALL of the wanted data points
        
        """
        #Update data (with the new _and_ the old points)
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        
    def final_update(self, xdata, ydata):
        print("Note: If the graph becomes covered by later plots, please use plt.figure() first")
        self.redraw(xdata, ydata)