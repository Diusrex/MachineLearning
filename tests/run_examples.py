import pkgutil
import matplotlib.pyplot as plt
import traceback

class ExampleError(Exception):
    def __init__(self, errors):

        # Call the base class constructor with the parameters it needs
        super().__init__(errors)

        # Now for your custom code...
        self.errors = errors
    
    def __str__(self):
        errors = "\n\n"
        for error_module in self.errors:
            error, traceback = self.errors[error_module]
            errors += "Error occurred in module {0}, Traceback:\n{1}\n".format(
                    error_module, traceback)
            print()
        return errors

def run_all_examples_in_module(module):
    """
    This function will run all of the main functions in each submodule within
    the provided module.
    If any of them have an exception, this function will raise an exception
    containing all of the inner ones.
    
    Note that plots may appear, but they will be closed before this function
    is done.
    
    Not the cleanest possible way to test all of the examples, but will ensure
    they run successfully.
    
    Warning
    ---------
    Assumes each submodule has a main function, will raise an error if they do not.
    """
    # This code is from https://stackoverflow.com/a/29593583/2648858
    errors = {}
    for loader, name, is_pkg in pkgutil.walk_packages(module.__path__):
        submodule = loader.find_module(name).load_module(name)
        
        try:
            submodule.main()
        except Exception as e:
            errors[submodule.__name__] = (e, traceback.format_exc())
    
    # Make sure no plots are left
    plt.close('all')
    
    if len(errors) != 0:
        raise ExampleError(errors)
