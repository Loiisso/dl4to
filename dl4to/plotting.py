"""
Plotting functionality has been removed from this fork.
This stub file prevents import errors.
"""

def plot(*args, **kwargs):
    """Stub function - plotting disabled"""
    print("Warning: Plotting functionality has been disabled in this fork")
    return None

def show(*args, **kwargs):
    """Stub function - plotting disabled"""
    print("Warning: Plotting functionality has been disabled in this fork")
    return None

def render(*args, **kwargs):
    """Stub function - plotting disabled"""
    print("Warning: Plotting functionality has been disabled in this fork")
    return None

def visualize(*args, **kwargs):
    """Stub function - plotting disabled"""
    print("Warning: Plotting functionality has been disabled in this fork")
    return None

class PlottingDisabled:
    """Stub class for disabled plotting"""
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        print("Warning: Plotting functionality has been disabled in this fork")
        return None
    
    def __getattr__(self, name):
        return PlottingDisabled()

# Create stubs for common plotting objects
Plotter = PlottingDisabled()