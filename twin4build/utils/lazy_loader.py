import importlib
class LazyLoader:
    """Lazily imports modules/objects on demand."""
    
    def __init__(self):
        self._modules = {}
        self._objects = {}
    
    def add_lazy_class(self, name, module_path, class_name):
        """Register a class to be lazily loaded."""
        self._objects[name] = (module_path, class_name)
    
    def __getattr__(self, name):
        # Check if it's a registered lazy object
        if name in self._objects:
            module_path, attr_name = self._objects[name]
            
            # Import the module if not already imported
            if module_path not in self._modules:
                self._modules[module_path] = importlib.import_module(module_path)
            
            # Get the actual class
            obj = getattr(self._modules[module_path], attr_name)
            
            # Cache it for future access
            setattr(self, name, obj)
            
            return obj
        
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")