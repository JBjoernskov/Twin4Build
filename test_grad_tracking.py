import torch
import torch.func

def test_normal_context():
    print("=== Normal Context ===")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    print(f"x: {x}, grad_fn: {x.grad_fn}")
    
    min_val = torch.min(x)
    print(f"min_val: {min_val}, type: {type(min_val)}")
    
    detached = min_val.detach()
    print(f"detached: {detached}, type: {type(detached)}")
    print()

def test_func(x):
    # Simulate what happens in your normalize method
    min_val = torch.min(x)
    max_val = torch.max(x)
    
    # Try to use these in a computation
    result = min_val + max_val
    return result

def test_automatic_differentiation():
    print("=== During Automatic Differentiation ===")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    try:
        # This simulates what happens during jacrev
        jac = torch.func.jacrev(test_func)(x)
        print(f"jac: {jac}")
    except Exception as e:
        print(f"Error during jacrev: {e}")
    
    # Let's try to see what happens inside the function during AD
    print("\n=== Inside function during AD ===")
    try:
        with torch.enable_grad():
            min_val = torch.min(x)
            print(f"min_val: {min_val}")
            print(f"min_val type: {type(min_val)}")
            
            detached = min_val.detach()
            print(f"detached: {detached}")
            print(f"detached type: {type(detached)}")
            
            # Try to use detached in a computation
            result = detached + 1.0
            print(f"result: {result}")
    except Exception as e:
        print(f"Error: {e}")

def test_simulation_context():
    print("=== Simulation Context (like your optimizer) ===")
    # Create a tensor that represents history data
    history = torch.tensor([13.3095, 26.8574, 20.0, 15.0], requires_grad=True)
    
    # Simulate the normalize method
    with torch.no_grad():
        min_val = torch.min(history).detach().clone()
        max_val = torch.max(history).detach().clone()
    
    print(f"min_val: {min_val}, type: {type(min_val)}")
    print(f"max_val: {max_val}, type: {type(max_val)}")
    
    # Now try to use these in a gradient context
    try:
        with torch.enable_grad():
            # This simulates what happens when normalize is called during optimization
            test_tensor = torch.tensor([20.0], requires_grad=True)
            normalized = (test_tensor - min_val) / (max_val - min_val)
            print(f"normalized: {normalized}")
    except Exception as e:
        print(f"Error in gradient context: {e}")

def test_gradtracking_with_item():
    print("=== Testing .item() on GradTrackingTensor ===")
    
    # Simulate what happens during jacrev
    def test_func(x):
        min_val = torch.min(x)
        # Try to extract scalar and wrap back
        min_scalar = torch.tensor(min_val.item(), dtype=torch.float64)
        return min_scalar
    
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    try:
        # This will show us what happens when .item() is called on GradTrackingTensor
        jac = torch.func.jacrev(test_func)(x)
        print(f"jac: {jac}")
        print(f"jac type: {type(jac)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Let's also test what happens when we call .item() directly
    print("\n=== Direct .item() test ===")
    try:
        with torch.enable_grad():
            min_val = torch.min(x)
            print(f"min_val: {min_val}")
            print(f"min_val type: {type(min_val)}")
            
            # Extract scalar
            min_scalar = min_val.item()
            print(f"min_scalar: {min_scalar}, type: {type(min_scalar)}")
            
            # Wrap back in tensor
            min_tensor = torch.tensor(min_scalar, dtype=torch.float64)
            print(f"min_tensor: {min_tensor}")
            print(f"min_tensor type: {type(min_tensor)}")
            
    except Exception as e:
        print(f"Error: {e}")

def test_gradtracking_reproduction():
    print("=== Reproducing GradTrackingTensor Issue ===")
    
    # Simulate the exact situation in your optimizer
    def objective_function(theta):
        # This simulates your __obj_fun_ad method
        # theta is a GradTrackingTensor during jacrev
        
        # Simulate history data that would be a GradTrackingTensor
        history = theta * 10.0  # This creates a GradTrackingTensor
        
        # This is what happens in your normalize method
        min_val = torch.min(history)
        max_val = torch.max(history)
        
        print(f"history type: {type(history)}")
        print(f"min_val type: {type(min_val)}")
        print(f"max_val type: {type(max_val)}")
        
        # Try to extract scalar values
        min_scalar = torch.tensor(min_val.item(), dtype=torch.float64)
        max_scalar = torch.tensor(max_val.item(), dtype=torch.float64)
        
        print(f"min_scalar type: {type(min_scalar)}")
        print(f"max_scalar type: {type(max_scalar)}")
        
        # Use these in a computation
        result = min_scalar + max_scalar
        return result
    
    # Create input tensor
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    try:
        # This simulates your jacrev call
        jac = torch.func.jacrev(objective_function)(x)
        print(f"jac: {jac}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_stored_values_issue():
    print("\n=== Testing Stored Values Issue ===")
    
    # Simulate storing min/max as instance variables
    class MockScalar:
        def __init__(self):
            self._min_history = None
            self._max_history = None
            self._history = None
        
        def set_history(self, history):
            self._history = history
        
        def normalize(self, v):
            # This simulates your original normalize method
            if self._min_history is None:
                print("FIRST CALL MIN")
                self._min_history = torch.tensor(torch.min(self._history).item(), dtype=torch.float64)
            if self._max_history is None:
                print("FIRST CALL MAX")
                self._max_history = torch.tensor(torch.max(self._history).item(), dtype=torch.float64)
            
            print("MIN BEFORE", self._min_history)
            print("MAX BEFORE", self._max_history)
            
            return (v - self._min_history) / (self._max_history - self._min_history)
    
    def objective_function(theta):
        # Create mock scalar
        scalar = MockScalar()
        scalar.set_history(theta * 10.0)
        
        # First call - should work fine
        result1 = scalar.normalize(theta[0:1])
        
        # Second call - this is where the issue occurs
        result2 = scalar.normalize(theta[1:2])
        
        return result1 + result2
    
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    try:
        jac = torch.func.jacrev(objective_function)(x)
        print(f"jac: {jac}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_cached_float_values():
    print("\n=== Testing Cached Float Values ===")
    
    # Simulate the new caching approach
    class MockScalarWithFloatCache:
        def __init__(self):
            self._min_history = None  # Will store float
            self._max_history = None  # Will store float
            self._history = None
        
        def set_history(self, history):
            self._history = history
        
        def normalize(self, v):
            # Cache min/max as Python floats
            if self._min_history is None:
                print("FIRST CALL MIN")
                with torch.no_grad():
                    self._min_history = torch.min(self._history).item()  # Store as float
            if self._max_history is None:
                print("FIRST CALL MAX")
                with torch.no_grad():
                    self._max_history = torch.max(self._history).item()  # Store as float
            
            # Convert cached floats to tensors when needed
            min_val = torch.tensor(self._min_history, dtype=torch.float64)
            max_val = torch.tensor(self._max_history, dtype=torch.float64)
            
            print("MIN BEFORE", min_val)
            print("MAX BEFORE", max_val)
            
            return (v - min_val) / (max_val - min_val)
    
    def objective_function(theta):
        # Create mock scalar
        scalar = MockScalarWithFloatCache()
        scalar.set_history(theta * 10.0)
        
        # First call - should work fine
        result1 = scalar.normalize(theta[0:1])
        
        # Second call - should also work fine with cached values
        result2 = scalar.normalize(theta[1:2])
        
        return result1 + result2
    
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    try:
        jac = torch.func.jacrev(objective_function)(x)
        print(f"jac: {jac}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_normal_context()
    test_automatic_differentiation()
    test_simulation_context()
    test_gradtracking_with_item()
    test_gradtracking_reproduction()
    test_stored_values_issue()
    test_cached_float_values() 