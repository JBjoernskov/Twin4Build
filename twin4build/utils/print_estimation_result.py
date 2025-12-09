# Utility function for printing estimation results

# Standard library imports
import os
import pickle
from typing import Dict, Union


def print_estimation_result(result: Union[str, Dict]) -> None:
    """
    Print estimation results in a human-readable format.

    This function can accept either a filename (string) pointing to a pickle file
    containing estimation results, or a result dictionary/object directly.

    Args:
        result: Either:
            - A string filename path to a pickle file containing estimation results
            - A dictionary or EstimationResult object containing estimation results with keys:
                - result_x: Array of parameter values
                - component_id: Array of component IDs (same order as result_x)
                - component_attr: Array of attribute names (same order as result_x)

    Raises:
        AssertionError: If the result dictionary doesn't contain required keys.
        FileNotFoundError: If the filename doesn't exist.
        ValueError: If the file extension is not .pickle.

    Examples
    --------
    >>> from twin4build.utils.print_estimation_result import print_estimation_result
    >>> 
    >>> # Print from a pickle file
    >>> print_estimation_result("path/to/result.pickle")
    >>> 
    >>> # Print from a result object
    >>> print_estimation_result(result_dict)
    """
    # If result is a string, treat it as a filename and load the pickle file
    if isinstance(result, str):
        if not os.path.exists(result):
            raise FileNotFoundError(f"The file {result} does not exist.")
        
        _, ext = os.path.splitext(result)
        if ext != ".pickle":
            raise ValueError(f"The file {result} is not a pickle file. Expected .pickle extension.")
        
        with open(result, "rb") as handle:
            result = pickle.load(handle)
    
    # Now result should be a dictionary-like object
    assert isinstance(result, dict), "Argument result must be a dictionary or dictionary-like object"
    assert "result_x" in result, "Result dictionary must contain 'result_x' key"
    assert "component_id" in result, "Result dictionary must contain 'component_id' key"
    assert "component_attr" in result, "Result dictionary must contain 'component_attr' key"
    
    result_x = result["result_x"]
    component_id = result["component_id"]
    component_attr = result["component_attr"]
    
    # Ensure all arrays have the same length
    assert len(result_x) == len(component_id), "result_x and component_id must have the same length"
    assert len(result_x) == len(component_attr), "result_x and component_attr must have the same length"
    
    # Convert numpy array to list if needed
    if hasattr(result_x, 'tolist'):
        result_x = result_x.tolist()
    
    # Group data by component
    component_data = {}
    for i in range(len(result_x)):
        comp_id = str(component_id[i])
        attr = str(component_attr[i])
        value = result_x[i]
        
        if comp_id not in component_data:
            component_data[comp_id] = []
        component_data[comp_id].append((attr, value))
    
    # Sort components and parameters within each component
    sorted_components = sorted(component_data.keys())
    for comp_id in sorted_components:
        component_data[comp_id].sort(key=lambda x: x[0])  # Sort by parameter name
    
    # Print header
    print("\n" + "="*80)
    print("ESTIMATION RESULTS")
    print("="*80)
    print(f"{'Component':<30} {'Parameter':<40} {'Value':<20}")
    print("-"*80)
    
    # Print grouped by component
    for comp_id in sorted_components:
        for attr, value in component_data[comp_id]:
            # Format value appropriately
            if isinstance(value, (int, float)):
                if abs(value) >= 1e6 or (abs(value) < 1e-3 and value != 0):
                    value_str = f"{value:.6e}"
                else:
                    value_str = f"{value:.6f}"
            else:
                value_str = str(value)
            
            print(f"{comp_id:<30} {attr:<40} {value_str:<20}")
    
    print("="*80 + "\n")

