# Standard library imports
import os


def get_path(list_: str) -> str:
    """
    Get the full path to a file in the examples directory.
    """
    path = os.path.join(os.path.dirname(__file__))
    for path_ in list_:
        path = os.path.join(path, path_)

    return os.path.join(os.path.dirname(__file__), path)


def validate_schema(data):
    if not isinstance(data, dict):
        raise TypeError("Data should be a dictionary.")
    for main_key in ["input", "output"]:
        if main_key not in data:
            raise ValueError(f"'{main_key}' key is required in the data.")
        if not isinstance(data[main_key], dict):
            raise TypeError(f"'{main_key}' should be a dictionary.")
        for param, param_data in data[main_key].items():
            if not isinstance(param_data, dict):
                raise TypeError(
                    f"Each parameter under '{main_key}' should be a dictionary."
                )
            required_keys = {
                "min": (float, int),
                "max": (float, int),
                "description": str,
            }
            for key, expected_type in required_keys.items():
                if key not in param_data:
                    raise ValueError(
                        f"'{key}' key is required for '{param}' in '{main_key}'."
                    )

                if not isinstance(param_data[key], expected_type):
                    raise TypeError(
                        f"'{key}' in '{param}' under '{main_key}' should be of type {expected_type.__name__}."
                    )
            if param_data["min"] > param_data["max"]:
                raise ValueError(
                    f"'min' value should be <= 'max' for '{param}' in '{main_key}'."
                )
    # print("Data is valid.")
