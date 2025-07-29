# Utility functions for dictionary operations

# Standard library imports
from typing import Any, Dict

# Local application imports
from twin4build.utils.rhasattr import rhasattr


def compare_dict_structure(
    dict1: Dict[str, Any], dict2: Dict[str, Any], path: str = ""
) -> Dict[str, Any]:
    """
    Compare the structure of two nested dictionaries.

    This function checks that both dictionaries have the same keys at all nested levels,
    without requiring the actual values to match. It returns detailed information about
    any differences found.

    Args:
        dict1: First dictionary to compare
        dict2: Second dictionary to compare
        path: Current path in the dictionary for error reporting

    Returns:
        Dict containing comparison results:
        - 'structures_match': bool - True if structures match, False otherwise
        - 'missing_in_2': set - keys in dict1 but not in dict2
        - 'missing_in_1': set - keys in dict2 but not in dict1

    Example:
        >>> dict1 = {"a": {"b": 1, "c": 2}, "d": 3}
        >>> dict2 = {"a": {"b": 10, "c": 20}, "d": 30}
        >>> result = compare_dict_structure(dict1, dict2)
        >>> result['structures_match']
        True

        >>> dict1 = {"a": {"b": 1, "c": 2}, "d": 3}
        >>> dict2 = {"a": {"b": 10}, "e": 4}  # Missing "c" and "d", extra "e"
        >>> result = compare_dict_structure(dict1, dict2)
        >>> result['structures_match']
        False
        >>> result['missing_in_2']
        {'d', 'a.c'}
        >>> result['missing_in_1']
        {'e'}
    """
    result = {"structures_match": True, "missing_in_2": set(), "missing_in_1": set()}

    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return result

    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    missing_in_2 = keys1 - keys2
    missing_in_1 = keys2 - keys1

    if missing_in_2 or missing_in_1:
        result["structures_match"] = False

        if missing_in_2:
            missing_keys = {f"{path}.{key}" if path else key for key in missing_in_2}
            result["missing_in_2"].update(missing_keys)

        if missing_in_1:
            extra_keys = {f"{path}.{key}" if path else key for key in missing_in_1}
            result["missing_in_1"].update(extra_keys)

    # Check nested dictionaries
    for key in keys1.intersection(keys2):
        current_path = f"{path}.{key}" if path else key
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            nested_result = compare_dict_structure(dict1[key], dict2[key], current_path)
            if not nested_result["structures_match"]:
                result["structures_match"] = False
                result["missing_in_2"].update(nested_result["missing_in_2"])
                result["missing_in_1"].update(nested_result["missing_in_1"])

    return result


def get_dict_differences(
    dict1: Dict[str, Any], dict2: Dict[str, Any], path: str = ""
) -> Dict[str, Any]:
    """
    Get detailed differences between two nested dictionaries.

    Args:
        dict1: First dictionary to compare
        dict2: Second dictionary to compare
        path: Current path in the dictionary for error reporting

    Returns:
        Dict containing information about differences:
        - 'missing_in_2': keys in dict1 but not in dict2
        - 'missing_in_1': keys in dict2 but not in dict1
        - 'structure_mismatch': True if structures don't match
        - 'differences': nested dictionary with specific differences

    Example:
        >>> dict1 = {"a": {"b": 1, "c": 2}, "d": 3}
        >>> dict2 = {"a": {"b": 10}, "e": 4}
        >>> get_dict_differences(dict1, dict2)
        {
            'missing_in_2': {'d', 'a.c'},
            'missing_in_1': {'e'},
            'structure_mismatch': True,
            'differences': {'a': {'missing_in_2': {'c'}}}
        }
    """
    result = {
        "missing_in_2": set(),
        "missing_in_1": set(),
        "structure_mismatch": False,
        "differences": {},
    }

    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return result

    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    missing_in_2 = keys1 - keys2
    missing_in_1 = keys2 - keys1

    if missing_in_2 or missing_in_1:
        result["structure_mismatch"] = True
        if missing_in_2:
            result["missing_in_2"].update(
                f"{path}.{key}" if path else key for key in missing_in_2
            )
        if missing_in_1:
            result["missing_in_1"].update(
                f"{path}.{key}" if path else key for key in missing_in_1
            )

    # Check nested dictionaries
    for key in keys1.intersection(keys2):
        current_path = f"{path}.{key}" if path else key
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            nested_diff = get_dict_differences(dict1[key], dict2[key], current_path)
            if nested_diff["structure_mismatch"]:
                result["structure_mismatch"] = True
                result["differences"][key] = nested_diff["differences"]
                result["missing_in_2"].update(nested_diff["missing_in_2"])
                result["missing_in_1"].update(nested_diff["missing_in_1"])

    return result


def merge_dicts(
    dict1: Dict[str, Any], dict2: Dict[str, Any], prioritize: str = None
) -> Dict[str, Any]:
    """
    Merge two nested dictionaries.

    Args:
        dict1: First dictionary
        dict2: Second dictionary to merge into dict1
        prioritize: If 'dict1', prioritize dict1 values (only overwrite if dict1 value is None)
                   If 'dict2', prioritize dict2 values (only overwrite if dict2 value is None)
                   If None, use standard merge behavior (dict2 overwrites dict1)

    Returns:
        Merged dictionary

    Example:
        >>> dict1 = {"a": {"b": 1, "c": 2}, "d": 3}
        >>> dict2 = {"a": {"b": 10, "e": 4}, "f": 5}
        >>> merge_dicts(dict1, dict2)  # Standard merge
        {"a": {"b": 10, "c": 2, "e": 4}, "d": 3, "f": 5}

        >>> dict1 = {"a": {"b": 1, "c": None}, "d": None}
        >>> dict2 = {"a": {"b": 10, "c": 20}, "d": 30, "e": 40}
        >>> merge_dicts(dict1, dict2, prioritize='dict1')
        {"a": {"b": 1, "c": 20}, "d": 30}
    """
    if prioritize == "dict1":
        # Prioritize dict1: only overwrite dict1 values if they are None
        result = dict1.copy()

        for key, value in dict2.items():
            if key in result:
                # Key exists in dict1
                if isinstance(result[key], dict) and isinstance(value, dict):
                    # Both are dictionaries, merge recursively
                    result[key] = merge_dicts(result[key], value, prioritize="dict1")
                elif result[key] is None:
                    # Value in dict1 is None, use value from dict2
                    result[key] = value
                # If result[key] is not None, keep the original value (dict1 priority)
            else:
                # Key doesn't exist in dict1, don't add it (dict1 priority)
                pass

        return result

    elif prioritize == "dict2":
        # Prioritize dict2: only overwrite dict2 values if they are None
        result = dict2.copy()

        for key, value in dict1.items():
            if key in result:
                # Key exists in dict2
                if isinstance(result[key], dict) and isinstance(value, dict):
                    # Both are dictionaries, merge recursively
                    result[key] = merge_dicts(value, result[key], prioritize="dict2")
                elif result[key] is None:
                    # Value in dict2 is None, use value from dict1
                    result[key] = value
                # If result[key] is not None, keep the original value (dict2 priority)
            else:
                # Key doesn't exist in dict2, don't add it (dict2 priority)
                pass

        return result

    else:
        # Standard merge behavior: dict2 overwrites dict1
        result = dict1.copy()

        for key, value in dict2.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = value

        return result


def flatten_dict(nested_dict: Dict[str, Any], obj: Any) -> list[tuple[str, Any]]:
    """
    Flatten a nested dictionary into a list of tuples with (key, value) pairs.

    This function recursively traverses nested dictionaries and creates flattened
    key-value pairs using only the final key names. Only final values
    (non-dictionary values) are included in the result.

    Args:
        nested_dict: The nested dictionary to flatten
        obj: The object to which the flattened dictionary belongs
    Returns:
        List of tuples containing (final_key, value) pairs

    Example:
        >>> nested = {"a": {"b": 1, "c": {"d": 2, "e": 3}}, "f": 4}
        >>> flatten_dict(nested)
        [('b', 1), ('d', 2), ('e', 3), ('f', 4)]

        >>> nested = {"user": {"profile": {"name": "John", "age": 30}}}
        >>> flatten_dict(nested)
        [('name', 'John'), ('age', 30)]

        >>> flatten_dict({})  # Empty dict
        []

        >>> flatten_dict({"a": None, "b": {"c": 0}})  # None and zero values
        [('a', None), ('c', 0)]
    """
    flattened = []

    if not isinstance(nested_dict, dict):
        return flattened

    for key, value in nested_dict.items():
        # Check if all keys in the value dict are valid attributes of the object
        cond = isinstance(value, dict) and all([rhasattr(obj, k) for k in value.keys()])
        if cond:
            # Recursively flatten nested dictionaries
            flattened.extend(flatten_dict(value, obj))
        else:
            # Add the final key-value pair using only the final key
            flattened.append((key, value))

    return flattened
