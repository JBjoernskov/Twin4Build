import os
def get_path(filename: str) -> str:
    """
    Get the full path to a file in the examples directory.
    """
    return os.path.join(os.path.dirname(__file__), filename)
