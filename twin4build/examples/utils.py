import os
def get_path(list_: str) -> str:
    """
    Get the full path to a file in the examples directory.
    """
    path = os.path.join(os.path.dirname(__file__))
    for path_ in list_:
        path = os.path.join(path, path_)

    return os.path.join(os.path.dirname(__file__), path)