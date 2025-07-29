# Standard library imports
import os

# Local application imports
from twin4build.utils.get_main_dir import get_main_dir


def mkdir_in_root(folder_list, filename=None, root=None):
    if root is None:
        current_dir = get_main_dir()
    else:
        current_dir = root
    if os.path.isdir(os.path.join(current_dir, *folder_list)) == False:
        for folder_name in folder_list:
            current_dir = os.path.join(current_dir, folder_name)
            if os.path.isdir(current_dir) == False:
                os.makedirs(current_dir)
    else:
        current_dir = os.path.join(current_dir, *folder_list)

    if filename is None:
        final_filename = current_dir
    else:
        final_filename = os.path.join(current_dir, filename)

    isfile = os.path.isfile(final_filename)
    return final_filename, isfile
