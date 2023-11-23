from twin4build.utils.get_main_dir import get_main_dir
import os
def mkdir_in_root(folder_list, filename=None, root=None):
    if root is None:
        current_dir = get_main_dir()
    else:
        current_dir = root
    for folder_name in folder_list:
        current_dir = os.path.join(current_dir, folder_name)
        if os.path.isdir(current_dir)==False:
            os.makedirs(current_dir)
    if filename is None:
        final_filename = current_dir
    else:
        final_filename = os.path.join(current_dir, filename)
    return final_filename
