import sys
import os
from twin4build.utils.uppath import uppath
def get_main_dir():
    main = sys.modules['__main__']
    assert hasattr(main, '__file__'), "__main__ module does not have __file__ attribute"
    return uppath(os.path.abspath(str(main.__file__)), 1)

