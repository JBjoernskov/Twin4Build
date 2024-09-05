# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import sys
import os
###Only for testing before distributing package
uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 2)
sys.path.append(file_path)

sys.path.insert(0, os.path.abspath('.'))

sys.path.insert(0, os.path.abspath('../'))

project = 'Twin4Build'
copyright = '2024, Jakob Bjørnskov, Sebastian Cubides'
author = 'Jakob Bjørnskov, Sebastian Cubides'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.napoleon', "sphinx.ext.autodoc", "sphinx.ext.viewcode", "sphinx_autodoc_typehints"]

autoapi_dirs = ['../twin4build']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

numpydoc_show_class_members = False



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
