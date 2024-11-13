# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('../..'))

project = 'Twin4Build'
copyright = '2024, Jakob Bjørnskov, Andres Sebastian Cespedes Cubides'
author = 'Jakob Bjørnskov, Andres Sebastian Cespedes Cubides'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'myst_parser'
]

# Files to exclude from documentation
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**/tests/*']

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'no-special-members': True,
    'exclude-members': '__weakref__,__dict__,__module__,__init__',
    'member-order': 'groupwise',
    'inherited-members': False
}

# Napoleon settings for docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_ivar = True
napoleon_custom_sections = ['Key Components']

# Hide implementation details
autodoc_mock_imports = ['tests']
autodoc_hide_private = True
autodoc_hide_special = True
autodoc_class_members = True
autodoc_docstring_signature = False

# -- Options for HTML output -------------------------------------------------

# HTML theme settings
html_theme = 'sphinx_rtd_theme'
html_title = 'Twin4Build Documentation'

# Theme options
html_theme_options = {
    'titles_only': False,
    'navigation_depth': 6,
    'collapse_navigation': True,
    'prev_next_buttons_location': 'none'
}

# Additional HTML settings
add_module_names = False
modindex_common_prefix = ['twin4build.']
html_show_sourcelink = False
html_copy_source = False
toc_object_entries = False

# Sidebars
html_sidebars = {
    '**': [
        'globaltoc.html',
        'searchbox.html'
    ]
}
