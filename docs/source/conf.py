# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Standard library imports
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath("../.."))

project = "Twin4Build"
copyright = "2024, Jakob Bjørnskov, Andres Sebastian Cespedes Cubides"
author = "Jakob Bjørnskov, Andres Sebastian Cespedes Cubides"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# Files to exclude from documentation
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/tests/*"]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "no-special-members": True,
    "exclude-members": "__weakref__,__dict__,__module__,__init__",
    "member-order": "groupwise",
    "inherited-members": False,
}

# Napoleon settings for docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_ivar = False  # Disable ivar to prevent duplication with properties
napoleon_custom_sections = ["Key Components"]

# Hide implementation details
autodoc_mock_imports = ["tests"]
autodoc_hide_private = True
autodoc_hide_special = True
autodoc_class_members = True
autodoc_docstring_signature = False

# Add these settings to modify how module names are displayed
add_module_names = False  # Don't prefix member names with module names
modindex_common_prefix = [
    "twin4build.",
    "physical_object.",
]  # Strip these prefixes from module names

# -- Options for HTML output -------------------------------------------------

# HTML theme settings
html_theme = "sphinx_rtd_theme"
html_title = "Twin4Build Documentation"

# Theme options
html_theme_options = {
    "titles_only": False,
    "navigation_depth": 6,
    "collapse_navigation": True,
    "prev_next_buttons_location": "none",
}

# Additional HTML settings
add_module_names = False
modindex_common_prefix = ["twin4build."]
html_show_sourcelink = False
html_copy_source = False
toc_object_entries = False

# Sidebars
html_sidebars = {"**": ["globaltoc.html", "searchbox.html"]}

# Static files configuration
html_static_path = ["_static"]

# Include custom CSS
html_css_files = [
    "custom.css",
]


# Recursively crawl through source directory and shorten titles in .rst files
def crawl_source_shorten_titles(path):
    # List files in directory
    for file_name in os.listdir(path):
        # Build path to file
        file_path = os.path.join(path, file_name)

        # Recursively crawl to next directory level
        if os.path.isdir(file_path):
            crawl_source_shorten_titles(file_path)

        # Modify .rst source file title
        else:
            _, extension = os.path.splitext(file_path)
            if extension == ".rst":
                # Read file
                with open(file_path, "r") as file:
                    lines = file.readlines()

                # Process each line
                modified = False
                for i in range(len(lines)):
                    # Look for module titles (they end with " module")
                    if " module\n" in lines[i] and "twin4build." in lines[i]:
                        # Get the last part of the module name
                        module_name = lines[i].split(".")[-1].strip()
                        lines[i] = module_name + "\n"
                        # Update the underline
                        if i + 1 < len(lines):
                            lines[i + 1] = "-" * (len(module_name)) + "\n"
                        modified = True
                    # Handle main page title
                    elif i == 0 and "twin4build." in lines[i]:
                        lines[i] = lines[i].split(".")[-1]
                        if i + 1 < len(lines):
                            lines[i + 1] = "=" * (len(lines[i].strip())) + "\n"
                        modified = True

                # Write back only if modifications were made
                if modified:
                    with open(file_path, "w") as file:
                        file.writelines(lines)


show_title_parents = False
source_path = "../source/auto"
# Remove parents from titles in all .rst files
if not show_title_parents:
    crawl_source_shorten_titles(source_path)
