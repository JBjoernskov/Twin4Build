# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Standard library imports
import os
import subprocess
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
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "auto/twin4build.core.rst",
    "auto/twin4build.tests.rst",
    "auto/twin4build.examples*.rst",
]

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
# autodoc_mock_imports = ["tests"]
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


def _github_notebook_branch() -> str:
    """Git ref Colab links should open for this doc build.

    On Read the Docs, prefer the build commit SHA whenever available. That is
    unambiguous for GitHub/Colab (unlike PR slugs ``119`` or slashy branch
    names) and matches the notebook content the docs were built from.
    Locally, fall back to the current git branch, then ``dev``.
    """
    rtd_type = os.environ.get("READTHEDOCS_VERSION_TYPE")
    rtd_ident = os.environ.get("READTHEDOCS_GIT_IDENTIFIER")
    rtd_version = os.environ.get("READTHEDOCS_VERSION")
    rtd_commit = os.environ.get("READTHEDOCS_GIT_COMMIT_HASH")

    def _git_head():
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    # Any RTD build: pin Colab to the exact commit that produced these docs.
    if rtd_commit:
        return rtd_commit

    # Pull-request / external builds without a commit env (shouldn't happen).
    if rtd_type == "external":
        return _git_head() or "dev"

    if rtd_type in {"branch", "tag"} and rtd_ident and not str(rtd_ident).isdigit():
        return rtd_ident

    # ``latest`` tracks ``main`` on this project.
    if rtd_version == "latest":
        return "main"
    if rtd_version == "stable":
        if rtd_ident and not str(rtd_ident).isdigit():
            return rtd_ident
        return "main"

    # Named branch versions (e.g. ``dev``), never numeric PR slugs.
    if rtd_type == "branch" and rtd_version and not str(rtd_version).isdigit():
        return rtd_version

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if branch and branch != "HEAD":
            return branch
    except (OSError, subprocess.SubprocessError):
        pass
    return "dev"


github_notebook_branch = _github_notebook_branch()


def _substitute_github_notebook_branch(app, docname, source):
    """Expand ``GITHUB_NOTEBOOK_BRANCH`` placeholders in Sphinx sources.

    The value is URL-encoded so slashy branch names work inside Colab/GitHub
    blob paths. Docs badges also append ``#t4b_ref=...`` so the notebook
    installer can recover the ref even when ``window.location`` is quirky.
    """
    # Standard library imports
    import urllib.parse

    ref = app.config.github_notebook_branch
    encoded = urllib.parse.quote(str(ref), safe="")
    source[0] = source[0].replace("GITHUB_NOTEBOOK_BRANCH", encoded)


def setup(app):
    app.add_config_value("github_notebook_branch", github_notebook_branch, "env")
    app.connect("source-read", _substitute_github_notebook_branch)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
