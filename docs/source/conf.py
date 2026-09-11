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
from importlib.metadata import PackageNotFoundError, version

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath("../.."))

project = "Twin4Build"
copyright = "2024–2026, Jakob Bjørnskov, Andres Sebastian Cespedes Cubides"
author = "Jakob Bjørnskov, Andres Sebastian Cespedes Cubides"
try:
    release = version("twin4build")
except PackageNotFoundError:
    release = "0+unknown"
version = release

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
    "auto/modules.rst",
    "auto/twin4build.tests*.rst",
    "auto/twin4build.examples*.rst",
    "auto/twin4build.generated_files*.rst",
    "auto/twin4build._*.rst",
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

# Third-party annotations in prettytable and rdflib refer to optional typing
# helpers not installed at runtime. These warning classes are emitted by the
# type-hints extension and do not indicate broken Twin4Build documentation.
suppress_warnings = [
    "sphinx_autodoc_typehints.guarded_import",
    "sphinx_autodoc_typehints.forward_reference",
]

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


def _github_notebook_branch() -> str:
    """Git ref Colab links should open for this doc build.

    On Read the Docs, prefer a ref GitHub/Colab can resolve (branch, tag, or
    commit SHA). PR preview builds use version slug ``118`` etc., which is
    *not* a git ref — those must use the commit hash instead.
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

    # Pull-request / external builds: VERSION is the PR number, not a branch.
    if rtd_type == "external":
        return rtd_commit or _git_head() or "dev"

    if rtd_type in {"branch", "tag"} and rtd_ident and not str(rtd_ident).isdigit():
        return rtd_ident

    # ``latest`` tracks ``main`` on this project.
    if rtd_version == "latest":
        return "main"
    if rtd_version == "stable":
        if rtd_ident and not str(rtd_ident).isdigit():
            return rtd_ident
        return rtd_commit or "main"

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
    """Expand ``GITHUB_NOTEBOOK_BRANCH`` placeholders in Sphinx sources."""
    source[0] = source[0].replace(
        "GITHUB_NOTEBOOK_BRANCH", app.config.github_notebook_branch
    )


def setup(app):
    app.add_config_value("github_notebook_branch", github_notebook_branch, "env")
    app.connect("source-read", _substitute_github_notebook_branch)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
