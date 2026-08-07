"""Colab-aware Twin4Build installer (stdlib only - safe before the package exists).

Notebooks opened from GitHub/Colab only fetch the ``.ipynb`` file; they do not
install that git ref. ``pip install twin4build`` hits PyPI (currently 1.x) and
is the wrong default for docs/dev/PR badges.

Notebook setup cells should call :func:`ensure_twin4build` after loading this
module (from a local checkout) or after exec'ing the inlined copy of this
file that the notebooks ship. Do not rely on fetching this path from
``raw.githubusercontent.com`` (branch names with ``/`` and refs not yet on
``dev`` both 404 easily).
"""

# Standard library imports
from __future__ import annotations

import os
import re
import subprocess
import sys
import urllib.parse


REPO_URL = "https://github.com/JBjoernskov/Twin4Build.git"
DEFAULT_REF = "dev"

# Colab / GitHub URLs look like:
#   .../github/<org>/<repo>/blob/<ref>/twin4build/examples/....ipynb
# ``<ref>`` may be a SHA, a simple branch (``dev``), or a slashy branch that is
# either URL-encoded (``fix%2Ffoo``) or left as ``fix/foo``. Capture through
# the ``/twin4build/`` path prefix so slashy refs are not truncated at the
# first ``/``.
_COLAB_BLOB_REF_RE = re.compile(
    r"/github/[^/]+/[^/]+/blob/(.+?)/twin4build/",
    re.IGNORECASE,
)


def in_colab() -> bool:
    return "google.colab" in sys.modules


def detect_git_ref(default: str = DEFAULT_REF) -> str:
    """Resolve branch / tag / commit for ``pip install git+...@ref``.

    Priority:
    1. ``T4B_REF`` environment variable
    2. Colab page URL (``.../github/<org>/<repo>/blob/<ref>/twin4build/...``)
    3. ``default`` (usually ``dev``)
    """
    env_ref = os.environ.get("T4B_REF")
    if env_ref:
        return env_ref

    if in_colab():
        try:
            # Local application imports
            from google.colab import output

            href = output.eval_js("window.location.href") or ""
            match = _COLAB_BLOB_REF_RE.search(href)
            if match:
                return urllib.parse.unquote(match.group(1))
        except Exception:
            pass

    return default


def ensure_twin4build(default_ref: str = DEFAULT_REF):
    """On Colab, install Twin4Build from git so notebooks match the badge ref.

    Locally this is a no-op (use the editable / site-packages install).

    Returns the git ref installed, or ``None`` when no install was performed.
    """
    if not in_colab():
        return None

    ref = detect_git_ref(default=default_ref)
    url = f"git+{REPO_URL}@{ref}"
    print(f"Colab: installing twin4build from {url}")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            url,
        ]
    )
    return ref
