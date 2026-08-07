"""Colab-aware Twin4Build installer (stdlib only — safe before the package exists).

Notebooks opened from GitHub/Colab only fetch the ``.ipynb`` file; they do not
install that git ref. ``pip install twin4build`` hits PyPI (currently 1.x) and
is the wrong default for docs/dev/PR badges.

Usage in a notebook setup cell::

    from pathlib import Path
    import urllib.request

    _bootstrap = Path("twin4build/examples/colab_bootstrap.py")
    if _bootstrap.is_file():
        exec(_bootstrap.read_text(encoding="utf-8"))
    else:
        # Notebook opened alone in Colab: load this helper from the same git ref
        # as the Colab URL when possible, else from ``dev``.
        import re, urllib.parse
        _ref = "dev"
        try:
            from google.colab import output as _out
            _href = _out.eval_js("window.location.href") or ""
            _m = re.search(r"/github/[^/]+/[^/]+/blob/([^/]+)/", _href)
            if _m:
                _ref = urllib.parse.unquote(_m.group(1))
        except Exception:
            pass
        _url = (
            "https://raw.githubusercontent.com/JBjoernskov/Twin4Build/"
            f"{_ref}/twin4build/examples/colab_bootstrap.py"
        )
        exec(urllib.request.urlopen(_url, timeout=30).read().decode("utf-8"))

    ensure_twin4build()
    import twin4build as tb
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


def in_colab() -> bool:
    return "google.colab" in sys.modules


def detect_git_ref(default: str = DEFAULT_REF) -> str:
    """Resolve branch / tag / commit for ``pip install git+...@ref``.

    Priority:
    1. ``T4B_REF`` environment variable
    2. Colab page URL (``.../github/<org>/<repo>/blob/<ref>/...``)
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
            match = re.search(r"/github/[^/]+/[^/]+/blob/([^/]+)/", href)
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
