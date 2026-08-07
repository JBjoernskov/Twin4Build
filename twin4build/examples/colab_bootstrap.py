"""Colab-aware Twin4Build installer (stdlib only - safe before the package exists).

Notebooks opened from GitHub/Colab only fetch the ``.ipynb``; they do not install
that git revision. ``pip install twin4build`` hits PyPI (1.x) and is wrong for
docs/dev/PR badges.

Colab runs cell JS inside an output iframe, so the real
``/github/.../blob/<ref>/...`` notebook URL is not available to Python. Do **not**
try to scrape ``window.location`` / ``document.referrer`` for the git ref.

Instead, install from:
1. ``T4B_REF`` if set, else
2. :data:`_T4B_EMBEDDED_REF` (commit SHA baked in by
   ``scripts/patch_colab_notebooks.py`` when example notebooks are refreshed).
"""

# Standard library imports
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/JBjoernskov/Twin4Build.git"
_INSTALL_MARKER = Path("/content/.twin4build_colab_ref")

# Branch or commit SHA. Updated by scripts/patch_colab_notebooks.py.
# Slashy branches are installed via refs/heads/... (see _pip_git_url).
_T4B_EMBEDDED_REF = "fix/full-workflow-portable-data"

_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)


def in_colab() -> bool:
    return "google.colab" in sys.modules


def _pip_git_url(ref: str) -> str:
    """Build a pip ``git+`` URL that tolerates slashy branch names."""
    ref = ref.strip()
    if not ref:
        raise ValueError("empty git ref")
    if _SHA_RE.fullmatch(ref) or ref.startswith("refs/"):
        spec = ref
    elif "/" in ref:
        # ``@fix/foo`` is ambiguous in pip/VCS URLs; use the heads ref.
        spec = f"refs/heads/{ref}"
    else:
        spec = ref
    return f"git+{REPO_URL}@{spec}"


def detect_git_ref() -> str:
    """Return ``T4B_REF`` or the baked-in notebook ref."""
    return os.environ.get("T4B_REF") or _T4B_EMBEDDED_REF


def _import_smoke_ok() -> bool:
    """Check in a fresh process (avoids half-upgraded in-memory numpy)."""
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-c",
                "import numpy; import twin4build",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _restart_colab_kernel() -> None:
    print(
        "Colab: restarting kernel so numpy/scipy pick up the new install.\n"
        "After reconnect, use Runtime > Run all (pip installs persist)."
    )
    os.kill(os.getpid(), 9)


def ensure_twin4build():
    """On Colab, install Twin4Build from git (baked-in SHA / ``T4B_REF``).

    Locally this is a no-op. Returns the ref installed, or ``None`` locally.
    """
    if not in_colab():
        return None

    ref = detect_git_ref()
    url = _pip_git_url(ref)

    marker_ref = (
        _INSTALL_MARKER.read_text(encoding="utf-8").strip()
        if _INSTALL_MARKER.is_file()
        else None
    )
    if marker_ref == ref and _import_smoke_ok():
        print(f"Colab: twin4build already installed ({url})")
        return ref

    print(f"Colab: installing twin4build from {url}")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            url,
        ]
    )
    try:
        _INSTALL_MARKER.parent.mkdir(parents=True, exist_ok=True)
        _INSTALL_MARKER.write_text(ref, encoding="utf-8")
    except OSError:
        pass

    if not _import_smoke_ok():
        print("Colab: install finished but import smoke-test failed.")
        _restart_colab_kernel()
        return ref

    if marker_ref != ref:
        _restart_colab_kernel()
    return ref
