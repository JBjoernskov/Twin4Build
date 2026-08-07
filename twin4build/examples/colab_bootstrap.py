"""Colab-aware Twin4Build installer (stdlib only - safe before the package exists).

Notebooks opened from GitHub/Colab only fetch the ``.ipynb`` file; they do not
install that git ref. ``pip install twin4build`` hits PyPI (currently 1.x) and
is the wrong default for docs/dev/PR badges.

Notebook setup cells should call :func:`ensure_twin4build` after loading this
module (from a local checkout) or after exec'ing the inlined copy of this
file that the notebooks ship.
"""

# Standard library imports
from __future__ import annotations

import os
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path


REPO_URL = "https://github.com/JBjoernskov/Twin4Build.git"
DEFAULT_REF = "dev"
_INSTALL_MARKER = Path("/content/.twin4build_colab_ref")

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
_COLAB_BLOB_SHA_RE = re.compile(
    r"/blob/([0-9a-f]{7,40})(?:/|[?#]|$)",
    re.IGNORECASE,
)
_COLAB_BLOB_SEGMENT_RE = re.compile(r"/blob/([^/?#]+)/", re.IGNORECASE)


def in_colab() -> bool:
    return "google.colab" in sys.modules


def _colab_page_url() -> str:
    """Best-effort Colab browser URL (empty when unavailable)."""
    if not in_colab():
        return ""
    try:
        # Local application imports
        from google.colab import output
    except Exception:
        return ""

    for expr in (
        "window.location.href",
        "window.top.location.href",
        "document.location.href",
        "document.referrer",
    ):
        try:
            href = output.eval_js(expr) or ""
        except Exception:
            continue
        if isinstance(href, str) and href.startswith("http"):
            return href
    return ""


def _ref_from_href(href: str) -> str | None:
    """Extract a git ref from a Colab/GitHub notebook URL."""
    if not href:
        return None

    parsed = urllib.parse.urlparse(href)
    frag_qs = urllib.parse.parse_qs(parsed.fragment)
    if frag_qs.get("t4b_ref"):
        return urllib.parse.unquote(frag_qs["t4b_ref"][0])

    query_qs = urllib.parse.parse_qs(parsed.query)
    if query_qs.get("t4b_ref"):
        return urllib.parse.unquote(query_qs["t4b_ref"][0])

    match = _COLAB_BLOB_REF_RE.search(href)
    if match:
        return urllib.parse.unquote(match.group(1))

    match = _COLAB_BLOB_SHA_RE.search(href)
    if match:
        return match.group(1)

    match = _COLAB_BLOB_SEGMENT_RE.search(href)
    if match:
        return urllib.parse.unquote(match.group(1))

    return None


def detect_git_ref(default: str = DEFAULT_REF) -> str:
    """Resolve branch / tag / commit for ``pip install git+...@ref``.

    Priority:
    1. ``T4B_REF`` environment variable
    2. Colab page URL (path, ``?t4b_ref=``, or ``#t4b_ref=``)
    3. ``default`` (usually ``dev``)
    """
    env_ref = os.environ.get("T4B_REF")
    if env_ref:
        return env_ref

    href = _colab_page_url()
    ref = _ref_from_href(href)
    if ref:
        return ref

    if in_colab():
        print(
            "Colab: could not detect git ref from page URL "
            f"({href!r}); falling back to {default!r}. "
            "Set os.environ['T4B_REF'] = '<branch-or-sha>' before "
            "ensure_twin4build() to override."
        )
    return default


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


def _restart_colab_runtime() -> None:
    """Kill the Colab kernel so binary deps (numpy/scipy) reload cleanly.

    Uses a hard kernel kill (not ``runtime.unassign()``) so ``/content`` and
    pip installs persist; the user only needs Run all after reconnect.
    """
    print(
        "Colab: restarting kernel so numpy/scipy pick up the new install.\n"
        "After reconnect, use Runtime > Run all (installed packages persist)."
    )
    os.kill(os.getpid(), 9)


def ensure_twin4build(default_ref: str = DEFAULT_REF):
    """On Colab, install Twin4Build from git so notebooks match the badge ref.

    Locally this is a no-op (use the editable / site-packages install).

    Returns the git ref installed, or ``None`` when no install was performed.
    """
    if not in_colab():
        return None

    ref = detect_git_ref(default=default_ref)
    url = f"git+{REPO_URL}@{ref}"

    marker_ref = (
        _INSTALL_MARKER.read_text(encoding="utf-8").strip()
        if _INSTALL_MARKER.is_file()
        else None
    )
    if marker_ref == ref and _import_smoke_ok():
        print(f"Colab: twin4build already installed from {url}")
        return ref

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
    try:
        _INSTALL_MARKER.parent.mkdir(parents=True, exist_ok=True)
        _INSTALL_MARKER.write_text(ref, encoding="utf-8")
    except OSError:
        pass

    # pip often upgrades numpy/scipy while those modules are already loaded in
    # this kernel; importing twin4build in-process then fails with obscure
    # numpy._core errors. Restart once so the next Run-all uses a clean interp.
    if not _import_smoke_ok():
        print(
            "Colab: install finished but import smoke-test failed; "
            "restarting runtime."
        )
        _restart_colab_runtime()
        return ref

    # Smoke test passed in a subprocess, but this kernel may still hold stale
    # numpy extensions from before the upgrade - restart when we just installed.
    if marker_ref != ref:
        _restart_colab_runtime()
    return ref
