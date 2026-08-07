"""Bake current git branch/SHA into colab_bootstrap and example notebooks.

Colab cannot read the real /github/.../blob/<ref>/ notebook URL from cell JS
(output iframe). Notebooks therefore ship an explicit ref and install that.
"""
# Standard library imports
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_PATH = ROOT / "twin4build" / "examples" / "colab_bootstrap.py"


def _current_ref() -> str:
    """Prefer branch name (tracks PR tip); fall back to HEAD SHA."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if branch and branch != "HEAD":
            return branch
    except (OSError, subprocess.SubprocessError):
        pass
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def main() -> None:
    ref = _current_ref()
    text = BOOTSTRAP_PATH.read_text(encoding="utf-8")
    text, n = re.subn(
        r'^_T4B_EMBEDDED_REF = ".*"$',
        f'_T4B_EMBEDDED_REF = "{ref}"',
        text,
        count=1,
        flags=re.M,
    )
    if n != 1:
        raise SystemExit("could not update _T4B_EMBEDDED_REF in colab_bootstrap.py")
    BOOTSTRAP_PATH.write_text(text, encoding="utf-8")
    print(f"embedded ref: {ref}")

    bootstrap = BOOTSTRAP_PATH.read_text(encoding="utf-8")
    prefix = f"""# Colab: install Twin4Build from the git ref baked into this notebook
# (docs/PR badges). Local checkouts skip install. Override with T4B_REF.
from pathlib import Path

_bootstrap = Path("twin4build/examples/colab_bootstrap.py")
if not _bootstrap.is_file():
    _bootstrap = Path("colab_bootstrap.py")

if _bootstrap.is_file():
    exec(_bootstrap.read_text(encoding="utf-8"))
else:
    exec({bootstrap!r})

ensure_twin4build()
"""
    block_re = re.compile(
        r"(?:# Colab:[\s\S]*?)?ensure_twin4build\(\)\s*\n",
    )

    updated = []
    for nb_path in sorted((ROOT / "twin4build" / "examples").glob("*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        changed = False
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if "ensure_twin4build" not in src and "colab_bootstrap" not in src:
                continue
            match = block_re.search(src)
            if not match:
                continue
            rest = src[match.end() :].lstrip("\n")
            new_src = prefix + rest
            if new_src == src:
                continue
            lines = new_src.splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            cell["source"] = lines
            changed = True
        if changed:
            nb_path.write_text(
                json.dumps(nb, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            updated.append(nb_path.name)
    print("updated:", ", ".join(updated) if updated else "(none)")


if __name__ == "__main__":
    main()
