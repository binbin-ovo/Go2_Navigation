"""Private-output guard: real imagery, geometry and videos only go to git-ignored paths."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _repo_root(path: Path):
    probe = path
    while not probe.exists():
        probe = probe.parent
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=probe if probe.is_dir() else probe.parent, capture_output=True, text=True, check=False)
    return Path(r.stdout.strip()) if r.returncode == 0 else None


def is_private_path(path):
    """True if ``path`` is outside every Git work tree, or ignored by the work tree containing it."""
    path = Path(path).resolve()
    root = _repo_root(path)
    if root is None:
        return True
    rel = path.relative_to(root.resolve())
    if str(rel) == ".":
        return False
    r = subprocess.run(["git", "check-ignore", "-q", "--no-index", str(rel)], cwd=root, check=False)
    return r.returncode == 0


def require_private_dir(path):
    if not is_private_path(path):
        raise SystemExit(f"Refusing to write private output to a path Git would track: {path}")
    return Path(path)
