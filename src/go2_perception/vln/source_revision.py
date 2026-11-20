"""Identify the navigation checkout whose application and policy collect this run."""

from hashlib import sha256
from pathlib import Path
import subprocess

from .motion_profile import Hold


def source_state(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(Path(root)), *args], stderr=subprocess.PIPE)

    try:
        commit = git("rev-parse", "--verify", "HEAD").decode().strip()
        changes = git("status", "--porcelain=v1", "-z", "--untracked-files=all").decode(errors="replace")
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "error": type(exc).__name__}
    state = {
        "commit": commit,
        "dirty": bool(changes),
        "status_porcelain": changes.rstrip("\0").split("\0") if changes else [],
    }
    if changes:
        # The HEAD comparison includes changes already staged for commit as well as working edits.
        try:
            diff = git("diff", "--no-ext-diff", "--binary", "HEAD")
        except (OSError, subprocess.CalledProcessError) as exc:
            state["diff_error"] = type(exc).__name__
        else:
            state["tracked_diff_sha256"] = sha256(diff).hexdigest()
    return state


def clean_navigation(root, expected=None):
    state = source_state(root)
    if state["dirty"] is not False:
        raise Hold("hardware_evidence_requires_clean_navigation_checkout")
    if expected is not None and state != expected:
        raise Hold("capability_navigation_source_mismatch")
    return state
