"""Helpers shared by ``convert_map.py`` and ``map_report.py``, kept free of heavy imports.

``map_report.py`` measures converter checkouts, so it must not load the converter's stack
(torch, crdesigner, ...) just to hash a file or read a git commit.
"""

from __future__ import annotations

import hashlib
import subprocess

# convert_map.py exits with this code when it declines a run on purpose (invalid configuration or
# an input it will not place); Python itself uses 1 for exceptions and 2 for "can't open file".
REFUSED_EXIT_CODE = 3
# convert_map.py exits with this code when AutoCritic's preset rejects the converted map (outputs withheld).
REJECTED_EXIT_CODE = 4


class MapRejected(Exception):
    """AutoCritic's preset failed the converted map; its outputs were withheld, its reports written."""


def sha256_file(*, path: str) -> str:
    """sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity(*, repo: str) -> dict:
    """Commit and dirty flag of a checkout; ``None`` values when it is not a git checkout."""
    try:
        head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return {"commit": None, "dirty": None}
    if head.returncode != 0:
        return {"commit": None, "dirty": None}
    status = subprocess.run(
        ["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}
