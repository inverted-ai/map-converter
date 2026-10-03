"""Guards ``requirements.txt`` against drift from upstream's ``pyproject.toml``.

``requirements.txt`` is the manifest customers install from; ``pyproject.toml`` is upstream's
poetry packaging. An upstream merge that adds a dependency must fail here until it is mirrored.
"""

import os
import re
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _normalise(*, name: str) -> str:
    return re.sub(pattern=r"[-_.]+", repl="-", string=name).lower()


def _pyproject_dependencies() -> set[str]:
    with open(os.path.join(REPO_ROOT, "pyproject.toml"), "rb") as f:
        dependencies = tomllib.load(f)["tool"]["poetry"]["dependencies"]
    return {_normalise(name=name) for name in dependencies if name != "python"}


def _requirements(*, filename: str) -> set[str]:
    names = set()
    with open(os.path.join(REPO_ROOT, filename)) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith("-r"):
                continue
            names.add(_normalise(name=re.split(pattern=r"[\s<>=!~;\[]", string=line, maxsplit=1)[0]))
    return names


def test_requirements_cover_pyproject_dependencies():
    missing = _pyproject_dependencies() - _requirements(filename="requirements.txt")
    assert not missing, f"Add to requirements.txt: {sorted(missing)}"


def test_requirements_declare_iai_dependencies():
    iai = {"lanelet2", "torchdrivesim", "torch", "imageio", "scikit-learn"}
    missing = iai - _requirements(filename="requirements.txt")
    assert not missing, f"Add to requirements.txt: {sorted(missing)}"
