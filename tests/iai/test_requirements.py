"""Guards ``requirements.txt`` against drift from upstream's ``pyproject.toml``.

``requirements.txt`` is the manifest customers install from; ``pyproject.toml`` is upstream's
poetry packaging. An upstream merge that adds a dependency or changes its bounds must fail here
until ``requirements.txt`` mirrors it.
"""

import os
import sys

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IAI_DEPENDENCIES = {"lanelet2", "torchdrivesim", "torch", "imageio", "scikit-learn", "click", "carla"}


def _caret(*, version: str) -> str:
    """Poetry's ``^version``: the left-most non-zero component may not change."""
    parts = [int(p) for p in version.split(".")]
    index = next((i for i, p in enumerate(parts) if p != 0), len(parts) - 1)
    upper = parts[: index + 1]
    upper[-1] += 1
    return f">={version},<{'.'.join(map(str, upper))}"


def _poetry_specifier(*, constraint: str) -> SpecifierSet:
    """Translate a poetry version constraint into a PEP 440 specifier set."""
    parts = []
    for part in constraint.split(","):
        part = part.strip()
        if part.startswith("^"):
            parts.append(_caret(version=part[1:]))
        elif part[0].isdigit():
            parts.append(f"=={part}")
        else:
            parts.append(part.replace(" ", ""))
    return SpecifierSet(",".join(parts))


def _pyproject_requirements() -> set[tuple[str, SpecifierSet, str]]:
    """``(name, specifier, python marker)`` per pyproject main dependency entry."""
    with open(os.path.join(REPO_ROOT, "pyproject.toml"), "rb") as f:
        dependencies = tomllib.load(f)["tool"]["poetry"]["dependencies"]
    expected = set()
    for name, spec in dependencies.items():
        if name == "python":
            continue
        for entry in spec if isinstance(spec, list) else [{"version": spec}]:
            marker = entry.get("python", "")
            marker = f'python_version {marker[:2] if marker[1] == "=" else marker[0]} "{marker.lstrip("<>=")}"' if marker else ""
            expected.add((Requirement(name).name.lower(), _poetry_specifier(constraint=entry["version"]), marker))
    return expected


def _requirements() -> list[Requirement]:
    with open(os.path.join(REPO_ROOT, "requirements.txt")) as f:
        lines = [line.split("#", 1)[0].strip() for line in f]
    return [Requirement(line) for line in lines if line and not line.startswith("-r")]


def test_requirements_mirror_pyproject_dependencies():
    actual = {(r.name.lower(), r.specifier, str(r.marker or "")) for r in _requirements()}
    missing = sorted((n, str(s), m) for n, s, m in _pyproject_requirements() - actual)
    assert not missing, f"requirements.txt does not mirror pyproject.toml for: {missing}"


def test_requirements_declare_iai_dependencies():
    missing = IAI_DEPENDENCIES - {r.name.lower() for r in _requirements()}
    assert not missing, f"Add to requirements.txt: {sorted(missing)}"
