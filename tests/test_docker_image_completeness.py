"""
The Dockerfile copies an explicit list of modules, not the whole tree.

That list is hand-maintained, so a new module is easy to add to the repo and
forget here — and nothing in the test suite notices, because the file is
present on disk when the tests run. The failure only shows up as the
container refusing to boot:

    ModuleNotFoundError: No module named 'table_service'

This test walks the real import graph from the production entrypoint and
asserts the image would contain every module it reaches.
"""

import ast
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRYPOINT = "server.py"


def _modules_copied_by_dockerfile() -> set:
    """Every `*.py` named on a COPY line, with line continuations joined."""
    with open(os.path.join(REPO_ROOT, "Dockerfile"), encoding="utf-8") as f:
        text = f.read()
    text = text.replace("\\\n", " ")
    copied = set()
    for line in text.splitlines():
        if line.startswith("COPY "):
            copied.update(re.findall(r"([A-Za-z_][A-Za-z0-9_]*\.py)", line))
    return copied


def _first_party_imports(filename: str) -> set:
    """Top-level module names imported by `filename` that exist in the repo root."""
    with open(os.path.join(REPO_ROOT, filename), encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filename)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return {
        f"{n}.py" for n in names if os.path.isfile(os.path.join(REPO_ROOT, f"{n}.py"))
    }


def _reachable_from(entrypoint: str) -> set:
    """Transitive closure of first-party imports, entrypoint included."""
    seen, queue = {entrypoint}, [entrypoint]
    while queue:
        for module in _first_party_imports(queue.pop()):
            if module not in seen:
                seen.add(module)
                queue.append(module)
    return seen


def test_dockerfile_copies_every_module_the_server_imports():
    missing = _reachable_from(ENTRYPOINT) - _modules_copied_by_dockerfile()
    assert not missing, (
        "These modules are imported by the production server but are NOT copied "
        f"into the Docker image, so the container will fail to boot: {sorted(missing)}. "
        "Add them to the COPY line in the Dockerfile."
    )
