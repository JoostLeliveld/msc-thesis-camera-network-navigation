"""Repo-root pytest bootstrap.

Puts the ROS 2 package roots on ``sys.path`` so tests (and files later moved to
a different depth) can ``from reliability import ...`` without hand-rolling
``ROOT = Path(__file__).resolve().parents[N]``. Purely additive — existing
per-file inserts still run and are idempotent.

Deliberately NOT added: the ``scripts/`` analysis dirs. Their import
conventions are inconsistent and mutually incompatible on a shared path —
``scripts/shared`` and ``scripts/visibility_comparison`` expect the dir itself
on the path (flat ``import metrics`` / ``import common``), whereas
``scripts/geometry_visibility`` expects its *parent* ``scripts/`` on the path
(``from geometry_visibility import geometry_visibility`` as a namespace
package). Putting ``scripts/`` on the path would also shadow the ROS
``reliability`` / ``perception`` packages with the ``scripts/reliability`` /
``scripts/perception`` dirs. The tests that need those roots bootstrap them
themselves; do not centralize here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

# ROS 2 package roots: each holds an inner python package (e.g.
# src/reliability/reliability/), so the wrapper dir goes on the path to enable
# `from reliability import ...`.
_SRC_PKG_DIRS = (
    "src/reliability",
    "src/unav_common",
    "src/experiments",
    "src/planning",
    "src/perception",
    "src/state",
    "src/sim",
)


for _rel in _SRC_PKG_DIRS:
    _path = _ROOT / _rel
    if _path.is_dir():
        _entry = str(_path)
        if _entry not in sys.path:
            sys.path.insert(0, _entry)
