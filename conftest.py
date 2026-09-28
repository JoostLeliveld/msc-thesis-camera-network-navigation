"""Repo-root pytest bootstrap.

Puts the ROS 2 package roots on ``sys.path`` so tests (and files later moved to
a different depth) can ``from reliability import ...`` without hand-rolling
``ROOT = Path(__file__).resolve().parents[N]``. Purely additive — existing
per-file inserts still run and are idempotent.
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
