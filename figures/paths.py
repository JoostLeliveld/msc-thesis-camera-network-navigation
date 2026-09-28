"""Depth-independent repo-root resolution.

Historically ~135 files derived the repository root with
``Path(__file__).resolve().parents[N]``, hardcoding each file's nesting depth.
Moving such a file to a different depth silently resolved ``REPO`` to the wrong
directory (no ImportError — just missing-file failures deep in a run).

``repo_root()`` instead walks upward to a sentinel, so a caller's depth no
longer matters. New code should prefer this over ``parents[N]``.
"""

from __future__ import annotations

from pathlib import Path

# A directory is the repo root when it contains any of these sentinels.
_SENTINELS = ("pyproject.toml", "CLAUDE.md", ".git")


def repo_root(start: Path | str | None = None) -> Path:
    """Return the repository root by walking up from ``start``.

    ``start`` defaults to this module's location. Raises ``RuntimeError`` if no
    sentinel is found before the filesystem root, so a misconfigured checkout
    fails loudly rather than resolving to ``/``.
    """
    here = Path(start).resolve() if start is not None else Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if any((candidate / sentinel).exists() for sentinel in _SENTINELS):
            return candidate
    raise RuntimeError(
        f"repo root not found above {here!r} (looked for {_SENTINELS})"
    )
