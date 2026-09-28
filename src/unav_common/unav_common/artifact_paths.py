"""Locate thesis evidence after moving a data bundle, without rewriting its provenance."""
from pathlib import Path


def thesis_artifact_path(value: str | Path, *, anchor: str | Path) -> Path:
    """Resolve a logs/thesis path against the checkout containing ``anchor``.

    Historical manifests contain absolute paths. Prefer the corresponding file
    in the current checkout even if the original machine's path still exists.
    Other absolute paths are unchanged. Relative non-thesis paths are resolved
    beside the anchor file. Hash verification remains the caller's responsibility.
    """
    path = Path(value).expanduser()
    base = Path(anchor).absolute()
    root = next((p for p in (base, *base.parents)
                 if (p / 'pyproject.toml').is_file() and (p / 'pipeline').is_dir()), None)
    parts = path.parts
    for i in range(len(parts) - 1):
        if parts[i:i + 2] == ('logs', 'thesis'):
            suffix = Path(*parts[i:])
            if '..' in suffix.parts:
                raise ValueError('artifact path must not escape logs/thesis')
            if root is not None:
                return root / suffix
    return path if path.is_absolute() else base.parent / path
