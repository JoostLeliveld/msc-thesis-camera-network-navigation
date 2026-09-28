import os
import hashlib
from typing import Any, Dict, List, Optional

from unav_common import manifest as common_manifest


_RUN_PROVENANCE_CACHE: Dict[str, Dict[str, Any]] = {}


def create_run_dir(log_dir: str, prefix: str = 'experiment') -> Dict[str, str]:
    common_manifest.ensure_dir(log_dir)
    for _attempt in range(10):
        run_id = common_manifest.generate_run_id(prefix)
        run_dir = os.path.join(log_dir, run_id)
        try:
            os.mkdir(run_dir)
        except FileExistsError:
            continue
        return {'run_id': run_id, 'run_dir': run_dir}
    raise RuntimeError(f'could not allocate a unique run directory under {log_dir}')


def snapshot_configs(run_dir: str, paths: List[str]) -> Dict[str, Optional[str]]:
    snapshots: Dict[str, Optional[str]] = {}
    for path in paths:
        if not path:
            continue
        name = os.path.basename(path)
        destination_name = name
        if destination_name in snapshots:
            source_tag = hashlib.sha256(os.path.realpath(path).encode('utf-8')).hexdigest()[:12]
            stem, extension = os.path.splitext(name)
            destination_name = f'{stem}.{source_tag}{extension}'
            if destination_name in snapshots:
                raise ValueError(f'duplicate configuration input {path!r}')
        snapshots[destination_name] = common_manifest.snapshot_file(
            path, run_dir, dest_name=destination_name
        )
    return snapshots


def write_manifest(run_dir: str, data: Dict[str, Any], repo_root: str) -> str:
    manifest = dict(data)
    root = os.path.realpath(repo_root)
    run_key = os.path.realpath(run_dir)
    if run_key not in _RUN_PROVENANCE_CACHE:
        raw_paths = os.environ.get('UNAV_EXECUTABLE_SOURCE_PATHS', '')
        paths = tuple(value for value in raw_paths.split(os.pathsep) if value) or None
        _RUN_PROVENANCE_CACHE[run_key] = common_manifest.git_provenance(root, paths)
    manifest.update(_RUN_PROVENANCE_CACHE[run_key])
    return common_manifest.write_manifest(run_dir, manifest)
