import json
import hashlib
import os
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Sequence


def generate_run_id(prefix: str = "run") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:12]}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def git_hash(repo_root: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, PermissionError, OSError, subprocess.CalledProcessError):
        return None


def _git_bytes(repo_root: str, *args: str) -> Optional[bytes]:
    try:
        result = subprocess.run(
            ["git", "-C", repo_root, *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return bytes(result.stdout)
    except (FileNotFoundError, PermissionError, OSError, subprocess.CalledProcessError):
        return None


def git_provenance(
    repo_root: str, paths: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Return compact, content-sensitive provenance for a possibly dirty checkout.

    A commit SHA alone is not reproducibility evidence when the executable source was
    modified. The hashes deliberately cover both the tracked diff and the contents of
    untracked, non-ignored files without embedding a potentially enormous patch in every run.

    ``paths`` optionally restricts the status/diff/untracked scan to those pathspecs,
    so a caller that knows which directories it actually executes from can ignore edits
    elsewhere in the checkout. ``git_sha`` always describes the whole repository.
    Omitting it scans everything, which is the historical behaviour.
    """

    pathspec = ["--", *paths] if paths else ["--"]
    status = _git_bytes(
        repo_root, "status", "--porcelain=v1", "-z", "--untracked-files=all", *pathspec
    )
    diff = _git_bytes(repo_root, "diff", "--binary", "HEAD", *pathspec)
    untracked = _git_bytes(
        repo_root, "ls-files", "--others", "--exclude-standard", "-z", *pathspec
    )
    provenance: Dict[str, Any] = {
        "git_sha": git_hash(repo_root),
        "git_dirty": bool(status),
        "git_status_sha256": hashlib.sha256(status).hexdigest() if status is not None else None,
        "git_diff_sha256": hashlib.sha256(diff).hexdigest() if diff is not None else None,
    }
    if untracked is None:
        provenance["git_untracked_content_sha256"] = None
        provenance["git_untracked_file_count"] = None
        return provenance
    digest = hashlib.sha256()
    count = 0
    for encoded_path in filter(None, untracked.split(b"\0")):
        path = os.path.join(repo_root, os.fsdecode(encoded_path))
        try:
            with open(path, "rb") as handle:
                digest.update(encoded_path)
                digest.update(b"\0")
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                digest.update(b"\0")
            count += 1
        except (OSError, ValueError):
            digest.update(encoded_path + b"\0<unreadable>\0")
    provenance["git_untracked_content_sha256"] = digest.hexdigest()
    provenance["git_untracked_file_count"] = count
    return provenance


def snapshot_file(src_path: str, dest_dir: str, dest_name: Optional[str] = None) -> Optional[str]:
    if not src_path or not os.path.isfile(src_path):
        return None
    ensure_dir(dest_dir)
    name = dest_name if dest_name else os.path.basename(src_path)
    dest_path = os.path.join(dest_dir, name)
    shutil.copyfile(src_path, dest_path)
    return dest_path


def atomic_write_json(path: str, data: Dict[str, Any]) -> str:
    """Durably replace a JSON file without exposing a truncated final path."""
    parent = os.path.dirname(path) or "."
    ensure_dir(parent)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=parent,
            prefix=os.path.basename(path) + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return path


def write_manifest(dest_dir: str, data: Dict[str, Any], filename: str = "run_manifest.json") -> str:
    ensure_dir(dest_dir)
    path = os.path.join(dest_dir, filename)
    return atomic_write_json(path, data)
