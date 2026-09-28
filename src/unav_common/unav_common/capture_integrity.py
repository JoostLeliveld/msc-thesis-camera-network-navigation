"""Durable capture I/O and exact-byte provenance, independent of ROS."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_bytes(path: Path, expected: str) -> bytes:
    if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError(f'missing or invalid SHA-256 for {path}')
    data = Path(path).read_bytes()
    if digest(data) != expected:
        raise ValueError(f'changed input: {path}')
    return data


def historical_bytes(path: Path, expected: str, *, repo: Path, snapshot: Path | None = None) -> bytes:
    """Read exactly the frozen bytes; Git is only a verified recovery source.

    Never accept a semantically similar/current registry, nor modify the manifest.
    A capture-local snapshot, when declared, is authoritative and must match.
    """
    path, repo = Path(path).resolve(), Path(repo).resolve()
    if snapshot is not None:
        return checked_bytes(snapshot, expected)
    try:
        return checked_bytes(path, expected)
    except (ValueError, FileNotFoundError):
        pass
    if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError(f'missing or invalid historical SHA-256 for {path}')
    try:
        relative = path.relative_to(repo).as_posix()
    except ValueError as exc:
        raise ValueError(f'cannot recover an external historical path: {path}') from exc
    history = subprocess.run(['git', '-C', str(repo), 'log', '--all', '--format=%H', '--', relative],
                             capture_output=True, check=True, text=True).stdout.splitlines()
    for revision in dict.fromkeys(['HEAD', *history]):
        blob = subprocess.run(['git', '-C', str(repo), 'show', f'{revision}:{relative}'],
                              capture_output=True)
        if blob.returncode == 0 and digest(blob.stdout) == expected:
            return blob.stdout
    raise ValueError(f'no exact historical bytes match {expected} for {path}')


def pixel_hash(image) -> str:
    h = hashlib.sha1()  # decoded identity, not an authentication primitive
    h.update(str(image.shape).encode('ascii'))
    h.update(str(image.dtype).encode('ascii'))
    h.update(image.tobytes())
    return h.hexdigest()


def capture_path(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not relative or not path.is_relative_to(root) or path == root:
        raise ValueError(f'image path outside capture: {relative!r}')
    return path


def checked_image(root: Path, row: dict, *, size: tuple[int, int] | None = None):
    data = capture_path(root, row['image']).read_bytes()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or pixel_hash(image) != row.get('image_sha1'):
        raise ValueError(f'decoded image hash mismatch: {row["image"]}')
    if size is not None and image.shape[:2] != (size[1], size[0]):
        raise ValueError(f'capture image dimensions mismatch: {row["image"]}')
    return image


def atomic_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: dict) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def atomic_csv(path: Path, rows: list[dict], fields) -> None:
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode())


@contextmanager
def capture_lock(root: Path):
    """One writer/converter per capture; released by the kernel on process exit."""
    with (Path(root) / '.capture.lock').open('a+b') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('capture is already locked by a writer or converter') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class CaptureIndexWriter:
    """Commit complete camera batches; interrupted partial batches never enter CSV."""
    def __init__(self, path: Path, fields, camera_count: int, rows=()):
        self.path, self.fields, self.camera_count = path, fields, camera_count
        self.rows, self.pending = list(rows), []
        if not path.exists():
            atomic_csv(path, self.rows, fields)

    def writerow(self, row):
        self.pending.append(row)
        if len(self.pending) == self.camera_count:
            combined = self.rows + self.pending
            atomic_csv(self.path, combined, self.fields)
            self.rows, self.pending = combined, []

    def flush(self):
        if self.pending:
            raise RuntimeError('incomplete camera transaction')
