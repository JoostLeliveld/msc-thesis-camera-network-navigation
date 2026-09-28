"""Durable, versioned camera outcome journals shared by producer and manager.

Caller payloads retain their stage (detector or manager). Each writer owns a
unique producer_epoch; source image epochs belong in member metadata.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import threading

OUTCOME_SCHEMA = 'camera_batch_outcome.v2'
OUTCOME_HISTORY_DEPTH = 4096
DEFAULT_JOURNAL_MAX_BYTES = 256 * 1024 * 1024


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def journal_path(configured_path, epoch, environ=None):
    env = os.environ if environ is None else environ
    if configured_path:
        return Path(str(configured_path).replace('{producer_epoch}', epoch)).expanduser().resolve()
    root = env.get('ROS_LOG_DIR', '')
    if not root:
        raise ValueError('outcome_journal_path or ROS_LOG_DIR is required for durable camera outcomes')
    return Path(root).expanduser().resolve() / 'camera_outcomes' / f'{epoch}.jsonl'


class OutcomeJournal:
    """Exclusive, bounded append-only journal; never overwrite or silently rotate.

    append returns only after fsync. A failed/partial write poisons this writer so
    a later retry cannot turn a torn record into apparently complete evidence.
    Topic delivery can be replayed independently using the immutable event ID.
    """
    def __init__(self, path, producer_epoch, *, max_bytes=DEFAULT_JOURNAL_MAX_BYTES):
        if not producer_epoch or isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError('producer_epoch and positive integer journal byte limit required')
        self.path = Path(path).expanduser().resolve()
        missing_directories = []
        ancestor = self.path.parent
        while not ancestor.exists():
            missing_directories.append(ancestor)
            ancestor = ancestor.parent
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.producer_epoch = str(producer_epoch)
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._failed = False
        self._sequence = 0
        self._size = 0
        self._previous_hash = ''
        self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.fsync(self._fd)
            # Persist both the file directory entry and any newly created parents.
            for directory_path in dict.fromkeys([self.path.parent, *(p.parent for p in missing_directories)]):
                directory = os.open(directory_path, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        except BaseException:
            os.close(self._fd)
            self._fd = None
            raise

    def append(self, event):
        with self._lock:
            if self._failed or self._fd is None:
                raise RuntimeError('camera outcome journal is unavailable; stop processing')
            sequence = self._sequence + 1
            payload = dict(event, schema_version=OUTCOME_SCHEMA,
                           producer_epoch=self.producer_epoch, event_seq=sequence,
                           event_id=f'{self.producer_epoch}:{sequence}',
                           journal_path=str(self.path), previous_event_sha256=self._previous_hash)
            payload['event_sha256'] = hashlib.sha256(canonical_json(payload).encode('utf-8')).hexdigest()
            encoded = (canonical_json(payload) + '\n').encode('utf-8')
            previous_signal_mask = _block_termination_signals()
            try:
                try:
                    if self._size + len(encoded) > self.max_bytes:
                        raise RuntimeError('camera outcome journal byte limit exceeded; stop processing')
                    written = 0
                    while written < len(encoded):
                        count = os.write(self._fd, encoded[written:])
                        if count <= 0:
                            raise OSError('zero-length camera journal write')
                        written += count
                    os.fsync(self._fd)
                except BaseException:
                    self._failed = True
                    raise
                # Commit the in-memory hash-chain state while termination remains
                # blocked. A pending SIGINT may be delivered as soon as the old mask is
                # restored, but by then this durable record is a complete transaction
                # and a shutdown handler can safely append ``session_stopped`` next.
                self._size += len(encoded)
                self._sequence = sequence
                self._previous_hash = payload['event_sha256']
            finally:
                _restore_signal_mask(previous_signal_mask)
            return payload

    def close(self):
        with self._lock:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None


def read_journal(path):
    """Yield verified records; fail explicitly on torn tails, gaps or alteration.

    An incomplete final line must not be interpreted as a detector miss or a
    successful terminal event. Earlier yielded rows retain their original IDs.
    """
    previous = ''
    epoch = None
    with Path(path).open('rb') as stream:
        for sequence, line in enumerate(stream, 1):
            if not line.endswith(b'\n'):
                raise ValueError('incomplete camera outcome journal tail')
            event = json.loads(line)
            claimed = event.pop('event_sha256')
            actual = hashlib.sha256(canonical_json(event).encode('utf-8')).hexdigest()
            if claimed != actual or event['previous_event_sha256'] != previous:
                raise ValueError('camera outcome journal hash chain mismatch')
            epoch = event['producer_epoch'] if epoch is None else epoch
            if (event['schema_version'] != OUTCOME_SCHEMA or event['producer_epoch'] != epoch
                    or event['event_seq'] != sequence or event['event_id'] != f'{epoch}:{sequence}'):
                raise ValueError('camera outcome journal identity or sequence mismatch')
            event['event_sha256'] = claimed
            previous = claimed
            yield event


def _block_termination_signals():
    """Delay process termination across one journal write transaction.

    Python delivers signals only on the main thread. On platforms with
    ``pthread_sigmask`` we therefore block SIGINT/SIGTERM only there; worker
    threads need no mask change. Returning ``None`` keeps the journal portable
    to platforms without this POSIX API.
    """

    if threading.current_thread() is not threading.main_thread():
        return None
    pthread_sigmask = getattr(signal, 'pthread_sigmask', None)
    if pthread_sigmask is None:
        return None
    signals = {signal.SIGINT, signal.SIGTERM}
    try:
        return pthread_sigmask(signal.SIG_BLOCK, signals)
    except (AttributeError, OSError, ValueError):
        return None


def _restore_signal_mask(previous_mask):
    """Restore a mask after the durable state is committed.

    A pending SIGINT is allowed to raise here. Deliberately do not catch it:
    shutdown must proceed, but the journal must not be poisoned merely because
    termination arrived during ``fsync``.
    """

    if previous_mask is not None:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
