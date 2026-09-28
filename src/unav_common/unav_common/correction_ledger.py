"""Raw correction publication/outcome integrity, independent of ROS and reference data.

Call before numerical filtering or truth pairing. This validates event accounting,
not deployment provenance, reference support or the availability of a posterior.
Transport retransmissions and per-camera publication representations must be handled
explicitly; duplicate terminal rows are never silently removed here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from numbers import Real
from types import MappingProxyType
from typing import Any


UPDATE_STATUSES = frozenset({'accepted', 'accepted_bootstrap', 'reanchored'})
REFUSAL_STATUSES = frozenset({'rejected', 'dropped'})
VALID_STATUSES = UPDATE_STATUSES | REFUSAL_STATUSES

# These are canonical shared-publication fields, not camera-specific row fields.
# File/schema adapters map their source columns explicitly to this vocabulary.
# Canonical epoch is the correction producer's epoch, never the independent
# recursive-belief epoch. Adapters retain belief epoch as separate provenance.
_PUBLICATION_FIELDS = (
    'correction_stamp', 'correction_stamp_ns', 'frame_id', 'epoch', 'payload_sha256', 'member_ids',
    'payload', 'z', 'R', 'mean', 'covariance', 'fused_x', 'fused_y', 'fused_cov_xx',
    'fused_cov_xy', 'fused_cov_yy',
)
_IDENTITY_FIELDS = ('frame_id', 'epoch', 'payload_sha256', 'member_ids')


@dataclass(frozen=True)
class LedgerIssue:
    code: str
    source_batch_id: str | None
    row_index: int | None
    message: str


class CorrectionLedgerError(ValueError):
    def __init__(self, result: 'LedgerValidation'):
        self.result = result
        self.errors = result.errors
        super().__init__('; '.join(
            f'{e.code}[{e.source_batch_id or "unidentified"}]: {e.message}'
            for e in result.errors
        ))


@dataclass(frozen=True)
class LedgerValidation:
    errors: tuple[LedgerIssue, ...]
    by_batch: Mapping[str, Mapping[str, Any]]
    publications_by_batch: Mapping[str, Mapping[str, Any]]
    accepted_update_ids: tuple[str, ...]
    refusal_counts: Mapping[str, int]
    publication_row_count: int
    outcome_row_count: int

    @property
    def valid(self) -> bool:
        return not self.errors

    def _ids(self, code: str) -> tuple[str, ...]:
        return tuple(sorted({e.source_batch_id for e in self.errors
                             if e.code == code and e.source_batch_id is not None}))

    @property
    def missing_ids(self) -> tuple[str, ...]:
        return self._ids('missing_assimilation')

    @property
    def extra_ids(self) -> tuple[str, ...]:
        return self._ids('extra_assimilation')

    @property
    def duplicate_ids(self) -> tuple[str, ...]:
        return self._ids('duplicate_assimilation')

    def require_valid(self) -> 'LedgerValidation':
        if not self.valid:
            raise CorrectionLedgerError(self)
        return self

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe integrity summary; full payloads remain in the raw evidence."""
        return {
            'valid': self.valid,
            'errors': [asdict(e) for e in self.errors],
            'missing_ids': list(self.missing_ids),
            'extra_ids': list(self.extra_ids),
            'duplicate_ids': list(self.duplicate_ids),
            'accepted_update_ids': list(self.accepted_update_ids),
            'refusal_counts': dict(self.refusal_counts),
            'publication_count': len(self.publications_by_batch),
            'publication_row_count': self.publication_row_count,
            'outcome_count': len(self.by_batch),
            'outcome_row_count': self.outcome_row_count,
        }


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return frozenset(_freeze(v) for v in value)
    return deepcopy(value)


def _copy_record(value: Any) -> Any:
    """Copy canonical containers, including this validator's immutable snapshots."""
    if isinstance(value, Mapping):
        return {k: _copy_record(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_copy_record(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_copy_record(v) for v in value)
    return deepcopy(value)


def _finite_time(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _exact_time(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _accepted_flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return {'1': True, '0': False, 'true': True, 'false': False}.get(value.strip().lower())
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    # Invalid/nonfinite payloads cannot establish agreement between copies.
    if isinstance(left, float) and not math.isfinite(left):
        return False
    if isinstance(right, float) and not math.isfinite(right):
        return False
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _numeric_payload(value: Any) -> bool:
    """Canonical numerical payload; shape/required-field contracts belong to adapters."""
    if isinstance(value, Mapping):
        return bool(value) and all(_numeric_payload(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_numeric_payload(v) for v in value)
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def validate_correction_ledger(
    publications: Iterable[Mapping[str, Any] | str],
    outcomes: Iterable[Mapping[str, Any]],
    *,
    require_timestamps: bool = True,
    allow_repeated_publications: bool = False,
    time_tolerance_s: float = 1e-9,
) -> LedgerValidation:
    """Validate raw canonical records without reference-data or score filtering.

    Publications may be IDs when only identity reconciliation is available. A mapping
    publication must provide ``correction_stamp`` when timestamps are required. Terminal
    records require an explicit accepted flag, one of five statuses and a reason for
    either refusal status. Timestamped outcomes require correction/apply timestamps.
    Reordered capture or callback order is valid; no sorting into a fictitious epoch is
    performed. A reasoned refusal of a future-stamped input is valid evidence, so only
    accepted updates require apply time not to precede correction time.

    ``by_batch`` contains terminal records; ``publications_by_batch`` contains publication
    records. Both are owned immutable snapshots. Call ``require_valid`` before consuming
    them as validated input. Empty streams are a valid empty ledger; manifest/capability
    wrappers must separately enforce presence of required files and expected activity.
    """
    if (not isinstance(time_tolerance_s, Real) or isinstance(time_tolerance_s, bool)
            or not math.isfinite(time_tolerance_s)
            or time_tolerance_s < 0):
        raise ValueError('time_tolerance_s must be finite and nonnegative')
    errors: list[LedgerIssue] = []
    published: dict[str, dict[str, Any]] = {}
    terminal: dict[str, dict[str, Any]] = {}
    publication_rows = outcome_rows = 0

    def issue(code: str, key: str | None, index: int | None, message: str) -> None:
        errors.append(LedgerIssue(code, key, index, message))

    def record(raw: Any, index: int, *, publication: bool) -> tuple[str, dict] | None:
        if publication and isinstance(raw, str):
            raw = {'source_batch_id': raw}
        elif not isinstance(raw, Mapping):
            issue('malformed_record', None, index, 'expected a record mapping')
            return None
        key = raw.get('source_batch_id')
        if not isinstance(key, str) or not key.strip() or key != key.strip():
            issue('invalid_source_batch_id', None, index, 'expected a nonempty canonical string ID')
            return None
        return key, _copy_record(raw)

    for index, raw in enumerate(publications):
        publication_rows += 1
        id_only = isinstance(raw, str)
        parsed = record(raw, index, publication=True)
        if parsed is None:
            continue
        key, row = parsed
        if 'correction_stamp_ns' in row and _exact_time(row['correction_stamp_ns']) is None:
            issue('invalid_timestamp', key, index, 'correction_stamp_ns must be nonnegative integer nanoseconds')
        if (require_timestamps and not id_only) or 'correction_stamp' in row:
            if _finite_time(row.get('correction_stamp')) is None:
                issue('invalid_timestamp', key, index, 'publication correction_stamp is missing or invalid')
        for field in ('payload', 'z', 'R', 'mean', 'covariance', 'fused_x', 'fused_y',
                      'fused_cov_xx', 'fused_cov_xy', 'fused_cov_yy'):
            if field in row and not _numeric_payload(row[field]):
                issue('invalid_publication_payload', key, index, f'{field} is not a finite numeric payload')
        if key in published:
            if not allow_repeated_publications:
                issue('duplicate_publication', key, index, 'publication identity occurs more than once')
            else:
                previous = published[key]
                for field in _PUBLICATION_FIELDS:
                    if ((field in row) != (field in previous)
                            or (field in row and not _same(row[field], previous[field]))):
                        issue('publication_conflict', key, index, f'repeated publication disagrees on {field}')
            continue
        published[key] = row

    for index, raw in enumerate(outcomes):
        outcome_rows += 1
        parsed = record(raw, index, publication=False)
        if parsed is None:
            continue
        key, row = parsed
        if key in terminal:
            issue('duplicate_assimilation', key, index, 'terminal identity occurs more than once')
            continue
        terminal[key] = row
        status = row.get('status')
        if not isinstance(status, str) or status not in VALID_STATUSES:
            issue('unclassifiable_status', key, index, 'terminal status is not one of the five declared statuses')
        elif status in REFUSAL_STATUSES:
            reason = row.get('reason')
            if not isinstance(reason, str) or not reason.strip():
                issue('refusal_without_reason', key, index, 'rejected and dropped outcomes require a reason')
        accepted = _accepted_flag(row.get('accepted'))
        if accepted is None or (isinstance(status, str) and status in VALID_STATUSES
                                and accepted != (status in UPDATE_STATUSES)):
            issue('inconsistent_accepted_flag', key, index, 'accepted must explicitly agree with status')
        times: dict[str, float | None] = {}
        for field in ('correction_stamp', 'apply_stamp'):
            if require_timestamps or field in row:
                times[field] = _finite_time(row.get(field))
                if times[field] is None:
                    issue('invalid_timestamp', key, index, f'{field} is missing or invalid')
        corr, apply = times.get('correction_stamp'), times.get('apply_stamp')
        for field in ('correction_stamp_ns', 'apply_stamp_ns'):
            if field in row and _exact_time(row[field]) is None:
                issue('invalid_timestamp', key, index, f'{field} must be nonnegative integer nanoseconds')
        corr_ns, apply_ns = (_exact_time(row.get(field))
                             for field in ('correction_stamp_ns', 'apply_stamp_ns'))
        if accepted is True and corr_ns is not None and apply_ns is not None and apply_ns < corr_ns:
            issue('inconsistent_timestamp', key, index, 'accepted update precedes exact correction time')
        if accepted is True and corr is not None and apply is not None and apply + time_tolerance_s < corr:
            issue('inconsistent_timestamp', key, index, 'accepted update precedes its correction time')
        publication = published.get(key)
        if publication is not None:
            pub_ns = _exact_time(publication.get('correction_stamp_ns'))
            if pub_ns is not None and corr_ns is not None and pub_ns != corr_ns:
                issue('inconsistent_timestamp', key, index, 'publication and outcome exact correction times differ')
            pub_time = _finite_time(publication.get('correction_stamp'))
            if pub_time is not None and corr is not None and abs(pub_time - corr) > time_tolerance_s:
                issue('inconsistent_timestamp', key, index, 'publication and outcome correction times differ')
            for field in _IDENTITY_FIELDS:
                if field in publication and field in row and not _same(publication[field], row[field]):
                    issue('inconsistent_event_field', key, index, f'publication and outcome disagree on {field}')

    for key in sorted(published.keys() - terminal.keys()):
        issue('missing_assimilation', key, None, 'published correction has no terminal outcome')
    for key in sorted(terminal.keys() - published.keys()):
        issue('extra_assimilation', key, None, 'terminal outcome has no published correction')
    accepted_ids = tuple(sorted(key for key, row in terminal.items()
                                if isinstance(row.get('status'), str) and row['status'] in UPDATE_STATUSES))
    counts = Counter(row['status'] for row in terminal.values()
                     if isinstance(row.get('status'), str) and row['status'] in REFUSAL_STATUSES)
    return LedgerValidation(
        tuple(errors), _freeze(terminal), _freeze(published), accepted_ids,
        MappingProxyType({status: counts[status] for status in sorted(REFUSAL_STATUSES)}),
        publication_rows, outcome_rows,
    )
