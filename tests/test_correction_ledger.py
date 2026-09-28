"""Correction accounting cannot depend on scoreability or callback order."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'unav_common'))
from unav_common.correction_ledger import (  # noqa: E402
    CorrectionLedgerError, validate_correction_ledger,
)


def publication(key='batch-a', stamp=10.0, **extra):
    return dict(source_batch_id=key, correction_stamp=stamp, **extra)


def outcome(key='batch-a', stamp=10.0, status='accepted', **extra):
    row = dict(source_batch_id=key, correction_stamp=stamp, apply_stamp=10.5,
               status=status, accepted=status in ('accepted', 'accepted_bootstrap', 'reanchored'),
               reason='nis_gate' if status in ('rejected', 'dropped') else '')
    return row | extra


def codes(result):
    return {error.code for error in result.errors}


@pytest.mark.parametrize('status', ['accepted', 'accepted_bootstrap', 'reanchored', 'rejected', 'dropped'])
def test_all_declared_outcomes_reconcile_without_reference_samples(status):
    result = validate_correction_ledger([publication()], [outcome(status=status)])
    assert result.require_valid() is result
    assert result.valid
    assert bool(result.accepted_update_ids) == (status not in ('rejected', 'dropped'))
    assert sum(result.refusal_counts.values()) == (status in ('rejected', 'dropped'))
    assert json.loads(json.dumps(result.to_dict()))['valid'] is True


def test_capture_and_callback_order_do_not_define_identity():
    pubs = [publication('new', 10.2), publication('old', 10.0)]
    outcomes = [outcome('old', 10.0), outcome('new', 10.2)]
    result = validate_correction_ledger(pubs, outcomes).require_valid()
    assert result.accepted_update_ids == ('new', 'old')
    assert result.by_batch['old']['correction_stamp'] == 10.0


def test_distinct_batches_at_same_capture_and_apply_tick_survive():
    result = validate_correction_ledger(
        [publication('a'), publication('b')], [outcome('a'), outcome('b')],
    ).require_valid()
    assert len(result.by_batch) == 2


def test_missing_and_extra_identity_are_not_hidden_by_equal_counts():
    result = validate_correction_ledger([publication('a')], [outcome('b')])
    assert not result.valid
    assert result.missing_ids == ('a',)
    assert result.extra_ids == ('b',)
    with pytest.raises(CorrectionLedgerError) as exc:
        result.require_valid()
    assert exc.value.result is result


def test_identical_terminal_duplicates_still_break_exactly_once_accounting():
    result = validate_correction_ledger([publication()], [outcome(), outcome()])
    assert result.duplicate_ids == ('batch-a',)
    assert result.outcome_row_count == 2
    assert not result.valid


@pytest.mark.parametrize('key', ['', ' ', 12, None, ' padded '])
def test_identity_is_never_invented_by_string_coercion(key):
    result = validate_correction_ledger([publication(key)], [outcome(key)])
    assert 'invalid_source_batch_id' in codes(result)


@pytest.mark.parametrize('row', [None, [], 3, 'not-a-terminal-record'])
def test_malformed_outcome_returns_structured_failure(row):
    assert 'malformed_record' in codes(validate_correction_ledger(['batch-a'], [row]))


@pytest.mark.parametrize('status', ['teleported', '', None, [], {}])
def test_unknown_status_is_not_a_refusal_or_an_update(status):
    row = outcome()
    row['status'] = status
    assert 'unclassifiable_status' in codes(validate_correction_ledger([publication()], [row]))


@pytest.mark.parametrize('status', ['rejected', 'dropped'])
@pytest.mark.parametrize('reason', ['', '  ', None, 0])
def test_both_refusal_statuses_require_a_real_reason(status, reason):
    result = validate_correction_ledger([publication()], [outcome(status=status, reason=reason)])
    assert 'refusal_without_reason' in codes(result)


@pytest.mark.parametrize('flag', [False, '0', 'false', None, 'yes', 2, float('nan')])
def test_update_cannot_disagree_with_accepted_flag(flag):
    result = validate_correction_ledger([publication()], [outcome(accepted=flag)])
    assert 'inconsistent_accepted_flag' in codes(result)


@pytest.mark.parametrize('flag', [True, '1', 'true', 1.0])
def test_explicit_json_and_csv_true_forms_are_accepted(flag):
    assert validate_correction_ledger([publication()], [outcome(accepted=flag)]).valid


@pytest.mark.parametrize('stamp', [None, '', 'nan', float('inf'), -1, True])
def test_missing_or_invalid_event_time_is_visible(stamp):
    result = validate_correction_ledger([publication()], [outcome(apply_stamp=stamp)])
    assert 'invalid_timestamp' in codes(result)


def test_time_disagreement_is_not_repaired_by_pairing_with_a_different_event():
    result = validate_correction_ledger([publication()], [outcome(stamp=10.1)])
    assert 'inconsistent_timestamp' in codes(result)


def test_future_input_refused_with_reason_is_valid_but_future_update_is_not():
    pub = publication(stamp=100)
    assert validate_correction_ledger([pub], [outcome(stamp=100, status='dropped', reason='future_stamp')]).valid
    assert 'inconsistent_timestamp' in codes(validate_correction_ledger([pub], [outcome(stamp=100)]))


def test_optional_time_mode_does_not_ignore_provided_invalid_times():
    assert validate_correction_ledger(['a'], [dict(source_batch_id='a', status='accepted', accepted=True)],
                                      require_timestamps=False).valid
    result = validate_correction_ledger(['a'], [outcome('a', apply_stamp='nan')], require_timestamps=False)
    assert 'invalid_timestamp' in codes(result)


def test_id_only_publication_input_still_requires_terminal_time():
    assert validate_correction_ledger(['batch-a'], [outcome()]).valid
    assert not validate_correction_ledger(['batch-a'], [dict(source_batch_id='batch-a', status='accepted', accepted=True)]).valid


def test_per_camera_representations_require_explicit_collapse_and_payload_agreement():
    a = publication(camera='a', z=[1, 2], R=[[1, 0], [0, 1]])
    b = publication(camera='b', z=[1, 2], R=[[1, 0], [0, 1]])
    assert 'duplicate_publication' in codes(validate_correction_ledger([a, b], [outcome()]))
    result = validate_correction_ledger([a, b], [outcome()], allow_repeated_publications=True)
    assert result.valid and result.publication_row_count == 2
    b['z'][0] = 9
    assert 'publication_conflict' in codes(validate_correction_ledger([a, b], [outcome()], allow_repeated_publications=True))


@pytest.mark.parametrize('field,left,right', [
    ('frame_id', 'map_bev', 'odom'), ('epoch', 1, 2),
    ('member_ids', ['a', 'b'], ['a', 'c']), ('payload_sha256', 'abc', 'def'),
])
def test_matching_id_does_not_override_conflicting_event_metadata(field, left, right):
    result = validate_correction_ledger([publication(**{field: left})], [outcome(**{field: right})])
    assert 'inconsistent_event_field' in codes(result)


def test_results_own_payloads_and_cannot_be_modified_after_validation():
    pub = publication(z=[1, 2])
    terminal = outcome(posterior={'mean': [1, 2, 0]})
    originals = deepcopy((pub, terminal))
    result = validate_correction_ledger([pub], [terminal]).require_valid()
    assert (pub, terminal) == originals
    terminal['posterior']['mean'][0] = 99
    pub['z'][0] = 99
    assert result.by_batch['batch-a']['posterior']['mean'][0] == 1
    assert result.publications_by_batch['batch-a']['z'][0] == 1
    with pytest.raises(TypeError):
        result.by_batch['batch-a']['status'] = 'dropped'


def test_nested_canonical_payload_is_checked_before_collapsing_camera_rows():
    a = publication(payload={'fused_x': 1.0, 'R': [[1, 0], [0, 1]]})
    b = deepcopy(a)
    assert validate_correction_ledger([a, b], [outcome()], allow_repeated_publications=True).valid
    b['payload']['R'][0][0] = 2
    assert 'publication_conflict' in codes(validate_correction_ledger([a, b], [outcome()], allow_repeated_publications=True))


@pytest.mark.parametrize('payload', [{'x': float('nan')}, {'x': None}, {}, {'x': 'nan'}])
def test_single_invalid_payload_does_not_need_a_duplicate_to_be_detected(payload):
    assert 'invalid_publication_payload' in codes(validate_correction_ledger(
        [publication(payload=payload)], [outcome()],
    ))


def test_empty_ledger_is_distinct_from_missing_required_files():
    result = validate_correction_ledger([], [])
    assert result.valid
    assert result.to_dict()['publication_count'] == 0


@pytest.mark.parametrize('tolerance', [-1, float('nan'), float('inf'), True, None, '0.01', {}])
def test_invalid_time_tolerance_is_a_configuration_error(tolerance):
    with pytest.raises(ValueError):
        validate_correction_ledger([], [], time_tolerance_s=tolerance)


def test_validated_nested_snapshots_can_be_revalidated_at_consumer_boundary():
    first = validate_correction_ledger(
        [publication(payload={'mean': [1, 2], 'R': [[1, 0], [0, 1]]})],
        [outcome(posterior={'mean': [1, 2, 0]})],
    ).require_valid()
    second = validate_correction_ledger(
        first.publications_by_batch.values(), first.by_batch.values(),
    ).require_valid()
    assert second.to_dict() == first.to_dict()
    assert second.by_batch == first.by_batch
    with pytest.raises(TypeError):
        second.by_batch['batch-a']['posterior']['mean'][0] = 7


def test_exact_timestamps_detect_one_nanosecond_conflict_hidden_by_float_rounding():
    stamp_ns = 1_700_000_000_000_000_000
    stamp = stamp_ns / 1e9
    result = validate_correction_ledger(
        [publication(stamp=stamp, correction_stamp_ns=stamp_ns)],
        [outcome(stamp=stamp, apply_stamp=stamp, correction_stamp_ns=stamp_ns + 1,
                 apply_stamp_ns=stamp_ns + 2)],
    )
    assert 'inconsistent_timestamp' in codes(result)


@pytest.mark.parametrize('value', [True, -1, 1.0, '1', None])
def test_supplied_exact_times_are_strict_integers(value):
    result = validate_correction_ledger(
        [publication(correction_stamp_ns=value)], [outcome(apply_stamp_ns=value)],
    )
    assert 'invalid_timestamp' in codes(result)


def test_exact_future_time_refusal_remains_valid_but_acceptance_fails():
    pub = publication(correction_stamp_ns=10_000_000_001)
    fields = dict(correction_stamp_ns=10_000_000_001, apply_stamp_ns=10_000_000_000)
    assert validate_correction_ledger([pub], [outcome(status='dropped', reason='future_stamp', **fields)]).valid
    assert not validate_correction_ledger([pub], [outcome(**fields)]).valid
