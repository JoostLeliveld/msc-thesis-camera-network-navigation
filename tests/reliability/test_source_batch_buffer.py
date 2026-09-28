"""Receipt contract tests: use synthetic cameras, clocks and opaque source IDs."""
from types import SimpleNamespace as NS

from reliability.source_batch_buffer import SourceBatchBuffer


def obs(cid, bid, stamp):
    return NS(camera_id=cid, source_batch_id=bid, timestamp_s=stamp)


def test_missing_camera_expires_without_any_more_input():
    events = []
    buf = SourceBatchBuffer(['left', 'right'], timeout_s=.5, on_event=events.append)
    buf.offer(obs('left', 'cycle', 10), 0)
    buf.expire(.6)
    assert not buf.pending
    assert events == [dict(source_batch_id='cycle', status='incomplete_timeout',
                           received_camera_ids=['left'], missing_camera_ids=['right'], pending_wall_s=.6)]
    assert buf.offer(obs('right', 'cycle', 10), .7) is None
    assert not buf.pending


def test_closed_id_eviction_cannot_reanimate_old_evidence():
    buf = SourceBatchBuffer(['left', 'right'], capacity=2)
    for stamp in range(40):
        assert buf.offer(obs('left', str(stamp), stamp), stamp) is None
        assert buf.offer(obs('right', str(stamp), stamp), stamp) is not None
    assert len(buf.closed) == 8
    assert '0' not in buf.closed
    for cid in buf.camera_ids:
        assert buf.offer(obs(cid, '0', 0), 41) is None
    assert not buf.pending


def test_timestamp_skew_does_not_reject_new_round_member_below_previous_max():
    buf = SourceBatchBuffer(['left', 'right'])
    buf.offer(obs('left', 'old', 1), 0)
    assert buf.offer(obs('right', 'old', 1.04), 0)
    buf.offer(obs('left', 'new', 1.02), 0)
    assert buf.offer(obs('right', 'new', 1.06), 0)


def test_first_receipt_owns_camera_slot_and_complete_has_fixed_order():
    buf = SourceBatchBuffer(['left', 'right'])
    first = obs('right', 'cycle', 1)
    buf.offer(first, 0)
    buf.offer(obs('right', 'cycle', 1), 0)
    result = buf.offer(obs('left', 'cycle', 1), .1)
    assert list(result) == ['left', 'right']
    assert result['right'] is first


def test_latest_complete_supersession_is_explicit():
    events = []
    buf = SourceBatchBuffer(['left', 'right'], on_event=events.append)
    buf.offer(obs('left', 'old', 1), 0)
    buf.offer(obs('left', 'new', 2), .1)
    assert buf.offer(obs('right', 'new', 2), .2)
    assert [(e['source_batch_id'], e['status']) for e in events] == [
        ('new', 'complete'), ('old', 'superseded_by_complete_batch')]


def test_conflicting_repeat_aborts_pending_transaction():
    events = []
    buf = SourceBatchBuffer(['left', 'right'], on_event=events.append)
    buf.offer(obs('right', 'cycle', 1), 0)
    buf.offer(obs('right', 'cycle', 2), 0)
    assert buf.offer(obs('left', 'cycle', 1), .1) is None
    assert not buf.pending
    assert events[0]['status'] == 'conflicting_duplicate'
