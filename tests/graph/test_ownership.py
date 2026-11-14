from dataclasses import replace

import pytest

from lib.graph.ownership import EmbeddingDirectory, LastReplica, Location, MissingState, Owner, StateKey


@pytest.mark.parametrize("fields", [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (0, 0, 0, "other")])
def test_state_key_rejects_invalid_identity(fields):
	with pytest.raises(ValueError, match="state key"):
		StateKey(*fields)


def test_owner_registration_and_live_owner_validation():
	directory = EmbeddingDirectory()
	owner = directory.register_owner(1)
	assert owner == Owner(1, 1)
	for rank in (-1, 1):
		with pytest.raises(ValueError, match="Owner rank"):
			directory.register_owner(rank)
	with pytest.raises(ValueError, match="not live"):
		directory.begin(StateKey(0, 1, 0), Owner(1, 2))


def test_state_is_unreadable_until_ready_location_is_published():
	directory = EmbeddingDirectory()
	owner = directory.register_owner(1)
	key = StateKey(4, 2, 8)
	with pytest.raises(MissingState):
		directory.locate(key)
	attempt = directory.begin(key, owner)
	with pytest.raises(MissingState):
		directory.locate(key)
	assert directory.publish(attempt, 9)
	assert directory.locate(key) == Location(owner, 9)
	assert directory.locate(key, preferred=owner) == Location(owner, 9)
	assert directory.locate(key, preferred=Owner(2, 1)) == Location(owner, 9)
	assert directory.publish(attempt, 9)
	with pytest.raises(ValueError, match="replace its buffer"):
		directory.publish(attempt, 10)
	with pytest.raises(ValueError, match="already published"):
		directory.begin(key, owner)


def test_reassigned_attempt_rejects_late_completion():
	directory = EmbeddingDirectory()
	first = directory.register_owner(1)
	second = directory.register_owner(2)
	key = StateKey(3, 1, 0)
	old = directory.begin(key, first)
	new = directory.begin(key, second)
	assert new.number > old.number
	assert not directory.publish(old, 1)
	with pytest.raises(MissingState):
		directory.locate(key)
	assert directory.publish(new, 2)
	assert directory.locate(key) == Location(second, 2)
	assert not directory.publish(old, 1)


def test_versions_layers_chunks_and_state_kinds_have_distinct_locations():
	directory = EmbeddingDirectory()
	owner = directory.register_owner(0)
	keys = [StateKey(1, 1, 0), StateKey(2, 1, 0), StateKey(1, 2, 0), StateKey(1, 1, 1), StateKey(1, 1, 0, "a")]
	for handle, key in enumerate(keys):
		assert directory.publish(directory.begin(key, owner), handle)
	for handle, key in enumerate(keys):
		assert directory.locate(key) == Location(owner, handle)


def test_publication_rejects_invalid_handle():
	directory = EmbeddingDirectory()
	owner = directory.register_owner(1)
	attempt = directory.begin(StateKey(0, 1, 0), owner)
	with pytest.raises(ValueError, match="nonnegative"):
		directory.publish(attempt, -1)
	with pytest.raises(MissingState):
		directory.locate(attempt.key)


def test_writeback_is_unreadable_until_copy_completion_and_is_idempotent():
	directory = EmbeddingDirectory()
	gpu = directory.register_owner(1)
	storage = directory.register_owner(0)
	key = StateKey(2, 1, 0)
	directory.publish(directory.begin(key, gpu), 10)
	copy = directory.begin_copy(key, storage)
	assert directory.locate(key, preferred=storage) == Location(gpu, 10)
	assert directory.complete_copy(copy, 20)
	assert directory.complete_copy(copy, 20)
	assert not directory.complete_copy(copy, 21)
	assert directory.locate(key, preferred=storage) == Location(storage, 20)
	with pytest.raises(ValueError, match="already owns"):
		directory.begin_copy(key, storage)


def test_owner_loss_before_writeback_requests_recomputation_and_fences_late_messages():
	directory = EmbeddingDirectory()
	gpu = directory.register_owner(1)
	storage = directory.register_owner(0)
	key = StateKey(3, 1, 0)
	old = directory.begin(key, gpu)
	directory.publish(old, 10)
	copy = directory.begin_copy(key, storage)
	assert directory.lose_owner(gpu) == (key,)
	assert not directory.publish(old, 10)
	assert not directory.complete_copy(copy, 20)
	with pytest.raises(MissingState):
		directory.locate(key)
	restarted = directory.register_owner(1)
	assert restarted.generation == gpu.generation + 1
	replacement = directory.begin(key, restarted)
	assert directory.publish(replacement, 30)
	assert directory.lose_owner(gpu) == ()
	assert directory.locate(key) == Location(restarted, 30)
	assert not directory.complete_copy(copy, 20)


def test_completed_writeback_survives_compute_owner_loss():
	directory = EmbeddingDirectory()
	gpu = directory.register_owner(1)
	storage = directory.register_owner(0)
	key = StateKey(3, 1, 0)
	directory.publish(directory.begin(key, gpu), 10)
	copy = directory.begin_copy(key, storage)
	directory.complete_copy(copy, 20)
	assert directory.lose_owner(gpu) == ()
	assert directory.locate(key) == Location(storage, 20)
	assert directory.lose_owner(storage) == (key,)


def test_lost_pending_compute_is_retried_without_accepting_old_completion():
	directory = EmbeddingDirectory()
	first = directory.register_owner(1)
	second = directory.register_owner(2)
	key = StateKey(0, 1, 0)
	old = directory.begin(key, first)
	assert directory.lose_owner(first) == (key,)
	new = directory.begin(key, second)
	assert not directory.publish(old, 1)
	assert directory.publish(new, 2)


def test_replica_removal_preserves_a_live_copy():
	directory = EmbeddingDirectory()
	gpu = directory.register_owner(1)
	storage = directory.register_owner(0)
	key = StateKey(0, 1, 0)
	directory.publish(directory.begin(key, gpu), 10)
	with pytest.raises(LastReplica):
		directory.remove_replica(key, Location(gpu, 10))
	copy = directory.begin_copy(key, storage)
	directory.complete_copy(copy, 20)
	assert not directory.remove_replica(key, Location(gpu, 99))
	assert directory.remove_replica(key, Location(gpu, 10))
	assert not directory.remove_replica(key, Location(gpu, 10))
	assert directory.locate(key) == Location(storage, 20)


def test_copy_can_use_surviving_replica_after_original_producer_fails():
	directory = EmbeddingDirectory()
	gpu = directory.register_owner(1)
	storage = directory.register_owner(0)
	peer = directory.register_owner(2)
	key = StateKey(0, 1, 0)
	directory.publish(directory.begin(key, gpu), 10)
	copy = directory.begin_copy(key, storage)
	directory.complete_copy(copy, 20)
	directory.lose_owner(gpu)
	copy = directory.begin_copy(key, peer)
	assert copy.source == Location(storage, 20)
	assert directory.complete_copy(copy, 30)
	assert directory.locate(key, preferred=peer) == Location(peer, 30)


def test_failed_or_malformed_copy_cannot_publish_or_cancel_another_transfer():
	directory = EmbeddingDirectory()
	gpu = directory.register_owner(1)
	storage = directory.register_owner(0)
	key = StateKey(0, 1, 0)
	directory.publish(directory.begin(key, gpu), 10)
	copy = directory.begin_copy(key, storage)
	wrong = replace(copy, source=Location(gpu, 99))
	assert not directory.complete_copy(wrong, 20)
	assert not directory.abort_copy(wrong)
	with pytest.raises(ValueError, match="nonnegative"):
		directory.complete_copy(copy, -1)
	assert directory.abort_copy(copy)
	assert not directory.abort_copy(copy)
	assert not directory.complete_copy(copy, 20)
	assert directory.locate(key) == Location(gpu, 10)
	with pytest.raises(ValueError, match="not live"):
		directory.begin_copy(key, Owner(99, 1))
	with pytest.raises(MissingState):
		directory.begin_copy(StateKey(1, 1, 0), storage)
	with pytest.raises(MissingState):
		directory.begin_copy(key, storage, source=Location(gpu, 99))


def test_failure_of_copy_destination_preserves_the_source():
	directory = EmbeddingDirectory()
	gpu = directory.register_owner(1)
	storage = directory.register_owner(0)
	key = StateKey(0, 1, 0)
	directory.publish(directory.begin(key, gpu), 10)
	copy = directory.begin_copy(key, storage)
	assert directory.lose_owner(storage) == ()
	assert not directory.complete_copy(copy, 20)
	assert directory.locate(key) == Location(gpu, 10)
