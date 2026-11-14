from contextlib import ExitStack, nullcontext

import pytest
import torch

from lib.graph.catalog import EmbeddingCatalog
from lib.graph.ownership import EmbeddingDirectory, MissingState
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.transport import ReadyResult
from tests.runtime.test_prepared import _actors


@torch.inference_mode()
def test_old_routing_and_lifetimes_survive_current_publication_and_exclude_already_replaced_rows():
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors())
		plane = RemoteEmbeddingPlane(transport, storage_bytes=4096)
		stack.callback(plane.close)
		catalog = EmbeddingCatalog(6, 2, plane.directory, kind="a")
		nodes = torch.arange(6)
		old_values = torch.arange(24, dtype=torch.float32).reshape(6, 4)

		def publish(selected, values, version):
			key = catalog.reserve(1, selected, version)
			worker = workers[0]
			handle = worker.store.put(values)
			worker.store.mark_published(handle)
			attempt = plane.directory.begin(key, worker.owner)
			plane.publish_result(
				attempt, ReadyResult(handle, version, 1, key.chunk, tuple(values.shape), worker.store.bytes_used)
			)
			for obsolete in catalog.publish(key):
				plane.retire(obsolete)
			return key

		old_key = publish(nodes, old_values, 0)
		catalog.begin_version(1, {}, pending_layers=(1, 2))
		publish(torch.tensor([0]), torch.ones(1, 4), 1)
		selected = torch.tensor([0, 5, 1, 5])
		with catalog.previous_rows(1, selected, 1, retain=plane.retain_aggregates) as old:
			assert old.version == 0 and old.layer == 1 and old.nodes.data_ptr() == selected.data_ptr()
			assert old.available.tolist() == [False, True, True, True]
			catalog.finish_analysis(1, nodes, 1)
			publish(nodes, old_values + 1, 1)
			assert plane.directory.locate(old_key)
			assert len(old.selections) == 1
			group = old.selections[0]
			assert group.positions.tolist() == [1, 2, 3] and group.rows.tolist() == [5, 1, 5]
			with ExitStack() as leases:
				torch.testing.assert_close(
					plane.read_rows(group.key, group.rows, leases), old_values[selected[group.positions]]
				)
		with pytest.raises(MissingState):
			plane.directory.locate(old_key)
		with (
			pytest.raises(ValueError, match="pending"),
			catalog.previous_rows(1, nodes, 1, retain=plane.retain_aggregates),
		):
			pass
		catalog.finish_analysis(2, torch.empty(0, dtype=torch.int64), 1)
		catalog.begin_version(5, {}, pending_layers=(1,))
		with catalog.previous_rows(1, nodes, 5, retain=plane.retain_aggregates) as old:
			assert old.version == 1 and old.available.all()
		assert not plane._retained_states


def test_prior_rows_do_not_turn_stale_lost_or_never_computed_rows_into_old_anchors():
	directory = EmbeddingDirectory()
	first, second = directory.register_owner(1), directory.register_owner(2)
	catalog = EmbeddingCatalog(4, 1, directory, kind="a")
	for owner, nodes in ((first, torch.tensor([0, 1])), (second, torch.tensor([2]))):
		key = catalog.reserve(1, nodes, 0)
		directory.publish(directory.begin(key, owner), key.chunk)
		catalog.publish(key)
	catalog.begin_version(1, {1: torch.tensor([1])})
	directory.lose_owner(second)
	catalog.begin_version(2, {}, pending_layers=(1,))
	retained = []

	def retain(keys):
		retained.append(tuple(keys))
		return nullcontext(None)

	with catalog.previous_rows(1, torch.arange(4), 2, retain=retain) as old:
		assert old.version == 1 and old.available.tolist() == [True, False, False, False]
		assert len(old.selections) == 1 and old.selections[0].positions.tolist() == [0]
		# An old physical chunk may be certified through a newer untouched version.
		assert old.selections[0].key.version == 0
	assert len(retained) == 1 and len(retained[0]) == 1
	with catalog.previous_rows(1, torch.empty(0, dtype=torch.int64), 2, retain=retain) as old:
		assert not old.available.numel() and not old.selections
	assert retained[-1] == ()
