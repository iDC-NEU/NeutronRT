"""Selected reconstructed h stays private and retires after its consumer."""

from contextlib import ExitStack

import pytest
import torch

from lib.graph.ownership import MissingState
from lib.runtime.aggregates.temporary import TemporaryReconstruction
from scripts.validate.inference import reference
from tests.runtime.test_aggregate_history import _history


@pytest.mark.parametrize("kind", ["features", "insert", "delete"])
@torch.no_grad()
def test_historical_temporary_h_has_old_identity_without_replacing_current_catalog_rows(kind):
	with _history(kind) as (service, context, expected, _features, _degrees, model, target):
		from lib.runtime.aggregates.history import HistoricalAggregates

		helper, temporary = HistoricalAggregates(service), TemporaryReconstruction(service)
		nodes = torch.tensor([0, 1, 3, 5])
		helper.prepare(context, nodes, 2)
		plane = service.remote_plane
		before = service.catalog.available(2, nodes, service.catalog.version)
		with ExitStack() as lifetime:
			workspace = lifetime.enter_context(temporary.workspace(target, nodes.numel() * service.widths[2] * 4))
			table = temporary.create(
				nodes,
				2,
				helper.tables(context, nodes, 2, target),
				helper.degrees(context, nodes, 2),
				target,
				0,
				lifetime,
				workspace=workspace,
			)
			location = table.chunks[0].location
			key = next(key for key in plane._shapes if plane.directory.locate(key) == location)
			assert key.version == 0 and key.kind == "h"
			assert not service.catalog.current_nodes(key).numel()
			torch.testing.assert_close(service.catalog.available(2, nodes, 1), before)
			actual = plane.transport.read(location, torch.arange(nodes.numel()), table.width).result()
			torch.testing.assert_close(actual, expected[nodes], rtol=2e-6, atol=2e-6)
		assert key not in plane._shapes
		with pytest.raises(MissingState):
			plane.directory.locate(key)
		assert not service.catalog.available(2, nodes, 1).any()
		assert service.counters.reconstructed_rows == nodes.numel()
		fresh = reference(model, service.graph, service.features, nodes)
		assert fresh.shape == actual.shape


def test_temporary_reconstruction_does_not_evict_without_workspace_admission():
	with _history("features") as (service, _context, _expected, _features, _degrees, _model, target):
		plane = service.remote_plane
		before = tuple(plane._shapes)
		with ExitStack() as stack, pytest.raises(MemoryError, match="not admitted"):
			temporary = TemporaryReconstruction(service)
			workspace = stack.enter_context(temporary.workspace(target, 0))
			temporary.create(torch.tensor([0]), 2, (), torch.tensor([0]), target, 0, stack, workspace=workspace)
		assert tuple(plane._shapes) == before


def test_failed_temporary_publication_retires_worker_output_and_leaves_catalog_unpublished(monkeypatch):
	with _history("insert") as (service, context, _expected, _features, _degrees, _model, target):
		from lib.runtime.aggregates.history import HistoricalAggregates

		helper, plane = HistoricalAggregates(service), service.remote_plane
		nodes = torch.tensor([0, 1])
		helper.prepare(context, nodes, 1)
		before = tuple(plane._shapes)
		publish = plane.publish_result

		def fail(attempt, result):
			publish(attempt, result)
			raise RuntimeError("temporary publication")

		monkeypatch.setattr(plane, "publish_result", fail)
		with ExitStack() as stack, pytest.raises(RuntimeError, match="temporary publication"):
			temporary = TemporaryReconstruction(service)
			workspace = stack.enter_context(temporary.workspace(target, nodes.numel() * service.widths[1] * 4))
			temporary.create(
				nodes,
				1,
				helper.tables(context, nodes, 1, target),
				helper.degrees(context, nodes, 1),
				target,
				0,
				stack,
				workspace=workspace,
			)
		assert tuple(plane._shapes) == before
		assert not service.catalog.available(1, nodes, 1).any()


def test_reserved_reconstructions_drain_retired_transfers_instead_of_readmitting_a_stale_resident_count(monkeypatch):
	from types import SimpleNamespace

	import dgl

	from lib.runtime.data_plane import RemoteEmbeddingPlane
	from lib.runtime.service import DistributedEmbeddingService
	from tests.runtime.test_prepared import _actors

	with ExitStack() as actors:
		transport, workers = actors.enter_context(_actors())
		plane = RemoteEmbeddingPlane(transport, storage_bytes=65536)
		actors.callback(plane.close)
		graph = dgl.graph(([0, 0, 1], [1, 1, 2]), num_nodes=6)
		service = DistributedEmbeddingService(
			graph, torch.ones(6, 4), (4, 4, 3), plane, retain_aggregates=True, aggregate_only=True
		)
		nodes = torch.arange(6)
		service.ensure(nodes)
		plane = service.remote_plane
		h = service.catalog.selections(1, nodes, 0)[0].key
		a = service.aggregate_catalog.selections(1, nodes, 0)[0].key
		target = plane.directory.locate(a).owner
		# The actor's worker shares this channel class; only its asynchronous
		# completion probe is delayed. wait() still completes the actual send.
		channel = type(plane.transport.comm)
		original = channel.isend

		def delayed(self, *args, **kwargs):
			request = original(self, *args, **kwargs)
			return SimpleNamespace(wait=request.wait, is_completed=lambda: False)

		monkeypatch.setattr(channel, "isend", delayed)
		plane.retire(h)
		plane.writeback(a, plane.storage.owner).result(5)
		plane.evict(a, target)
		plane.remotes[target].max_bytes = 176
		next(worker for worker in workers if worker.owner == target).store.max_bytes = 176
		temporary = TemporaryReconstruction(service)
		selected = torch.arange(4)
		with ExitStack() as stack:
			workspace = stack.enter_context(temporary.workspace(target, 128))
			assert service._aggregate_only is not None
			tables = service._aggregate_only.tables(selected, 1, target)
			first = temporary.create(
				selected, 1, tables, service.graph.in_degrees(selected), target, 0, stack, workspace=workspace
			)
			assert plane.remotes[target].bytes_used == 160
			second = temporary.create(
				selected, 1, tables, service.graph.in_degrees(selected), target, 0, stack, workspace=workspace
			)
			assert plane.remotes[target].bytes_used == 128
			assert workspace.remaining == 0
			assert first.chunks[0].location != second.chunks[0].location
			with pytest.raises(MemoryError, match="not admitted"):
				temporary.create(
					selected, 1, tables, service.graph.in_degrees(selected), target, 0, stack, workspace=workspace
				)
		with ExitStack() as stack, pytest.raises(ValueError, match="live worker workspace"):
			temporary.create(
				selected, 1, tables, service.graph.in_degrees(selected), target, 0, stack, workspace=workspace
			)
