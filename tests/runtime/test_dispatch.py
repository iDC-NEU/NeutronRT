from contextlib import ExitStack, closing
from threading import Event

import pytest
import torch

from lib.graph.admission import UpdateAdmission, UpdateKind
from lib.graph.on_demand import OnDemand
from lib.graph.ownership import MissingState
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.service import DistributedEmbeddingService
from lib.runtime.transport import ModelExecutionError
from tests.runtime.test_prepared import _actors, _graph


@pytest.mark.parametrize("preemption", [False, True])
def test_strict_queries_preempt_only_after_a_complete_wave(preemption):
	with _actors() as (transport, workers), ExitStack() as stack:
		plane = RemoteEmbeddingPlane(transport, storage_bytes=65536)
		stack.callback(plane.close)
		graph, features = _graph(), torch.ones(4, 4)
		service = DistributedEmbeddingService(
			graph,
			features,
			(4, 4, 3),
			plane,
			batch_size=1,
			concurrent_compute=True,
			pipeline=True,
			preemption_enabled=preemption,
		)
		service.ensure(torch.arange(4))
		started, proceed = Event(), Event()
		original = workers[0]._apply

		def apply(task, inputs):
			if not started.is_set():
				started.set()
				assert proceed.wait(5)
			return original(task, inputs)

		workers[0]._apply = apply
		with closing(UpdateAdmission(service)) as admission, closing(OnDemand(service)) as demand:
			try:
				update = admission.submit(UpdateKind.FEATURES, torch.arange(4), features + 0.25)
				assert started.wait(5)
				observed = []

				def consume(values):
					observed.append(service.counters.updates)
					return values.clone()

				query = demand.submit(torch.tensor([3, 0, 3]), plane.storage.owner, consume)
				proceed.set()
				answer = query.result(10)
				assert update.result(10).version == answer.version == 1
				assert observed == [0 if preemption else 1]
				model = workers[0].model
				with torch.inference_mode():
					expected = model(1, graph, model(0, graph, features))
				torch.testing.assert_close(answer.value, expected[[3, 0, 3]], rtol=2e-5, atol=2e-5)
			finally:
				proceed.set()


@pytest.mark.parametrize("name", ["sage", "gat", "gcn"])
@pytest.mark.parametrize("coalesced,peer,gather", [(False, False, False), (True, True, False), (True, True, True)])
def test_concurrent_service_publishes_exact_feature_and_edge_updates(name, coalesced, peer, gather):
	with _actors(name, coalesced=coalesced, gather=gather) as (transport, workers), ExitStack() as stack:
		plane = RemoteEmbeddingPlane(transport, storage_bytes=65536)
		stack.callback(plane.close)
		graph, features = _graph(), torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10
		pointer = features.data_ptr()
		service = DistributedEmbeddingService(
			graph,
			features,
			(4, 4, 3),
			plane,
			batch_size=1,
			peer_fetch=peer,
			gather_inputs=gather,
			concurrent_compute=True,
			source_degree_sensitive=name == "gcn",
			pipeline=True,
		)
		service.ensure(torch.arange(4))
		for mutation in (
			lambda: service.change_features(torch.tensor([0, 2]), features[[0, 2]] + 0.5),
			lambda: service.insert_edges(torch.tensor([1, 3]), torch.tensor([3, 2])),
			lambda: service.delete_edges(torch.tensor([0, 3]), torch.tensor([1, 2])),
		):
			mutation()
			model = workers[0].model
			with torch.inference_mode():
				expected = model(1, graph, model(0, graph, features))
			with service.query(torch.tensor([3, 0, 3, 1]), plane.storage.owner) as values:
				torch.testing.assert_close(values, expected[[3, 0, 3, 1]], rtol=2e-5, atol=2e-5)
		assert service.counters.updates == 3 and service.counters.input_changes == 6
		assert features.data_ptr() == pointer
		assert all(worker._prepared is None for worker in workers)


@pytest.mark.parametrize("phase", ["prepare", "model", "publish"])
def test_partial_wave_failure_drains_launches_cancels_inputs_and_retains_resumable_publications(phase):
	with _actors() as (transport, workers):
		plane = RemoteEmbeddingPlane(transport, storage_bytes=65536)
		try:
			graph, features = _graph(), torch.ones(4, 4)
			service = DistributedEmbeddingService(
				graph, features, (4, 4, 3), plane, batch_size=1, concurrent_compute=True
			)
			service.ensure(torch.arange(4))
			if phase == "prepare":
				original = transport.prepare
				calls = 0

				def prepare(*args, **kwargs):
					nonlocal calls
					calls += 1
					if calls == 2:
						raise ValueError("injected prepare failure")
					return original(*args, **kwargs)

				transport.prepare = prepare
				restore_target, restore_name = transport, "prepare"
			elif phase == "model":
				original = workers[1]._apply
				workers[1]._apply = lambda *_args: (_ for _ in ()).throw(ValueError("injected model failure"))
				restore_target, restore_name = workers[1], "_apply"
			else:
				original = service.catalog.publish
				calls = 0

				def publish(key):
					nonlocal calls
					calls += 1
					if calls == 1:
						raise ValueError("injected publish failure")
					return original(key)

				service.catalog.publish = publish
				restore_target, restore_name = service.catalog, "publish"
			with pytest.raises(ModelExecutionError if phase == "model" else ValueError):
				service.change_features(torch.arange(4), features + 0.5)
			setattr(restore_target, restore_name, original)
			assert service.counters.updates == 0 and service._pending_update.version == 1
			result = service.resume_update(1)
			assert result.version == 1 and service.counters.updates == 1
			assert service.counters.input_changes == 4
			assert all(worker._prepared is None for worker in workers)
			with service.query(torch.arange(4), plane.storage.owner) as actual:
				model = workers[0].model
				with torch.inference_mode():
					expected = model(1, graph, model(0, graph, features))
				torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
			# Failed reservations cannot become readable catalog state.
			for key in tuple(plane._shapes):
				try:
					plane.directory.locate(key)
				except MissingState:
					pytest.fail("A cancelled chunk leaked into the remote shape table")
		finally:
			plane.close()
