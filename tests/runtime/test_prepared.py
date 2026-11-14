"""Real worker actors over in-memory FIFO channels, without sockets or CUDA."""

from collections import defaultdict
from contextlib import contextmanager
from queue import Queue
from threading import Barrier, Lock, Thread
from types import SimpleNamespace

import dgl
import pytest
import torch

from lib.dist.buffers import PackedLayout
from lib.graph.affected import AffectedPlanner
from lib.graph.ownership import ComputeAttempt, Location, Owner, StateKey
from lib.graph.serving import select_rows
from lib.models.spec import ModelSpec
from lib.runtime.protocol import CANCEL_PREPARED, STOP, InputChunk, pack_compute
from lib.runtime.transport import OwnedTransport, RemoteEndpoint
from lib.runtime.worker import OwnedWorker


class _Fabric:
	def __init__(self):
		self.channels = defaultdict(Queue)
		self.lock = Lock()

	def channel(self, source, target, epoch):
		with self.lock:
			return self.channels[source, target, epoch]


class _Channel:
	device = torch.device("cpu")

	def __init__(self, fabric, rank):
		self.fabric, self.rank = fabric, rank

	def empty(self, shape, dtype=torch.float32):
		return torch.empty(shape, dtype=dtype)

	def send(self, value, target, *, epoch=None, **_kwargs):
		self.fabric.channel(self.rank, target, epoch).put(value)

	def isend(self, value, target, **kwargs):
		self.send(value, target, **kwargs)
		return SimpleNamespace(wait=lambda: None, is_completed=lambda: True)

	def isend_tensors(self, values, target):
		from lib.dist.buffers import TensorSpec

		layout = PackedLayout(tuple(TensorSpec.of(value) for value in values))
		buffer = torch.empty(layout.nbytes, dtype=torch.uint8)
		for view, value in zip(layout.views(buffer), values, strict=True):
			view.copy_(value)
		return self.isend(buffer, target)

	def recv_tensors(self, specs, source):
		buffer = self.fabric.channel(source, self.rank, None).get(timeout=10)
		return PackedLayout(specs).views(buffer)

	def recv_tensor(self, shape, *, src, dtype=torch.float32, epoch=None, **_kwargs):
		value = self.fabric.channel(src, self.rank, epoch).get(timeout=10)
		assert value.shape == torch.Size((shape,) if isinstance(shape, int) else shape)
		assert value.dtype == dtype
		return value

	def irecv(self, target, source, *, epoch=None, **_kwargs):
		def wait():
			value = self.fabric.channel(source, self.rank, epoch).get(timeout=10)
			target.copy_(value)

		return SimpleNamespace(wait=wait)

	def wait_all(self, requests):
		for request in requests:
			request.wait()

	def destroy(self):
		pass


@contextmanager
def _actors(name="sage", *, coalesced=True, gather=False):
	fabric, workers, threads, errors = _Fabric(), [], [], []
	spec = ModelSpec(name, 4, 4, 3)
	for rank in (1, 2):
		worker = OwnedWorker(
			_Channel(fabric, rank),
			spec.build(),
			Owner(rank, 1),
			max_bytes=4096,
			control_epoch=9,
			coalesced=coalesced,
			gather_inputs=gather,
		)
		workers.append(worker)

		def run(worker=worker):
			try:
				worker.run()
			except BaseException as error:
				errors.append(error)

		thread = Thread(target=run, daemon=True)
		thread.start()
		threads.append(thread)
	root = _Channel(fabric, 0)
	transport = OwnedTransport(
		root, 9, [RemoteEndpoint(worker.owner, root.device, 4096) for worker in workers], coalesced=coalesced
	)
	try:
		yield transport, workers
	finally:
		for worker in workers:
			root.send(torch.tensor([STOP, 0, 1, 0, 0, 0, 0, 0]), worker.owner.rank, epoch=9)
		for thread in threads:
			thread.join(10)
		assert all(not thread.is_alive() for thread in threads)
		assert not errors, errors


def _graph():
	return dgl.graph(([0, 1, 2, 3, 0, 1, 2, 3], [1, 2, 3, 0, 0, 1, 2, 3]), num_nodes=4)


def _initialize(transport, workers, graph, features, name):
	planner = AffectedPlanner(graph, 2, source_degree_sensitive=name == "gcn")
	task = planner.block(1, torch.arange(4))
	locations = []
	for worker in workers:
		packet = pack_compute(
			task,
			ComputeAttempt(StateKey(0, 1, worker.owner.rank), worker.owner, worker.owner.rank),
			width=4,
			features=select_rows(features, task.sources),
		)
		result = transport.compute(worker.owner, packet, output_width=4, supply_inputs=lambda: None)
		locations.append(Location(worker.owner, result.handle))
	return planner, locations


@pytest.mark.parametrize("name", ["sage", "gat", "gcn"])
@pytest.mark.parametrize("coalesced,gather", [(False, False), (True, False), (True, True)])
def test_cyclic_peer_inputs_are_ready_before_models_run_concurrently(name, coalesced, gather):
	graph, features = _graph(), torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10
	with _actors(name, coalesced=coalesced, gather=gather) as (transport, workers):
		planner, locations = _initialize(transport, workers, graph, features, name)
		barrier = Barrier(2, timeout=5)
		executions = []
		for worker in workers:
			original = worker._apply

			def apply(task, values, original=original):
				executions.append(task.layer)
				barrier.wait()
				return original(task, values)

			worker._apply = apply
		prepared, tasks = [], []
		for index, worker in enumerate(workers):
			task = planner.block(2, torch.arange(index * 2, index * 2 + 2))
			tasks.append(task)
			other = locations[1 - index]
			positions = torch.arange(task.sources.numel())
			inputs = [InputChunk(other, task.sources, positions)]
			if gather:
				inputs = [
					InputChunk(other, rows, pos)
					for rows, pos in zip(task.sources.split(2), positions.split(2), strict=True)
				]
			packet = pack_compute(
				task, ComputeAttempt(StateKey(0, 2, index), worker.owner, index + 3), width=4, inputs=inputs
			)

			def supply(inputs=inputs, worker=worker):
				if gather:
					transport.fetch_peer_many([(item.location, item.rows) for item in inputs], 4, worker.owner)
				else:
					for item in inputs:
						transport.fetch_peer(item.location, item.rows, 4, worker.owner)

			prepared.append(transport.prepare(worker.owner, packet, output_width=3, supply_inputs=supply))
		assert executions == [], "PREPARE must not execute the model"
		pending = [transport.launch(task) for task in prepared]
		results = [item.result() for item in pending]
		assert executions == [2, 2]
		with torch.inference_mode():
			model = ModelSpec(name, 4, 4, 3).build()
			for layer in model.layers:
				if hasattr(layer, "set_allow_zero_in_degree"):
					layer.set_allow_zero_in_degree(True)
			expected = model(1, graph, model(0, graph, features))
		for worker, task, result in zip(workers, tasks, results, strict=True):
			actual = transport.read(Location(worker.owner, result.handle), torch.arange(2), 3).result()
			torch.testing.assert_close(actual, expected[task.targets], rtol=2e-5, atol=2e-5)


def test_cancellation_releases_local_input_lease_without_running_model():
	graph, features = _graph(), torch.ones(4, 4)
	with _actors() as (transport, workers):
		planner, locations = _initialize(transport, workers, graph, features, "sage")
		worker = workers[0]
		task = planner.block(2, torch.arange(4))
		packet = pack_compute(
			task,
			ComputeAttempt(StateKey(0, 2, 4), worker.owner, 4),
			width=4,
			inputs=[InputChunk(locations[0], task.sources, torch.arange(4))],
		)
		prepared = transport.prepare(worker.owner, packet, output_width=3, supply_inputs=lambda: None)
		assert worker.store.bytes_used == 64
		worker.store.retire(locations[0].handle)
		assert worker.store.bytes_used == 64, "Prepared local inputs must pin their resident source"
		worker.cancel_prepared(torch.tensor([CANCEL_PREPARED, prepared.packet.header.request, 1, 0, 0, 0, 0, 0]))
		assert worker.store.bytes_used == 0
		assert worker._prepared is None
