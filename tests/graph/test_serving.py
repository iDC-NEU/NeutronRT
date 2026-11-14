from dataclasses import replace
from unittest.mock import Mock

import dgl
import dgl.nn as dglnn
import pytest
import torch
from torch import nn

from lib.graph.data_plane import LocalEmbeddingPlane
from lib.graph.methods import neighborhood_policy
from lib.graph.ownership import MissingState
from lib.graph.serving import EmbeddingService, select_rows
from lib.models.operators import full_neighbor_forward


def _setup(name, devices=("cpu", "cpu"), *, batch_size=8192):
	torch.manual_seed(17)
	graph = dgl.graph(([0, 0, 1, 1, 2, 3, 4], [0, 1, 2, 2, 3, 4, 3]), num_nodes=6)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	layers = []
	for index in range(2):
		width = 3 if index == 1 else 4
		if name == "sage":
			layers.append(dglnn.SAGEConv(4, width, "mean"))
		elif name == "gat":
			layers.append(dglnn.GATConv(4, width, 2, residual=True, allow_zero_in_degree=True))
		else:
			layers.append(dglnn.GraphConv(4, width, norm="both", allow_zero_in_degree=True))
	base = nn.ModuleList(layers).eval()
	plane = LocalEmbeddingPlane()
	storage = plane.add_node(0, "cpu", 1024 * 1024)
	workers = [plane.add_node(index + 1, device, 1024 * 1024) for index, device in enumerate(devices)]
	models = {}
	for worker in workers:
		import copy

		models[worker.owner] = copy.deepcopy(base).to(worker.device)

	def apply(worker, task, inputs):
		task = replace(task, block=task.block.to(inputs.device))
		output = full_neighbor_forward(models[worker.owner][task.layer - 1], task, inputs)
		if output.ndim == 3:
			output = output.mean(1)
		return output.relu() if task.layer == 1 else output

	service = EmbeddingService(
		graph, features, (4, 4, 3), plane, workers, apply, batch_size=batch_size, source_degree_sensitive=name == "gcn"
	)
	service._reference_model = base
	with torch.inference_mode():
		full = features
		for index, layer in enumerate(base):
			full = layer(graph, full)
			if full.ndim == 3:
				full = full.mean(1)
			if index == 0:
				full = full.relu()
	return service, plane, storage, full


@pytest.mark.parametrize("name", ["sage", "gat"])
@pytest.mark.parametrize("batch_size", [2, 8192])
def test_owned_recomputation_matches_full_inference_without_eager_master_copy(name, batch_size):
	service, plane, storage, expected = _setup(name, batch_size=batch_size)
	try:
		pointer = service.features.data_ptr()
		query = torch.tensor([5, 3, 3, 0])
		service.ensure(query)
		assert storage.store.bytes_used == 0
		assert service.features.data_ptr() == pointer
		tasks = service.counters.tasks
		with service.query(query, storage.owner) as values:
			torch.testing.assert_close(values, expected[query], atol=2e-5, rtol=2e-5)
		assert service.counters.tasks == tasks
		with service.query(torch.empty(0, dtype=torch.int64), storage.owner) as values:
			assert values.shape == (0, 3)
	finally:
		plane.close()


@pytest.mark.parametrize("name", ["sage", "gat"])
def test_missing_gpu_state_is_recomputed_using_surviving_intermediate_inputs(name):
	service, plane, storage, expected = _setup(name)
	try:
		query = torch.tensor([5, 3, 0])
		service.ensure(query)
		assert storage.store.bytes_used == 0
		before = service.counters.tasks
		owner = service.catalog.selections(2, query, 0)[0].key
		producer = plane.directory.locate(owner).owner
		assert service.recover(producer)
		assert service.counters.tasks == before + 1
		assert service.counters.recovery_rows == query.numel()
		assert storage.store.bytes_used == 0
		with service.query(query, storage.owner) as values:
			torch.testing.assert_close(values, expected[query], atol=2e-5, rtol=2e-5)
	finally:
		plane.close()


def test_compute_failure_cancels_unpublished_state_and_can_be_retried():
	service, plane, storage, expected = _setup("sage")
	try:
		original = service.apply
		service.apply = Mock(side_effect=RuntimeError("injected compute failure"))
		query = torch.tensor([1, 3])
		with pytest.raises(RuntimeError, match="injected"):
			service.ensure(query)
		assert storage.store.bytes_used == 0
		assert not service.catalog.available(2, query, 0).any()
		service.apply = original
		with service.query(query, storage.owner) as values:
			torch.testing.assert_close(values, expected[query])
	finally:
		plane.close()


@pytest.mark.parametrize("method", ["NrtInc", "RTEC-Full", "RTEC-NS"])
@pytest.mark.parametrize("name", ["sage", "gat", "gcn"])
def test_service_splits_requests_by_dependency_edges_without_changing_model_outputs(method, name):
	service, plane, storage, expected = _setup(name)
	service.planner.policy = neighborhood_policy(method, fanout=5)
	service.planner.max_edges = 10
	original = service._execute_plan
	executed = []

	def execute(plan):
		assert sum(task.block.num_edges() for task in plan.tasks) <= 10
		executed.append(plan)
		return original(plan)

	service._execute_plan = execute
	nodes = torch.tensor([5, 3, 2, 4, 1, 0, 3])
	try:
		with service.query(nodes, storage.owner) as actual:
			torch.testing.assert_close(actual, expected[nodes], atol=2e-5, rtol=2e-5)
		assert len(executed) > 1
		service.insert_edges(torch.tensor([2]), torch.tensor([0]))
		with service.query(nodes, storage.owner) as actual:
			torch.testing.assert_close(actual, _current_reference(service)[nodes], atol=2e-5, rtol=2e-5)
	finally:
		plane.close()


@pytest.mark.parametrize("method", ["RTEC-Full", "RTEC-NS"])
@pytest.mark.parametrize("name", ["sage", "gat"])
def test_nonreusing_plans_release_dependencies_after_draining_existing_readers(method, name):
	service, plane, storage, expected = _setup(name, batch_size=2)
	service.planner.policy = neighborhood_policy(method, fanout=5)
	nodes = torch.arange(6)
	try:
		with service.query(nodes, storage.owner, layer=1) as borrowed:
			original = borrowed.clone()
			keys = tuple(group.key for group in service.catalog.selections(1, nodes, 0))
			service.ensure(nodes)
			assert not service.catalog.available(1, nodes, 0).any()
			torch.testing.assert_close(borrowed, original)
			for key in keys:
				with pytest.raises(MissingState):
					plane.directory.locate(key)
		tasks = service.counters.tasks
		with service.query(nodes, storage.owner) as actual:
			torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
		assert service.counters.tasks == tasks
		service.insert_edges(torch.tensor([2]), torch.tensor([0]))
		with service.query(nodes, storage.owner) as actual:
			torch.testing.assert_close(actual, _current_reference(service), atol=2e-5, rtol=2e-5)
		assert not service.catalog.available(1, nodes, service.catalog.version).any()
	finally:
		plane.close()


def test_failed_nonreusing_plan_releases_published_dependencies_before_retry():
	service, plane, storage, expected = _setup("sage", batch_size=2)
	service.planner.policy = neighborhood_policy("RTEC-Full")
	apply = service.apply

	def fail_final(worker, task, inputs):
		if task.layer == 2:
			raise RuntimeError("Injected final-layer failure")
		return apply(worker, task, inputs)

	service.apply = fail_final
	nodes = torch.arange(6)
	try:
		with pytest.raises(RuntimeError, match="final-layer"):
			service.ensure(nodes)
		assert not service.catalog.available(1, nodes, 0).any()
		assert not service.catalog.available(2, nodes, 0).any()
		assert not sum(node.store.bytes_used for node in service.workers)
		service.apply = apply
		with service.query(nodes, storage.owner) as actual:
			torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
	finally:
		plane.close()


def test_row_selection_preserves_views_when_possible():
	tensor = torch.arange(24).reshape(6, 4)
	view = select_rows(tensor, torch.tensor([2, 3, 4]))
	assert view.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
	torch.testing.assert_close(view, tensor[2:5])
	torch.testing.assert_close(select_rows(tensor, torch.tensor([4, 1, 4])), tensor[[4, 1, 4]])
	assert select_rows(tensor, torch.empty(0, dtype=torch.int64)).shape == (0, 4)


@torch.inference_mode()
def _current_reference(service):
	values = service.features
	for index, layer in enumerate(service._reference_model):
		values = layer(service.graph, values)
		if values.ndim == 3:
			values = values.mean(1)
		if index + 2 < len(service.widths):
			values = values.relu()
	return values


@pytest.mark.parametrize("name", ["sage", "gat", "gcn"])
@pytest.mark.parametrize("batch_size", [2, 8192])
@pytest.mark.parametrize(
	"devices",
	[
		("cpu", "cpu"),
		pytest.param(
			("cuda:0", "cuda:1"),
			marks=pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two visible GPUs"),
		),
	],
)
def test_real_updates_match_full_inference_and_keep_the_feature_storage(name, batch_size, devices):
	service, plane, storage, _ = _setup(name, devices=devices, batch_size=batch_size)
	try:
		query = torch.arange(6)
		service.ensure(query)
		pointer = service.features.data_ptr()
		results = [service.change_features(torch.tensor([1, 5]), service.features[[1, 5]] + 0.75)]
		with service.query(query, storage.owner) as actual:
			torch.testing.assert_close(actual, _current_reference(service), rtol=2e-5, atol=2e-5)
		results.append(service.insert_edges(torch.tensor([2, 4]), torch.tensor([0, 5])))
		with service.query(query, storage.owner) as actual:
			torch.testing.assert_close(actual, _current_reference(service), rtol=2e-5, atol=2e-5)
		results.append(service.delete_edges(torch.tensor([1]), torch.tensor([2])))
		assert not service.graph.has_edges_between(1, 2)
		with service.query(query, storage.owner) as actual:
			torch.testing.assert_close(actual, _current_reference(service), rtol=2e-5, atol=2e-5)
		assert [result.version for result in results] == [1, 2, 3]
		assert all(len(result.affected) == 2 for result in results)
		assert service.features.data_ptr() == pointer
		assert service.counters.updates == 3
		assert service.counters.input_changes == 5
	finally:
		plane.close()


def test_invalid_updates_leave_root_state_and_versions_unchanged():
	service, plane, _, _ = _setup("sage")
	try:
		before = service.features.clone()
		for nodes, values in (
			(torch.tensor([1, 1]), torch.ones(2, 4)),
			(torch.tensor([8]), torch.ones(1, 4)),
			(torch.tensor([1]), torch.ones(1, 3)),
			(torch.tensor([1]), torch.ones(1, 4, dtype=torch.float64)),
		):
			with pytest.raises(ValueError):
				service.change_features(nodes, values)
		for sources, targets in ((torch.tensor([0]), torch.tensor([1, 2])), (torch.tensor([8]), torch.tensor([0]))):
			with pytest.raises(ValueError):
				service.insert_edges(sources, targets)
		assert service.catalog.version == 0
		assert service.graph.num_edges() == 7
		torch.testing.assert_close(service.features, before)
	finally:
		plane.close()


def test_feature_updates_snapshot_only_overlapping_input_rows():
	service, plane, storage, _ = _setup("sage")
	try:
		expected_rows = service.features[:2].clone()
		pointer = service.features.data_ptr()
		service.change_features(torch.tensor([1, 0]), service.features[:2])
		torch.testing.assert_close(service.features[[1, 0]], expected_rows)
		assert service.features.data_ptr() == pointer
		with service.query(torch.arange(6), storage.owner) as actual:
			torch.testing.assert_close(actual, _current_reference(service))
	finally:
		plane.close()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two visible GPUs")
@pytest.mark.parametrize("name", ["sage", "gat"])
def test_real_gpu_ownership_and_recomputation_before_writeback(name):
	service, plane, storage, expected = _setup(name, devices=("cuda:0", "cuda:1"))
	try:
		query = torch.tensor([0, 3, 5])
		service.ensure(query)
		assert storage.store.bytes_used == 0
		key = service.catalog.selections(2, query, 0)[0].key
		producer = plane.directory.locate(key).owner
		service.recover(producer)
		assert storage.store.bytes_used == 0
		with service.query(query, storage.owner) as values:
			torch.testing.assert_close(values, expected[query], atol=2e-5, rtol=2e-5)
	finally:
		plane.close()
