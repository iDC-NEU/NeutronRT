from types import SimpleNamespace

import dgl
import torch

from lib.graph.capacity import CapacityPolicy
from lib.graph.mutation import apply_prepared_mutation, prepare_mutation, prepare_structural_delta


def _request(kind, first, second):
	return SimpleNamespace(kind=kind, first=first, second=second)


def test_prepared_mutation_resolves_parallel_delete_once_and_preserves_signed_counts():
	graph = dgl.graph((torch.tensor([0, 0, 0, 1]), torch.tensor([1, 1, 1, 1])), num_nodes=3)
	features = torch.zeros(3, 2)
	insert = prepare_mutation(graph, features, _request("insert", torch.tensor([2, 2, 2]), torch.tensor([2, 2, 1])))
	assert insert.edge_weights.numel() == 0
	assert prepare_structural_delta(insert).edge_weights.tolist() == [1, 2]
	apply_prepared_mutation(graph, features, insert)
	assert graph.num_edges() == 7
	delete = prepare_mutation(graph, features, _request("delete", torch.tensor([0, 2, 2]), torch.tensor([1, 2, 2])))
	assert delete.edge_weights.numel() == 0
	assert prepare_structural_delta(delete).edge_weights.tolist() == [-3, -2]
	assert delete.edge_ids.numel() == 5
	apply_prepared_mutation(graph, features, delete)
	assert graph.num_edges() == 2
	absent = prepare_mutation(graph, features, _request("delete", torch.tensor([0]), torch.tensor([2])))
	assert not absent.topology_changed and absent.edge_ids.numel() == 0


def test_capacity_policy_does_not_probe_device_inside_reserved_envelope():
	policy = CapacityPolicy.create(
		capacity_bytes=10_000,
		device=torch.device("cpu"),
		initial_bytes=1_000,
		initial_edges=10,
		estimate=lambda edges: 1_000 + edges,
	)
	for edges in range(11, 20):
		policy.admit(edges, lambda value: 1_000 + value)
	assert policy.physical_checks == 0


def test_cuda_capacity_policy_only_rechecks_when_envelope_grows(monkeypatch):
	calls = []
	monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: calls.append(device) or (20_000, 20_000))
	policy = CapacityPolicy.create(10_000, torch.device("cuda:0"), 1_000, 10, lambda edges: 1_000 + edges)
	for edges in range(11, 20):
		policy.admit(edges, lambda value: 1_000 + value)
	assert len(calls) == policy.physical_checks == 1
	policy.admit(5_000, lambda value: 1_000 + value)
	assert len(calls) == policy.physical_checks == 2
