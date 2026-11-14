from unittest.mock import Mock

import dgl
import dgl.nn as dglnn
import pytest
import torch
from torch import nn

from lib.graph.affected import AffectedPlanner
from lib.graph.recompute import PlanTooLarge
from lib.models.operators import full_neighbor_forward


def _graph():
	return dgl.graph(([0, 0, 1, 1, 2, 3, 4], [0, 1, 2, 2, 3, 4, 3]), num_nodes=6)


@torch.inference_mode()
def _predict(model, graph, features):
	states = [features]
	for index, layer in enumerate(model):
		output = layer(graph, states[-1])
		if output.ndim == 3:
			output = output.mean(1)
		states.append(output if index + 1 == len(model) else output.relu())
	return states


@pytest.mark.parametrize("name", ["sage", "gat", "gcn"])
@pytest.mark.parametrize("operation", ["features", "insert", "delete"])
def test_affected_analysis_covers_changes_and_blocks_match_full_inference(name, operation):
	torch.manual_seed(41)
	graph = _graph()
	features = torch.randn(6, 4)
	if name == "sage":
		model = nn.ModuleList([dglnn.SAGEConv(4, 4, "mean") for _ in range(3)])
	elif name == "gat":
		model = nn.ModuleList([dglnn.GATConv(4, 4, 2, residual=True, allow_zero_in_degree=True) for _ in range(3)])
	else:
		model = nn.ModuleList([dglnn.GraphConv(4, 4, norm="both", allow_zero_in_degree=True) for _ in range(3)])
		features = features.abs() + 0.1
		with torch.no_grad():
			for parameter in model.parameters():
				parameter.fill_(0.2)
	model.eval()
	before = _predict(model, graph, features)
	changed = targets = sources = None
	if operation == "features":
		changed = torch.tensor([1, 5])
		features[changed] += 0.75
	elif operation == "insert":
		sources, targets = torch.tensor([2, 4]), torch.tensor([0, 5])
		graph.add_edges(sources, targets)
	else:
		sources, targets = graph.find_edges(torch.tensor([1]))
		graph.remove_edges(torch.tensor([1]))
	after = _predict(model, graph, features)
	assert any((old - new).abs().max() > 1e-6 for old, new in zip(before[1:], after[1:], strict=True))
	planner = AffectedPlanner(graph, 3, source_degree_sensitive=name == "gcn")
	for affected in planner.analyze(
		changed_features=changed, edge_targets=targets, degree_sources=sources if name == "gcn" else None
	):
		actually_changed = (after[affected.layer] - before[affected.layer]).abs().amax(1) > 1e-6
		covered = torch.zeros(6, dtype=torch.bool)
		covered[affected.targets] = True
		assert not (actually_changed & ~covered).any()
		task = planner.block(affected.layer, affected.targets)
		with torch.inference_mode():
			actual = full_neighbor_forward(model[affected.layer - 1], task, after[affected.layer - 1][task.sources])
			if actual.ndim == 3:
				actual = actual.mean(1)
			if affected.layer != 3:
				actual = actual.relu()
		torch.testing.assert_close(actual, after[affected.layer][task.targets], rtol=2e-5, atol=2e-5)


def test_analysis_is_lazy_and_edge_changes_are_included_at_every_layer(monkeypatch):
	graph = _graph()
	read = Mock(wraps=graph.out_edges)
	monkeypatch.setattr(graph, "out_edges", read)
	planner = AffectedPlanner(graph, 3)
	iterator = planner.analyze(edge_targets=torch.tensor([5]))
	read.assert_not_called()
	first = next(iterator)
	assert first.targets.tolist() == [5]
	read.assert_not_called()
	second = next(iterator)
	assert second.targets.tolist() == [5]
	assert read.call_count == 1
	assert next(iterator).targets.tolist() == [5]
	assert read.call_count == 2


def test_empty_updates_and_invalid_metadata_are_handled_before_allocation(monkeypatch):
	planner = AffectedPlanner(_graph(), 2)
	assert all(not item.targets.numel() for item in planner.analyze())
	for nodes in (torch.ones(2), torch.tensor([-1]), torch.tensor([6])):
		with pytest.raises(ValueError, match="node ID"):
			next(planner.analyze(changed_features=nodes))
	with pytest.raises(ValueError, match="unique"):
		planner.block(1, torch.tensor([1, 1]))
	with pytest.raises(ValueError, match="layer"):
		planner.block(3, torch.tensor([1]))
	allocate = Mock(side_effect=AssertionError("Must enforce budget first"))
	monkeypatch.setattr(dgl, "in_subgraph", allocate)
	with pytest.raises(PlanTooLarge):
		planner.block(1, torch.tensor([1]), max_edges=0)
	allocate.assert_not_called()
