from dataclasses import replace

import dgl
import pytest
import torch

from lib.models.spec import ModelSpec
from lib.profiling.experiment import ProfileCase, make_trace
from lib.rpc import inference_pb2 as wire
from lib.rpc.service import _to_wire
from lib.runtime.config import RuntimeFeatures
from scripts.evaluate.profile import check_query_samples, reference_oracles
from scripts.validate.inference import reference


def test_reference_replay_keeps_only_selected_rows_and_covers_every_observed_version():
	graph = dgl.graph(([0, 1, 2], [1, 2, 0]), num_nodes=4)
	features = torch.arange(16, dtype=torch.float32).reshape(4, 4)
	model = ModelSpec("sage", 4, 4, 3).build()
	trace = make_trace(features, updates=6, changes=1, pool_size=4, mixed=True)
	nodes = torch.tensor([2, 0, 2])
	oracles = reference_oracles(graph, features, trace, nodes, model, every_version=True)
	assert list(oracles) == list(range(7))
	assert all(values.shape == (3, 3) and values.untyped_storage().nbytes() == 36 for values in oracles.values())
	torch.testing.assert_close(oracles[6], reference(model, graph, features, nodes))


def test_sample_validation_enforces_per_row_shared_bound_without_promising_whole_tensor_snapshots():
	oracles = {version: torch.tensor([[float(version), 1], [float(10 * version), 2]]) for version in range(5)}
	case = ProfileCase(
		"bounded", query_staleness=2, switches=replace(RuntimeFeatures.baseline(), bounded_staleness=True)
	)
	values = torch.stack([oracles[2][0], oracles[3][1]])
	reply = wire.QueryResponse(version=4, max_age=2, min_fresh_version=2, values=_to_wire(values))
	metrics, samples = check_query_samples(
		[reply], oracles, case, torch.tensor([0, 0]), torch.ones(2, dtype=torch.bool)
	)
	assert metrics[0]["max_age"] == 2 and metrics[0]["max_abs_error"] == 10
	torch.testing.assert_close(samples[0]["values"], values)
	for change in ("outside", "budget", "certificate"):
		bad = wire.QueryResponse()
		bad.CopyFrom(reply)
		if change == "outside":
			bad.values.CopyFrom(_to_wire(oracles[0]))
		elif change == "budget":
			bad.max_age = 3
			bad.min_fresh_version = 1
		else:
			bad.min_fresh_version = 0
		with pytest.raises(RuntimeError):
			check_query_samples([bad], oracles, case, None, None)
