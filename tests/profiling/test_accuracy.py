import dgl
import pytest
import torch

from lib.profiling.accuracy import query_error, targeted_sample, verification_sample


def test_oracle_sample_covers_degree_extremes_and_preserves_them_in_update_probes():
	graph = dgl.graph(
		(
			torch.cat((torch.arange(1, 65), torch.full((128,), 70))),
			torch.cat((torch.zeros(64, dtype=torch.int64), torch.arange(128))),
		),
		num_nodes=128,
	)
	fixed = verification_sample(graph, 32, torch.Generator().manual_seed(71))
	assert fixed.numel() == fixed.unique().numel() == 32
	assert {0, 70}.issubset(set(fixed.tolist()))
	assert torch.equal(fixed, verification_sample(graph, 32, torch.Generator().manual_seed(71)))
	probed = targeted_sample(graph, fixed, torch.arange(100, 128), depth=2, seed=71)
	assert probed.numel() == probed.unique().numel() == 32
	assert {0, 70}.issubset(set(probed.tolist()))
	assert set(range(100, 128)).intersection(probed.tolist())
	assert torch.equal(verification_sample(graph, 128, torch.Generator()), torch.arange(128))


def test_large_graph_probe_includes_modified_rows_and_downstream_dependencies():
	import dgl

	graph = dgl.graph((torch.arange(9999), torch.arange(1, 10000)), num_nodes=10000)
	fixed = torch.arange(8)
	selected = targeted_sample(graph, fixed, torch.tensor([5000]), depth=2, seed=71)
	assert selected.numel() == selected.unique().numel() == 8
	assert {5000, 5001, 5002}.issubset(set(selected.tolist()))
	assert set(range(5)).issubset(set(selected.tolist()))
	assert torch.equal(selected, targeted_sample(graph, fixed, torch.tensor([5000]), depth=2, seed=71))
	assert torch.equal(fixed, torch.arange(8))


def test_accuracy_loss_is_measured_against_fresh_logits_on_identical_labelled_test_rows():
	fresh = torch.tensor([[4.0, 1], [1.0, 3], [2.0, 1], [0.0, 1]])
	actual = fresh.clone()
	actual[0] = torch.tensor([0.0, 1])
	actual[3] = torch.tensor([2.0, 1])
	report = query_error(
		actual, fresh, labels=torch.tensor([0, 1, -1, 1]), test_mask=torch.tensor([True, True, True, False])
	)
	assert report["max_abs_error"] == 4 and report["prediction_disagreement"] == 0.5
	assert report["mean_squared_error"] == 2.5
	assert report["labelled_test_rows"] == 2 and report["sample_test_accuracy"] == 0.5
	assert report["fresh_sample_test_accuracy"] == 1 and report["accuracy_loss_percentage_points"] == 50
	assert query_error(fresh, fresh)["prediction_disagreement"] == 0
	assert query_error(fresh[:0], fresh[:0])["max_abs_error"] == 0


def test_nonfinite_outputs_and_mismatched_labels_are_rejected_before_reporting_a_tradeoff():
	values = torch.ones(2, 3)
	with pytest.raises(ValueError, match="Nonfinite"):
		query_error(values * float("nan"), values)
	with pytest.raises(ValueError, match="Labels"):
		query_error(values, values, labels=torch.tensor([1]))
	report = query_error(values, values, labels=torch.tensor([-1, -1]))
	assert report["labelled_test_rows"] == 0 and report["accuracy_loss_percentage_points"] is None
