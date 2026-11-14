"""Accuracy/error evaluation for captured query rows, outside performance windows."""

from typing import Optional

import torch
from torch import Tensor

from lib.profiling.experiment import sample_ids


def verification_sample(graph, count: int, generator: torch.Generator) -> Tensor:
	"""Cover uniform rows and degree extremes in an independent CPU oracle."""
	if graph.device.type != "cpu" or count < 0:
		raise ValueError("Verification sampling requires a CPU graph and nonnegative row count")
	count = min(count, graph.num_nodes())
	if count == graph.num_nodes():
		return torch.arange(count, dtype=torch.int64)
	fixed = sample_ids(graph.num_nodes(), count, generator)
	if count < 8:
		return fixed
	extremes = count // 8
	hubs = torch.cat((graph.in_degrees().topk(extremes).indices, graph.out_degrees().topk(extremes).indices)).unique()
	return torch.cat((hubs, fixed[~torch.isin(fixed, hubs)]))[:count]


def targeted_sample(graph, fixed: Tensor, changed: Tensor, *, depth: int, seed: int) -> Tensor:
	"""Keep uniform coverage and probe modified endpoints and bounded downstream rows.

	The independent oracle's CSR supplies at most four successors per selected
	vertex. Scratch is proportional to the requested verification count, never E.
	Call outside measured windows; CSR construction may index the complete graph.
	"""
	budget = fixed.numel()
	if budget >= graph.num_nodes() or budget < 4 or not changed.numel():
		return fixed
	generator = torch.Generator().manual_seed(seed)
	changed = changed.unique()
	selected = changed[sample_ids(changed.numel(), min(changed.numel(), budget // 4), generator)]
	frontier = selected
	offsets, neighbors, _ = graph.adj_tensors("csr")
	for _ in range(depth):
		rows = []
		for node in frontier.tolist():
			start, stop = int(offsets[node]), int(offsets[node + 1])
			positions = sample_ids(stop - start, min(4, stop - start), generator)
			rows.append(neighbors[start + positions])
		if not rows:
			break
		candidates = torch.cat(rows).unique()
		candidates = candidates[~torch.isin(candidates, selected)]
		remaining = budget // 2 - selected.numel()
		if not remaining or not candidates.numel():
			break
		frontier = candidates[sample_ids(candidates.numel(), min(remaining, candidates.numel()), generator)]
		selected = torch.cat((selected, frontier))
	return torch.cat((selected, fixed[~torch.isin(fixed, selected)]))[:budget]


def query_error(
	actual: Tensor, fresh: Tensor, *, labels: Optional[Tensor] = None, test_mask: Optional[Tensor] = None
) -> dict:
	if actual.shape != fresh.shape or actual.ndim != 2 or actual.device.type != "cpu" or fresh.device.type != "cpu":
		raise ValueError("Query comparisons need matching CPU matrices")
	if actual.dtype != torch.float32 or fresh.dtype != torch.float32 or actual.shape[1] < 1:
		raise ValueError("Query comparisons need float32 class scores")
	if not actual.isfinite().all() or not fresh.isfinite().all():
		raise ValueError("Nonfinite query output")
	rows = actual.shape[0]
	predicted, expected = actual.argmax(1), fresh.argmax(1)
	error = (actual - fresh).abs()
	report: dict = {
		"rows": rows,
		"max_abs_error": float(error.max()) if error.numel() else 0.0,
		"mean_abs_error": float(error.mean()) if error.numel() else 0.0,
		"mean_squared_error": float((actual.double() - fresh.double()).square().mean()) if error.numel() else 0.0,
		"prediction_disagreement": float((predicted != expected).float().mean()) if rows else 0.0,
	}
	if labels is not None:
		if labels.dtype != torch.int64 or labels.shape != (rows,) or labels.device.type != "cpu":
			raise ValueError("Labels must match query rows")
		valid = (labels >= 0) & (labels < actual.shape[1])
		if test_mask is not None:
			if test_mask.shape != (rows,) or test_mask.dtype != torch.bool or test_mask.device.type != "cpu":
				raise ValueError("Test mask must match query rows")
			valid &= test_mask
		count = int(valid.sum())
		candidate_accuracy = float((predicted[valid] == labels[valid]).float().mean()) if count else None
		fresh_accuracy = float((expected[valid] == labels[valid]).float().mean()) if count else None
		report["labelled_test_rows"] = count
		report["sample_test_accuracy"] = candidate_accuracy
		report["fresh_sample_test_accuracy"] = fresh_accuracy
		report["accuracy_loss_percentage_points"] = (
			(fresh_accuracy - candidate_accuracy) * 100
			if fresh_accuracy is not None and candidate_accuracy is not None
			else None
		)
	return report
