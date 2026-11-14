from dataclasses import replace

import pytest
import torch

from lib.graph.anchors import AnchoredRows
from lib.graph.catalog import PriorRows, RowSelection
from lib.graph.ownership import StateKey


def _snapshot():
	return PriorRows(
		3,
		1,
		torch.tensor([1, 3, 4, 6]),
		torch.tensor([True, False, True, True]),
		(
			RowSelection(StateKey(1, 1, 5, "a"), torch.tensor([0, 2]), torch.tensor([10, 20])),
			RowSelection(StateKey(3, 1, 6, "a"), torch.tensor([3]), torch.tensor([4])),
		),
	)


def test_old_row_lookup_preserves_duplicates_and_distinguishes_unchanged_from_missing_history():
	snapshot = _snapshot()
	anchors = AnchoredRows(snapshot)
	actual = anchors.lookup(torch.tensor([6, 0, 1, 4, 3, 6]))
	assert actual.covered.tolist() == [True, False, True, True, True, True]
	assert actual.available.tolist() == [True, False, True, True, False, True]
	assert actual.selections[0].key == snapshot.selections[0].key
	assert actual.selections[0].positions.tolist() == [2, 3]
	assert actual.selections[0].rows.tolist() == [10, 20]
	assert actual.selections[1].positions.tolist() == [0, 5]
	assert actual.selections[1].rows.tolist() == [4, 4]
	assert anchors.snapshot.nodes.data_ptr() == snapshot.nodes.data_ptr()
	assert not anchors.lookup(torch.empty(0, dtype=torch.int64)).selections


@pytest.mark.parametrize("change", ["sort", "missing-route", "duplicate-route", "future"])
def test_old_row_metadata_cannot_certify_missing_duplicate_or_future_routes(change):
	snapshot = _snapshot()
	if change == "sort":
		snapshot = replace(snapshot, nodes=torch.tensor([6, 3, 4, 1]))
	elif change == "missing-route":
		snapshot = replace(snapshot, selections=snapshot.selections[:1])
	elif change == "duplicate-route":
		snapshot = replace(snapshot, selections=(*snapshot.selections, snapshot.selections[0]))
	else:
		group = replace(snapshot.selections[0], key=StateKey(4, 1, 5, "a"))
		snapshot = replace(snapshot, selections=(group, snapshot.selections[1]))
	with pytest.raises(ValueError):
		AnchoredRows(snapshot)


def test_empty_anchor_set_marks_requested_rows_as_outside_its_changed_frontier():
	anchors = AnchoredRows(PriorRows(0, 1, torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.bool), ()))
	actual = anchors.lookup(torch.tensor([0, 2]))
	assert not actual.covered.any() and not actual.available.any() and not actual.selections
