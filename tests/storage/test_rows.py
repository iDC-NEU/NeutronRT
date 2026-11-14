from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lib.storage.rows import MappedRows


def _rows(folder: Path, *, start=0, stop=4, budget=1024):
	# Includes parallel edges, a self-loop, and an isolated final vertex.
	np.save(folder / "offsets.npy", np.array([0, 3, 4, 5, 5], dtype=np.int64))
	np.save(folder / "neighbors.npy", np.array([1, 1, 0, 2, 3], dtype=np.int64))
	return MappedRows(folder / "offsets.npy", folder / "neighbors.npy", start, stop, max_overlay_bytes=budget)


def test_base_rows_borrow_read_only_mapping_and_preserve_duplicates(tmp_path):
	rows = _rows(tmp_path)
	assert isinstance(rows.neighbors, np.memmap)
	assert np.shares_memory(rows.row(0), rows.neighbors)
	assert not rows.row(0).flags.writeable and rows.bytes_used == 0
	np.testing.assert_array_equal(rows.row(0), [1, 1, 0])
	np.testing.assert_array_equal(rows.degrees(np.array([3, 0, 0, 2], dtype=np.int64)), [0, 3, 3, 1])
	np.testing.assert_array_equal(rows.multiplicities(np.array([0, 0, 0, 3]), np.array([1, 1, 0, 2])), [2, 2, 1, 0])
	with pytest.raises(ValueError):
		rows.row(0)[0] = 9


def test_preparation_is_invisible_abort_discards_it_and_commit_keeps_old_borrowed_rows(tmp_path):
	rows = _rows(tmp_path)
	old = rows.row(0)
	prepared = rows.prepare(np.array([0, 0]), np.array([3, 1]), delete=False)
	np.testing.assert_array_equal(rows.row(0), [1, 1, 0])
	assert prepared.edge_delta == 2
	rows.abort()
	assert rows.bytes_used == 0
	with pytest.raises(ValueError, match="prepared"):
		rows.commit(prepared)
	prepared = rows.prepare(np.array([0, 0]), np.array([3, 1]), delete=False)
	rows.commit(prepared)
	np.testing.assert_array_equal(rows.row(0), [1, 1, 0, 3, 1])
	np.testing.assert_array_equal(rows.multiplicities(np.array([0, 0]), np.array([1, 3])), [3, 1])
	np.testing.assert_array_equal(old, [1, 1, 0])
	assert not rows.row(0).flags.writeable
	assert rows.bytes_used == 40 and set(rows.overlays) == {0}
	assert np.shares_memory(rows.row(1), rows.neighbors)


def test_degree_queries_follow_only_committed_rows_including_empty_replacements(tmp_path):
	rows = _rows(tmp_path)
	query = np.array([3, 0, 1, 3, 0, 2], dtype=np.int64)
	first = rows.prepare(np.array([0, 3, 1]), np.array([3, 0, 0]), delete=False)
	np.testing.assert_array_equal(rows.degrees(query), [0, 3, 1, 0, 3, 1])
	rows.abort()
	np.testing.assert_array_equal(rows.degrees(query), [0, 3, 1, 0, 3, 1])
	with pytest.raises(ValueError):
		rows.commit(first)
	rows.commit(rows.prepare(np.array([0, 3, 1]), np.array([3, 0, 0]), delete=False))
	np.testing.assert_array_equal(rows.degrees(query), [1, 4, 2, 1, 4, 1])
	rows.commit(rows.prepare(np.array([3, 0, 2]), np.array([0, 1, 3]), delete=True))
	np.testing.assert_array_equal(rows.degrees(query), [0, 2, 2, 0, 2, 0])
	rows.commit(rows.prepare(np.array([3]), np.array([2]), delete=False))
	np.testing.assert_array_equal(rows.degrees(query), [1, 2, 2, 1, 2, 0])


def test_delete_removes_all_parallel_edges_and_absent_pairs_are_noops(tmp_path):
	rows = _rows(tmp_path)
	prepared = rows.prepare(np.array([0, 0, 1]), np.array([1, 1, 3]), delete=True)
	assert prepared.edge_delta == -2
	rows.commit(prepared)
	np.testing.assert_array_equal(rows.row(0), [0])
	np.testing.assert_array_equal(rows.exists(np.array([0, 0, 1]), np.array([1, 0, 2])), [False, True, True])
	np.testing.assert_array_equal(rows.multiplicities(np.array([0, 0, 1]), np.array([1, 0, 2])), [0, 1, 1])
	assert rows.bytes_used == 8
	prepared = rows.prepare(np.array([0]), np.array([1]), delete=True)
	assert prepared.edge_delta == 0 and not prepared.rows
	rows.commit(prepared)
	assert rows.bytes_used == 8


def test_budget_failure_has_no_visible_or_pending_mutation(tmp_path):
	rows = _rows(tmp_path, budget=16)
	with pytest.raises(MemoryError, match="overlay budget"):
		rows.prepare(np.array([0]), np.array([3]), delete=False)
	assert not rows.overlays and rows.bytes_used == 0
	prepared = rows.prepare(np.array([1]), np.array([3]), delete=False)
	rows.commit(prepared)
	assert rows.bytes_used == 16
	assert set(rows.overlays) == {1}


def test_validation_and_forged_commits_cannot_change_storage(tmp_path):
	rows = _rows(tmp_path, start=1, stop=3)
	for nodes, others in (
		(np.array([0]), np.array([1])),
		(np.array([3]), np.array([1])),
		(np.array([1]), np.array([4])),
		(np.array([1]), np.array([0, 1])),
		(np.array([1.0]), np.array([1])),
	):
		with pytest.raises(ValueError):
			rows.prepare(nodes, others, delete=False)
	prepared = rows.prepare(np.array([1]), np.array([0]), delete=False)
	with pytest.raises(RuntimeError, match="already prepared"):
		rows.prepare(np.array([2]), np.array([0]), delete=True)
	with pytest.raises(ValueError):
		rows.commit(replace(prepared))
	rows.commit(prepared)
	assert rows.bytes_used == 16


def test_empty_partition_and_empty_update_are_valid(tmp_path):
	rows = _rows(tmp_path, start=2, stop=2)
	prepared = rows.prepare(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), delete=False)
	rows.commit(prepared)
	assert rows.degrees(np.empty(0, dtype=np.int64)).size == 0
	assert rows.multiplicities(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)).size == 0
	assert rows.bytes_used == 0


def test_multiplicities_count_every_part_of_a_high_degree_mapped_row(tmp_path):
	count = 131077
	np.save(tmp_path / "offsets.npy", np.array([0, count, count, count, count], dtype=np.int64))
	values = np.arange(count, dtype=np.int64) % 3
	np.save(tmp_path / "neighbors.npy", values)
	rows = MappedRows(tmp_path / "offsets.npy", tmp_path / "neighbors.npy", 0, 4, max_overlay_bytes=0)
	np.testing.assert_array_equal(
		rows.multiplicities(np.array([0, 0, 0, 3]), np.array([0, 1, 2, 0])),
		[np.count_nonzero(values == 0), np.count_nonzero(values == 1), np.count_nonzero(values == 2), 0],
	)
	assert np.shares_memory(rows.row(0), rows.neighbors) and not rows.overlays and rows.bytes_used == 0
