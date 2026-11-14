import pytest
import torch

from lib.graph.catalog import EmbeddingCatalog
from lib.graph.ownership import EmbeddingDirectory, MissingState


@pytest.fixture(params=["h", "a"])
def state(request):
	directory = EmbeddingDirectory()
	owner = directory.register_owner(1)
	return EmbeddingCatalog(8, 2, directory, kind=request.param), owner


def _publish(catalog, owner, layer, nodes):
	key = catalog.reserve(layer, torch.tensor(nodes, dtype=torch.int64), catalog.version)
	catalog.directory.publish(catalog.directory.begin(key, owner), key.chunk)
	return key, catalog.publish(key)


def test_catalog_reorders_rows_and_preserves_duplicate_queries(state):
	catalog, owner = state
	first, _ = _publish(catalog, owner, 1, [4, 1])
	second, _ = _publish(catalog, owner, 1, [0, 3])
	query = torch.tensor([3, 1, 4, 1, 0])
	selections = catalog.selections(1, query, 0)
	assert len(selections) == 2
	for group in selections:
		nodes = torch.tensor([4, 1]) if group.key == first else torch.tensor([0, 3])
		torch.testing.assert_close(nodes[group.rows], query[group.positions])
	assert {group.key for group in selections} == {first, second}
	assert catalog.available(0, query, 0).all()
	assert not catalog.available(2, query, 0).any()


def test_fragmented_catalog_preserves_query_order_after_partial_replacement():
	directory = EmbeddingDirectory()
	owner = directory.register_owner(1)
	catalog = EmbeddingCatalog(1024, 1, directory)
	chunk_nodes = {}
	for nodes in torch.arange(1024).split(16):
		key, _ = _publish(catalog, owner, 1, nodes.tolist())
		chunk_nodes[key] = nodes
	replacement = torch.arange(0, 1024, 7)
	key, _ = _publish(catalog, owner, 1, replacement.tolist())
	chunk_nodes[key] = replacement
	query = torch.randint(1024, (4096,), generator=torch.Generator().manual_seed(71))
	selections = catalog.selections(1, query, 0)
	assert [group.key.chunk for group in selections] == sorted(group.key.chunk for group in selections)
	positions = torch.cat([group.positions for group in selections]).sort().values
	torch.testing.assert_close(positions, torch.arange(query.numel()))
	for group in selections:
		assert (group.positions[1:] > group.positions[:-1]).all()
		torch.testing.assert_close(chunk_nodes[group.key][group.rows], query[group.positions])
		assert torch.isin(query[group.positions], replacement).all() == (group.key == key)


def test_reservation_is_unavailable_until_data_and_row_publication(state):
	catalog, owner = state
	nodes = torch.tensor([4, 1])
	key = catalog.reserve(1, nodes, 0)
	assert catalog.current_row_count(key) == 0
	with pytest.raises(MissingState):
		catalog.publish(key)
	assert not catalog.available(1, nodes, 0).any()
	catalog.directory.publish(catalog.directory.begin(key, owner), 10)
	assert not catalog.available(1, nodes, 0).any()
	assert catalog.publish(key) == ()
	assert catalog.publish(key) == ()
	torch.testing.assert_close(catalog.current_nodes(key), nodes)
	assert catalog.current_row_count(key) == 2


def test_partial_replacement_retires_chunk_only_after_its_last_row_is_replaced(state):
	catalog, owner = state
	old, _ = _publish(catalog, owner, 1, [1, 2, 3])
	_, retired = _publish(catalog, owner, 1, [2])
	assert retired == ()
	torch.testing.assert_close(catalog.current_nodes(old), torch.tensor([1, 3]))
	assert catalog.current_row_count(old) == 2
	_, retired = _publish(catalog, owner, 1, [1, 3])
	assert retired == (old,)
	assert catalog.current_nodes(old).numel() == 0
	assert catalog.current_row_count(old) == 0
	assert catalog.available(1, torch.tensor([1, 2, 3]), 0).all()


def test_discarded_layer_can_be_recomputed_without_affecting_other_layers_or_reservations(state):
	catalog, owner = state
	first, _ = _publish(catalog, owner, 1, [1, 2, 3])
	second, _ = _publish(catalog, owner, 1, [2])
	final, _ = _publish(catalog, owner, 2, [1, 2, 3])
	pending = catalog.reserve(1, torch.tensor([4]), 0)
	with pytest.raises(ValueError, match="version"):
		catalog.discard_layer(1, 1)
	with pytest.raises(ValueError, match="features"):
		catalog.discard_layer(0, 0)
	assert set(catalog.discard_layer(1, 0)) == {first, second}
	assert catalog.discard_layer(1, 0) == ()
	assert not catalog.available(1, torch.arange(8), 0).any()
	assert catalog.selections(2, torch.tensor([1, 2, 3]), 0)[0].key == final
	assert catalog.current_row_count(first) == catalog.current_row_count(second) == 0
	catalog.directory.publish(catalog.directory.begin(pending, owner), pending.chunk)
	catalog.publish(pending)
	catalog.begin_version(1, {}, pending_layers=(1,))
	catalog.finish_analysis(1, torch.tensor([1, 2, 4]), 1)
	_publish(catalog, owner, 1, [1, 2, 4])
	assert catalog.available(1, torch.tensor([1, 2, 4]), 1).all()


def test_version_change_invalidates_affected_rows_and_fences_old_publication(state):
	catalog, owner = state
	old, _ = _publish(catalog, owner, 1, [1, 2, 3])
	pending = catalog.reserve(1, torch.tensor([4]), 0)
	catalog.begin_version(1, {1: torch.tensor([2])})
	torch.testing.assert_close(catalog.available(1, torch.tensor([1, 2, 3]), 1), torch.tensor([True, False, True]))
	with pytest.raises(ValueError, match="version"):
		catalog.publish(pending)
	with pytest.raises(ValueError, match="version"):
		catalog.available(1, torch.tensor([1]), 0)
	with pytest.raises(MissingState):
		catalog.selections(1, torch.tensor([2]), 1)
	new, _ = _publish(catalog, owner, 1, [2])
	assert new.version == 1
	torch.testing.assert_close(catalog.current_nodes(old), torch.tensor([1, 3]))
	assert catalog.available(1, torch.tensor([1, 2, 3]), 1).all()


def test_lost_owner_makes_rows_unavailable_and_recovery_can_replace_them(state):
	catalog, owner = state
	old, _ = _publish(catalog, owner, 1, [1, 2])
	assert catalog.directory.lose_owner(owner) == (old,)
	assert not catalog.available(1, torch.tensor([1, 2]), 0).any()
	replacement = catalog.directory.register_owner(2)
	_, retired = _publish(catalog, replacement, 1, [1, 2])
	assert retired == (old,)
	assert catalog.available(1, torch.tensor([1, 2]), 0).all()


def test_invalid_version_update_is_atomic_and_reservation_validation_is_strict(state):
	catalog, owner = state
	_publish(catalog, owner, 1, [1, 2])
	with pytest.raises(ValueError, match="range"):
		catalog.begin_version(1, {1: torch.tensor([1]), 2: torch.tensor([8])})
	assert catalog.version == 0
	assert catalog.available(1, torch.tensor([1, 2]), 0).all()
	for nodes in (torch.tensor([1, 1]), torch.tensor([9]), torch.ones(2)):
		with pytest.raises(ValueError):
			catalog.reserve(1, nodes, 0)
	with pytest.raises(ValueError):
		catalog.reserve(0, torch.tensor([1]), 0)
	with pytest.raises(ValueError):
		catalog.begin_version(0, {})
	with pytest.raises(ValueError):
		catalog.begin_version(1, {0: torch.tensor([1])})


def test_cancellation_and_empty_row_publication_do_not_leak_catalog_entries(state):
	catalog, owner = state
	pending = catalog.reserve(1, torch.tensor([1]), 0)
	catalog.cancel(pending)
	assert catalog.current_nodes(pending).numel() == 0
	empty, retired = _publish(catalog, owner, 1, [])
	assert retired == (empty,)
	assert catalog.selections(1, torch.empty(0, dtype=torch.int64), 0) == ()
	key, _ = _publish(catalog, owner, 1, [1])
	with pytest.raises(ValueError, match="unpublished"):
		catalog.cancel(key)


def test_pending_analysis_never_reuses_unvalidated_old_rows_but_preserves_current_results(state):
	catalog, owner = state
	_publish(catalog, owner, 1, [1, 2, 3])
	_publish(catalog, owner, 2, [1, 2, 3])
	nodes = torch.tensor([1, 2, 3])
	catalog.begin_version(1, {}, pending_layers=(1, 2))
	assert not catalog.available(1, nodes, 1).any()
	assert not catalog.available(2, nodes, 1).any()
	catalog.finish_analysis(1, torch.tensor([2]), 1)
	torch.testing.assert_close(catalog.available(1, nodes, 1), torch.tensor([True, False, True]))
	_publish(catalog, owner, 2, [1])
	torch.testing.assert_close(catalog.available(2, nodes, 1), torch.tensor([True, False, False]))
	catalog.finish_analysis(2, torch.tensor([1, 2]), 1)
	torch.testing.assert_close(catalog.available(2, nodes, 1), torch.tensor([True, False, True]))
	catalog.begin_version(2, {})


def test_pending_analysis_validation_is_atomic_and_versions_cannot_skip_unfinished_analysis(state):
	catalog, _ = state
	with pytest.raises(ValueError, match="range"):
		catalog.begin_version(1, {}, pending_layers=(0,))
	assert catalog.version == 0
	catalog.begin_version(1, {}, pending_layers=(1,))
	with pytest.raises(RuntimeError, match="incomplete"):
		catalog.begin_version(2, {})
	with pytest.raises(ValueError, match="range"):
		catalog.finish_analysis(1, torch.tensor([9]), 1)
	with pytest.raises(ValueError, match="pending"):
		catalog.finish_analysis(2, torch.tensor([1]), 1)
	catalog.finish_analysis(1, torch.tensor([1]), 1)
	with pytest.raises(ValueError, match="pending"):
		catalog.finish_analysis(1, torch.tensor([1]), 1)
	catalog.begin_version(2, {})


def test_embedding_and_aggregate_catalogs_keep_independent_chunks_and_live_owners():
	directory = EmbeddingDirectory()
	h_owner, a_owner = directory.register_owner(1), directory.register_owner(2)
	h, a = EmbeddingCatalog(8, 2, directory), EmbeddingCatalog(8, 2, directory, kind="a")
	h_key, _ = _publish(h, h_owner, 1, [4, 1])
	a_key, _ = _publish(a, a_owner, 1, [1, 4])
	assert h_key.chunk == a_key.chunk and h_key != a_key
	assert h_key.kind == "h" and a_key.kind == "a"
	assert directory.locate(h_key).owner == h_owner and directory.locate(a_key).owner == a_owner
	with pytest.raises(ValueError, match="state key"):
		h.publish(a_key)
	with pytest.raises(ValueError, match="unpublished"):
		a.cancel(h_key)
	assert not h.current_nodes(a_key).numel()
	assert h.available(1, torch.tensor([1, 4]), 0).all()
	assert a.available(1, torch.tensor([1, 4]), 0).all()
	assert directory.lose_owner(h_owner) == (h_key,)
	assert not h.available(1, torch.tensor([1, 4]), 0).any()
	assert a.available(1, torch.tensor([1, 4]), 0).all()
	a.begin_version(1, {1: torch.tensor([1])})
	torch.testing.assert_close(a.available(1, torch.tensor([1, 4]), 1), torch.tensor([False, True]))
	assert h.version == 0
	with pytest.raises(ValueError, match="Catalog kind"):
		EmbeddingCatalog(8, 2, directory, kind="degree")
