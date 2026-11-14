import pytest
import torch

from lib.graph.catalog import EmbeddingCatalog
from lib.graph.ownership import EmbeddingDirectory, MissingState
from tests.graph.test_catalog import _publish


def test_shared_output_age_starts_at_first_dirty_version_and_unaffected_old_rows_remain_fresh():
	directory = EmbeddingDirectory()
	owner = directory.register_owner(1)
	catalog = EmbeddingCatalog(4, 2, directory)
	catalog.enable_staleness()
	for layer in (1, 2):
		_publish(catalog, owner, layer, [0, 1, 2])
	query = torch.tensor([0, 1, 2, 3])
	for version in (1, 2, 3):
		catalog.begin_version(version, {1: torch.tensor([1]), 2: torch.tensor([1])})
		assert catalog.ages(2, query, version).tolist() == [0, version, 0, 0]
		assert catalog.servable(2, query, version, max_staleness=2).tolist() == [True, version <= 2, True, False]
	assert catalog.relaxed_selections(2, torch.tensor([0, 2]), 3, max_staleness=0)
	with pytest.raises(MissingState):
		catalog.relaxed_selections(2, torch.tensor([1]), 3, max_staleness=2)
	_publish(catalog, owner, 2, [1])
	assert catalog.ages(2, query, 3).tolist() == [0, 0, 0, 0]
	# Refreshing a lower layer never refreshes a dependent output implicitly.
	assert catalog.ages(1, torch.tensor([1]), 3).item() == 3


def test_stale_rows_still_require_live_owners_and_complete_analysis():
	directory = EmbeddingDirectory()
	owner = directory.register_owner(1)
	catalog = EmbeddingCatalog(3, 2, directory)
	catalog.enable_staleness()
	_publish(catalog, owner, 1, [0, 1])
	catalog.begin_version(1, {}, pending_layers=[1])
	query = torch.tensor([0, 1])
	assert not catalog.servable(1, query, 1, max_staleness=2).any()
	_publish(catalog, owner, 1, [0])
	catalog.finish_analysis(1, query, 1)
	assert catalog.ages(1, query, 1).tolist() == [0, 1]
	assert catalog.servable(1, query, 1, max_staleness=1).all()
	directory.lose_owner(owner)
	assert not catalog.servable(1, query, 1, max_staleness=1).any()
	for budget in (-1, True, 0.5):
		with pytest.raises(ValueError, match="budget"):
			catalog.servable(1, query, 1, max_staleness=budget)


def test_version_metadata_is_opt_in_and_cannot_be_enabled_after_updates():
	catalog = EmbeddingCatalog(4, 2, EmbeddingDirectory())
	assert catalog._invalid_since is None
	with pytest.raises(ValueError, match="disabled"):
		catalog.servable(1, torch.tensor([0]), 0, max_staleness=1)
	catalog.begin_version(1, {})
	with pytest.raises(RuntimeError, match="before"):
		catalog.enable_staleness()
