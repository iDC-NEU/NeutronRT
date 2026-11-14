import pytest
import torch

from lib.graph.catalog import EmbeddingCatalog
from tests.graph.test_catalog import _publish
from tests.graph.test_serving import _setup


def test_sum_and_embedding_state_share_invalidation_but_keep_independent_chunks():
	service, plane, _, _ = _setup("sage")
	try:
		aggregate = EmbeddingCatalog(6, 2, plane.directory, kind="a")
		service._additional_catalogs = (aggregate,)
		owner = service.workers[0].owner
		_publish(service.catalog, owner, 1, [0, 1, 2])
		_publish(aggregate, owner, 1, [2, 1])
		nodes = torch.tensor([0, 1, 2])
		torch.testing.assert_close(service._state_available(1, nodes, 0), torch.tensor([False, True, True]))
		service._begin_version(1, {}, pending_layers=(1, 2))
		assert service._pending_analysis() == (1, 2)
		assert not service._state_available(1, nodes, 1).any()
		# A resumed analysis must also finish the remaining aggregate catalog.
		service.catalog.finish_analysis(1, torch.tensor([1]), 1)
		service._finish_analysis(1, torch.tensor([1]), 1)
		torch.testing.assert_close(service._state_available(1, nodes, 1), torch.tensor([False, False, True]))
		service._finish_analysis(2, torch.tensor([1]), 1)
		assert service._pending_analysis() == ()
		with pytest.raises(ValueError, match="not pending"):
			service._finish_analysis(2, nodes, 1)
		service._begin_version(2, {1: torch.tensor([2])})
		assert service.catalog.version == aggregate.version == 2
		assert not service._state_available(1, nodes, 2).any()
	finally:
		plane.close()


def test_unsynchronized_catalog_rejects_version_advance_before_changing_embedding_catalog():
	service, plane, _, _ = _setup("sage")
	try:
		aggregate = EmbeddingCatalog(6, 2, plane.directory, kind="a")
		service._additional_catalogs = (aggregate,)
		aggregate.begin_version(1, {})
		with pytest.raises(RuntimeError, match="versions disagree"):
			service._begin_version(1, {})
		assert service.catalog.version == 0 and aggregate.version == 1
	finally:
		plane.close()
