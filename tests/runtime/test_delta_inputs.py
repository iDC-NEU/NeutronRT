from contextlib import ExitStack

import dgl
import pytest
import torch

from lib.graph.anchors import AnchoredRows
from lib.graph.delta_capture import capture_features
from lib.runtime.aggregates.delta_inputs import DeltaInputs
from lib.runtime.aggregates.tables import pack_tables
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.service import DistributedEmbeddingService
from tests.runtime.test_prepared import _actors


@torch.inference_mode()
def test_input_routes_keep_old_changed_h_and_current_unchanged_h_in_the_requested_order():
	graph = dgl.graph(([], []), num_nodes=6)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	features[0, 0] = 1e-20
	original_features = features.clone()
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors())
		plane = RemoteEmbeddingPlane(transport, storage_bytes=65536)
		stack.callback(plane.close)
		service = DistributedEmbeddingService(
			graph, features, (4, 4, 3), plane, retain_aggregates=True, peer_fetch=False
		)
		service.ensure(torch.arange(6), 1)
		model = workers[0].model
		expected_old = model(0, graph, features)
		changed = torch.tensor([0, 3])
		old_features = capture_features(features, changed, max_bytes=48)
		features[changed] += 1.25
		service._begin_version(1, {}, pending_layers=(1, 2))
		with (
			service.catalog.previous_rows(1, changed, 1, retain=plane.retain_states) as old_h,
			service.aggregate_catalog.previous_rows(1, changed, 1, retain=plane.retain_aggregates) as old_sum,
		):
			history, aggregate = AnchoredRows(old_h), AnchoredRows(old_sum)
			service._finish_analysis(1, changed, 1)
			service.ensure(changed, 1)
			inputs = DeltaInputs(service)
			nodes = torch.tensor([5, 0, 3, 1, 0])
			inputs.prepare_old(history, nodes)
			target = workers[1].owner
			old = inputs.old(history, nodes, target)
			new = inputs.current(1, nodes, target)
			pack_tables((old, new))
			for table, expected in ((old, expected_old[nodes]), (new, model(0, graph, features)[nodes])):
				actual = torch.empty_like(expected)
				for group in table.chunks:
					key = next(
						selection.key
						for selection in (*old_h.selections, *service.catalog.selections(1, nodes, 1))
						if plane.directory.locate(selection.key, preferred=target) == group.location
					)
					with ExitStack() as leases:
						actual[group.positions] = plane.read_rows(key, group.rows, leases)
				torch.testing.assert_close(actual, expected)
			assert inputs.old_sum(aggregate, changed, target).rows == 2
			with pytest.raises(ValueError, match="aggregate keys"):
				inputs.old_sum(history, changed, target)
			with pytest.raises(ValueError, match="embedding keys"):
				inputs.old(aggregate, changed, target)
			old_root = inputs.features_old(nodes, old_features).values
			torch.testing.assert_close(old_root, original_features[nodes], rtol=0, atol=0)
			before = features.clone()
			old_root.zero_()
			torch.testing.assert_close(features, before)
			assert inputs.features_old(torch.tensor([1, 2]), old_features).values.data_ptr() == features[1:].data_ptr()
