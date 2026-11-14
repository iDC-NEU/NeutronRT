"""Real communicator failure while a merged update is awaiting publication."""

from dataclasses import replace

import dgl
import torch

from lib.graph.admission import UpdateAdmission, UpdateKind
from lib.models.spec import ModelSpec
from lib.runtime.config import RuntimeFeatures
from lib.runtime.managed import ManagedRuntime
from scripts.validate.inference import reference


def test_merged_publication_resets_failed_workers_and_credits_each_applied_input_once():
	graph = dgl.graph(([0, 1, 1, 2, 3, 4, 5], [1, 2, 3, 3, 4, 5, 0]), num_nodes=6)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	spec = ModelSpec("sage", 4, 5, 3)
	switches = replace(RuntimeFeatures(), bounded_staleness=True, pipeline=True)
	with ManagedRuntime(
		graph,
		features,
		spec,
		["cpu", "cpu"],
		switches=switches,
		max_stale_versions=2,
		idle_flush_seconds=60,
		batch_size=2,
		timeout=5,
	) as runtime:
		service = runtime.service
		service.ensure(torch.arange(6))
		with service.query(torch.arange(6), runtime.plane.storage.owner) as initial:
			initial = initial.clone()
		admission = UpdateAdmission(service)
		try:
			tickets = [
				admission.submit(UpdateKind.FEATURES, torch.tensor([1]), features[[1]] + offset) for offset in (1, 2)
			]
			admission._pool.submit(lambda: None).result(10)
			with service.query(torch.arange(6), runtime.plane.storage.owner, max_staleness=2) as stale:
				torch.testing.assert_close(stale, initial)
			failed = runtime.processes[1]
			failed.terminate()
			failed.join(5)
			completion = admission.sync(UpdateKind.FEATURES, torch.tensor([1]), features[[1]] + 3)
			assert completion.version == 3 and runtime.reset_count == 1
			assert [ticket.result().input_version for ticket in tickets] == [1, 2]
			assert service.counters.updates == service.counters.input_changes == 3
			with service.query(torch.arange(6), runtime.plane.storage.owner) as actual:
				torch.testing.assert_close(
					actual, reference(spec.build(), graph, features, torch.arange(6)), rtol=2e-5, atol=2e-5
				)
		finally:
			admission.close()
