"""The streaming accuracy gate checks completed versions without logit history."""

import pytest
import torch

from lib.profiling.experiment import make_trace
from scripts.validate.delta_drift import observe_trace
from tests.runtime.test_delta_service import _service


@torch.no_grad()
@pytest.mark.parametrize("aggregate_only", [False, True])
def test_streaming_trace_checks_baseline_intervals_and_final_partial_interval(aggregate_only):
	with _service(delta_max_steps=2, aggregate_only=aggregate_only) as (service, model):
		service.ensure(torch.arange(6))
		reference = service.graph.clone()
		trace = make_trace(service.features, updates=13, changes=2, pool_size=6, mixed=True)
		emitted = []
		records = observe_trace(
			service, model, reference, trace, torch.arange(6), torch.arange(6) % 3, every=4, emit=emitted.append
		)
		assert records == emitted and [record["version"] for record in records] == [0, 4, 8, 12, 13]
		assert all(record["valid"] and record["labelled_test_rows"] == 6 for record in records)
		assert all(record["prediction_disagreement"] == 0 for record in records)
		assert records[-1]["counters"]["updates"] == 13
		assert records[-1]["counters"]["delta_refresh_rows"] > 0
		assert not service.remote_plane._retained_states
		assert not any(isinstance(value, torch.Tensor) for record in records for value in record.values())


@pytest.mark.parametrize("interval", [0, -1, True])
def test_invalid_observation_interval_fails_before_serving(interval):
	with pytest.raises(ValueError, match="interval"):
		observe_trace(None, None, None, None, None, None, every=interval, emit=None)
