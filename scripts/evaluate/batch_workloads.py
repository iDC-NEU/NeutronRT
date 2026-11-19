"""Prepare arXiv feature-update recordings with a common event stream."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import torch
import typer

from lib.dataset.inference import load_graph
from lib.profiling.experiment import inference_identity
from lib.profiling.workload import RecordedWorkload
from lib.rpc import inference_pb2 as wire
from scripts.validate.inference_rpc import tensor

BATCH_SIZES = (32, 128, 512)
EVENT_COUNT = 2560
QUERY_INTERVAL = 512


def prepare(recording: RecordedWorkload, features: torch.Tensor, output: Path) -> dict:
	generator = torch.Generator().manual_seed(recording.seed)
	nodes = torch.randperm(len(features), generator=generator)[:EVENT_COUNT]
	if nodes.numel() != EVENT_COUNT or len(recording.steps) < EVENT_COUNT // QUERY_INTERVAL:
		raise ValueError("Input needs enough vertices and recorded queries for the batch sweep")
	values = features[nodes] + 0.1 * torch.randn(
		(EVENT_COUNT, features.shape[1]), dtype=features.dtype, generator=generator
	)
	event_digest = hashlib.sha256(nodes.numpy().tobytes() + values.numpy().tobytes()).hexdigest()
	empty_query = tensor(torch.empty(0, dtype=torch.int64))
	workloads = []
	for batch_size in BATCH_SIZES:
		steps = []
		for start in range(0, EVENT_COUNT, batch_size):
			stop = start + batch_size
			request = wire.Update(
				kind=wire.Update.FEATURES,
				first=tensor(nodes[start:stop]),
				second=tensor(values[start:stop]),
				request_id=f"batch-sweep-{start}-{stop}",
			)
			query = recording.steps[stop // QUERY_INTERVAL - 1][1] if stop % QUERY_INTERVAL == 0 else empty_query
			steps.append((request, query))
		result = replace(recording, steps=tuple(steps))
		path = output / f"arxiv-r{recording.repetition}-batch{batch_size}.pt"
		result.save(path)
		workloads.append({"batch_size": batch_size, "path": str(path), "trace_digest": result.digest})
	return {"repetition": recording.repetition, "event_digest": event_digest, "workloads": workloads}


def main(
	data: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	workload: Annotated[list[Path], typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path, typer.Option()],
):
	graph = load_graph(data)
	features = graph.ndata.pop("feat")
	recordings = [RecordedWorkload.load(path, checkpoint=checkpoint) for path in workload]
	repetitions = {recording.repetition for recording in recordings}
	if len(repetitions) != len(recordings):
		raise ValueError("Workload repetitions must be unique")
	for recording in recordings:
		if inference_identity(graph, features, recording.spec) != recording.input_identity:
			raise ValueError("Recorded input differs from the supplied graph and checkpoint")
	output.mkdir(parents=True, exist_ok=False)
	receipt = {
		"update_kind": "features",
		"event_count": EVENT_COUNT,
		"query_interval_events": QUERY_INTERVAL,
		"repetitions": [prepare(recording, features, output) for recording in recordings],
	}
	(output / "workloads.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
	typer.run(main)
