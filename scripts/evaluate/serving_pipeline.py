"""Matched serial/lookahead measurements on real owned GPU embedding updates."""

import copy
import json
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from statistics import median
from typing import Annotated, Optional, Sequence

import torch
import typer

from lib.dataset.inference import load_graph as load_graph_data
from lib.graph.data_plane import LocalEmbeddingPlane
from lib.graph.serving import EmbeddingService
from lib.models.operators import full_neighbor_forward
from scripts.validate.recompute import apply_layer, build_model


@torch.inference_mode()
def evaluate(
	raw: Path,
	checkpoint: Optional[Path],
	device: str,
	updates: Sequence[int],
	repeats: int,
	batch_size: int,
	model_names: Sequence[str],
) -> dict:
	graph = load_graph_data(Path(raw))
	features = graph.ndata.pop("feat")
	graph.create_formats_()
	all_nodes = torch.arange(graph.num_nodes())
	generator = torch.Generator().manual_seed(83)
	records = []
	for name in model_names:
		base = build_model(name, features.shape[1], checkpoint)
		model = copy.deepcopy(base).to(device)
		widths = [features.shape[1]]
		for layer in base:
			widths.append(int(layer._out_feats))
		with closing(LocalEmbeddingPlane()) as plane:
			worker = plane.add_node(1, device, 2 * 1024**3)

			def apply(endpoint, task, inputs, model=model):
				task = replace(task, block=task.block.to(endpoint.device))
				output = full_neighbor_forward(model[task.layer - 1], task, inputs)
				if output.ndim == 3:
					output = output.mean(1)
				return output.relu() if task.layer < len(model) else output

			service = EmbeddingService(graph, features, widths, plane, [worker], apply, batch_size=batch_size)
			service.ensure(all_nodes)
			for size in updates:
				if not 0 < size <= graph.num_nodes():
					raise ValueError("Update size must fit the dataset")
				nodes = torch.randperm(graph.num_nodes(), generator=generator)[:size].sort().values
				original = features[nodes].clone()
				query = torch.cat([nodes[:64], torch.randint(graph.num_nodes(), (64,), generator=generator)])
				measurements: dict[bool, list[dict]] = {False: [], True: []}
				try:
					for repeat in range(repeats + 1):
						pair = []
						for mode in (False, True) if repeat % 2 == 0 else (True, False):
							service.pipeline = mode
							# Both modes receive identical rows and values. An unchanged
							# assignment still schedules the same conservative affected set.
							values = original + (repeat + 1) * 0.005
							result = service.change_features(nodes, values)
							pair.append(result)
							full_graph = graph.to(device)
							full = features.to(device)
							for index in range(len(model)):
								full = apply_layer(model, index, full_graph, full)
							with service.query(query, worker.owner) as actual:
								expected = full[query.to(device)]
								torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
								error = float((actual - expected).abs().max())
							if repeat:
								measurements[mode].append({**asdict(result), "max_abs_error": error})
							if pair[0].affected != result.affected or pair[0].tasks != result.tasks:
								raise AssertionError("Serial and pipeline modes executed different work")
				finally:
					features.index_copy_(0, nodes, original)
					# Refresh state after restoring authoritative inputs.
					service.change_features(nodes, original)
				serial = median(item["seconds"] for item in measurements[False])
				lookahead = median(item["seconds"] for item in measurements[True])
				record = {
					"model": name,
					"update_rows": size,
					"serial_median_seconds": serial,
					"lookahead_median_seconds": lookahead,
					"speedup": serial / lookahead,
					"serial": measurements[False],
					"lookahead": measurements[True],
					"resident_bytes": worker.store.bytes_used,
				}
				records.append(record)
				print(
					json.dumps({key: value for key, value in record.items() if key not in ("serial", "lookahead")}),
					flush=True,
				)
	return {
		"dataset_raw": str(raw),
		"nodes": graph.num_nodes(),
		"edges": graph.num_edges(),
		"device": device,
		"batch_size": batch_size,
		"repeats": repeats,
		"checkpoint": str(checkpoint) if checkpoint else None,
		"scope": "Local endpoint, real GPU full-neighbor updates; no RPC, remote transport, or fine-grained deltas",
		"timing": "Mutation through ready publication; compute_seconds is worker wall time, not CUDA kernel time",
		"method": "Identical preparation/chunking; alternate order; discard first pair; reference outside timing",
		"records": records,
	}


ModelsChoice = StrEnum("ModelsChoice", {str(value).replace("-", "_").upper(): value for value in ["sage", "gat"]})  # type: ignore[misc]  # Mypy requires literal Enum members.


@dataclass
class Options:
	data: Path
	checkpoint: Path | None
	device: str
	updates: list[int]
	repeats: int
	batch_size: int
	models: list[str]
	output: Path


app = typer.Typer(help=__doc__, no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def main(
	data: Annotated[Path, typer.Option("--data")],
	output: Annotated[Path, typer.Option("--output")],
	checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
	device: Annotated[str, typer.Option("--device")] = "cuda:0",
	updates: Annotated[list[int] | None, typer.Option("--updates")] = None,
	repeats: Annotated[int, typer.Option("--repeats")] = 5,
	batch_size: Annotated[int, typer.Option("--batch-size")] = 8192,
	models: Annotated[list[ModelsChoice] | None, typer.Option("--models")] = None,
):
	args = Options(
		**{
			**locals(),
			"updates": updates if updates is not None else [100, 1000, 10000],
			"models": [item.value for item in models] if models is not None else ["sage", "gat"],
		}
	)
	if args.repeats < 1 or args.batch_size < 1:
		raise typer.BadParameter("repeats and batch-size must be positive")
	torch.set_num_threads(4)
	result = evaluate(args.data, args.checkpoint, args.device, args.updates, args.repeats, args.batch_size, args.models)
	args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
	app()
