"""Profile matched arXiv RPC workloads on separately launched storage/GNN processes.

This command opens sockets and may use CUDA. Review --dry-run in the sandbox;
execute it as part of the consolidated host validation batch.
"""

import json
import os
import signal
import sys
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, suppress
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Annotated

import grpc
import pandas as pd
import torch
import torch.multiprocessing as mp
import typer

from lib.dataset.inference import load_graph as load_graph_data
from lib.graph.methods import method_from_switches
from lib.models import MODEL_NAMES
from lib.models.spec import ModelSpec
from lib.profiling.accuracy import query_error
from lib.profiling.controller import RuntimeProfiler
from lib.profiling.experiment import (
	ProfileCase,
	latency_summary,
	make_trace,
	paired_comparison,
	read_cases,
	trace_digest,
)
from lib.profiling.experiment import (
	inference_identity as _identity,
)
from lib.profiling.report import markdown_report
from lib.rpc import inference_pb2 as wire
from lib.rpc.inference_pb2_grpc import OwnedInferenceStub, add_OwnedInferenceServicer_to_server
from lib.rpc.limits import message_options
from lib.rpc.service import OwnedServicer
from lib.rpc.tensors import TensorPayload, decode_tensor
from lib.runtime.managed import ManagedRuntime
from scripts.validate.inference import reference
from scripts.validate.inference_rpc import tensor


def _server(connection, case, options, spec, folder):
	# All processes launched for this trial share a private OS process group.
	# Cleanup never targets another experiment or a pre-existing GPU process.
	os.setsid()
	folder = Path(folder)
	with (folder / "server.log").open("w") as log:
		os.dup2(log.fileno(), 1)
		os.dup2(log.fileno(), 2)
		try:
			torch.set_num_threads(options["threads"])
			graph = load_graph_data(Path(options["data"]), add_inverse_edges=options["add_inverse_edges"])
			features = graph.ndata.pop("feat")
			graph.create_formats_()
			identity = _identity(graph, features, spec)
			with ExitStack() as stack:
				runtime = stack.enter_context(
					ManagedRuntime(
						graph,
						features,
						spec,
						case.devices,
						worker_bytes=options["worker_bytes"],
						storage_bytes=options["storage_bytes"],
						timeout=options["timeout"],
						batch_size=case.batch_size,
						switches=case.measured_switches,
						profile_dir=str(folder),
						native_threads=case.native_threads,
						storage_partitions=case.storage_partitions,
						storage_directory=options.get("topology_root"),
						max_stale_versions=case.max_stale_versions,
						idle_flush_seconds=case.idle_flush_seconds,
						delta_max_steps=case.delta_max_steps,
						delta_anchor_bytes=case.delta_anchor_bytes,
					)
				)
				graph = runtime.service.graph
				# Equal coverage for every setting; dependency-closure cold starts are
				# a separate experiment and cannot bias this steady-state comparison.
				nodes = torch.arange(graph.num_nodes())
				for layer in range(1, spec.layers + 1):
					runtime.service.ensure(nodes, layer)
				profiler = stack.enter_context(RuntimeProfiler(runtime))
				assert runtime.plane is not None
				frontend = OwnedServicer(
					runtime.service,
					runtime.plane.storage.owner,
					max_pending=max(32, case.max_inflight),
					max_tickets=max(1024, case.max_inflight * 2),
				)
				stack.callback(frontend.close)
				executor = stack.enter_context(ThreadPoolExecutor(max_workers=8))
				server = grpc.server(executor, options=message_options())
				stack.callback(lambda: server.stop(5).wait())
				add_OwnedInferenceServicer_to_server(frontend, server)
				port = server.add_insecure_port("127.0.0.1:0")
				if not port:
					raise RuntimeError("Could not bind profiling RPC server")
				server.start()
				connection.send(
					{
						"address": f"127.0.0.1:{port}",
						"identity": identity,
						"switches": runtime.switches.as_dict(),
						"initialized_rows_per_layer": graph.num_nodes(),
					}
				)
				while True:
					command = connection.recv()
					if command["op"] == "close":
						break
					if command["op"] == "start":
						profiler.start(trace=command["trace"], trace_task_limit=options["trace_tasks"])
						connection.send({"started": True})
					elif command["op"] == "stop":
						connection.send(profiler.stop())
					else:
						raise ValueError("Unknown profile control operation")
		except BaseException:
			failure = traceback.format_exc()
			print(failure, file=sys.stderr, flush=True)
			with suppress(BrokenPipeError, EOFError, OSError):
				connection.send({"error": failure})
			raise
		finally:
			connection.close()


def _receive(connection, seconds):
	if not connection.poll(seconds):
		raise TimeoutError("Timed out waiting for profiling coordinator; inspect its server.log")
	result = connection.recv()
	if "error" in result:
		raise RuntimeError(result["error"])
	return result


def exercise(stub, trace, case, session, query, *, prefix, timeout, capture_queries=False):
	"""Client wall time includes serialization/admission/queueing and RPC waits.

	Async completion latency is observed by ordered waits at bounded concurrency;
	we do not pretend it is the exact server publication timestamp.
	"""
	pending = deque()
	admissions, completions, queries, query_versions = [], [], [], []
	completed = changes = 0
	captured = []
	captured_bytes = 0
	start = last_completion = perf_counter()

	def finish(ticket, began, completion=None):
		nonlocal completed, changes, last_completion
		if completion is None:
			completion = stub.Wait(wire.WaitRequest(ticket=ticket), timeout=timeout)
		last_completion = perf_counter()
		if not completion.ready:
			raise RuntimeError("Blocking completion returned an unpublished update")
		completions.append(last_completion - began)
		completed += 1
		changes += completion.input_changes
		stub.Release(completion.ticket, timeout=timeout)

	for index, update in enumerate(trace):
		if len(pending) >= case.max_inflight:
			finish(*pending.popleft())
		began = perf_counter()
		request = wire.Update(
			kind={"features": wire.Update.FEATURES, "insert": wire.Update.INSERT, "delete": wire.Update.DELETE}[
				update.kind
			],
			first=tensor(update.first),
			second=tensor(update.second),
			request_id=f"{prefix}-{index}",
			session=session,
		)
		if case.interface == "sync":
			completion = stub.Sync(request, timeout=timeout)
			finish(completion.ticket, began, completion)
		else:
			ticket = stub.Async(request, timeout=timeout)
			admissions.append(perf_counter() - began)
			pending.append((ticket, began))
		if case.query_every and (index + 1) % case.query_every == 0:
			began = perf_counter()
			reply = stub.Query(
				wire.QueryRequest(nodes=tensor(query), max_staleness=case.query_staleness), timeout=timeout
			)
			queries.append(perf_counter() - began)
			query_versions.append(reply.version)
			if capture_queries:
				captured_bytes += reply.ByteSize()
				if captured_bytes > 64 * 1024**2:
					raise MemoryError("Captured query samples exceed 64 MiB; reduce query frequency or size")
				captured.append(reply)
	while pending:
		finish(*pending.popleft())
	client_window = perf_counter() - start
	return {
		"client_seconds": client_window,
		"last_observed_publication_seconds": last_completion - start,
		"completed_updates": completed,
		"completed_input_changes": changes,
		"client_changes_per_second": changes / client_window if completed else None,
		"admission_latency": latency_summary(admissions),
		"completion_latency": latency_summary(completions),
		"query_latency": latency_summary(queries),
		"raw_admission_seconds": admissions,
		"raw_completion_seconds": completions,
		"raw_query_seconds": queries,
		"query_versions": query_versions,
		"_query_responses": captured,
	}


def _apply_reference(graph, features, trace):
	for update in trace:
		if update.kind == "features":
			features.index_copy_(0, update.first, update.second)
		elif update.kind == "insert":
			graph.add_edges(update.first, update.second)
		else:
			_, _, edges = graph.edge_ids(update.first, update.second, return_uv=True)
			graph.remove_edges(edges.unique())


def reference_oracles(graph, features, trace, query, model, *, every_version):
	"""Replay once, retaining only selected rows per version, never whole matrices."""
	oracles = {0: reference(model, graph, features, query)} if every_version else {}
	for version, update in enumerate(trace, start=1):
		_apply_reference(graph, features, [update])
		if every_version or version == len(trace):
			oracles[version] = reference(model, graph, features, query)
	return oracles


def check_query_samples(responses, oracles, case, labels, test_mask):
	samples, metrics = [], []
	for reply in responses:
		if reply.max_age > case.query_staleness or reply.min_fresh_version + reply.max_age != reply.version:
			raise RuntimeError("Query freshness certificate exceeds its requested budget")
		values = decode_tensor(TensorPayload(reply.values.dtype, tuple(reply.values.shape), reply.values.data))
		fresh = oracles[reply.version]
		metrics.append(
			{
				"version": reply.version,
				"max_age": reply.max_age,
				"min_fresh_version": reply.min_fresh_version,
				**query_error(values, fresh, labels=labels, test_mask=test_mask),
			}
		)
		# Independently check every row against full inference within its allowed
		# version interval; a shared bound must not silently compound by layer.
		within_budget = torch.zeros(values.shape[0], dtype=torch.bool)
		for version in range(reply.min_fresh_version, reply.version + 1):
			within_budget |= torch.isclose(values, oracles[version], rtol=2e-4, atol=3e-5).all(1)
		if not within_budget.all():
			raise RuntimeError("Query contains rows outside its shared version certificate")
		samples.append({"version": reply.version, "max_age": reply.max_age, "values": values})
	return metrics, samples


def _query_nodes(graph, trace, count, seed, selection, test_nodes):
	generator = torch.Generator().manual_seed(seed)
	if selection == "test":
		if test_nodes is None:
			raise ValueError("Test query sampling requires an official test split")
		candidates = test_nodes
	elif selection == "affected":
		candidates = torch.cat(
			[update.first for update in trace] + [update.second for update in trace if update.kind != "features"]
		).unique()
		for _ in range(2):
			candidates = torch.cat([candidates, graph.out_edges(candidates)[1]]).unique()
	else:
		candidates = torch.arange(graph.num_nodes())
	if not candidates.numel():
		raise ValueError("Query sampling pool is empty")
	return (
		candidates[torch.randperm(candidates.numel(), generator=generator)[:count]]
		if count <= candidates.numel()
		else candidates[torch.randint(candidates.numel(), (count,), generator=generator)]
	)


def run_case(
	case: ProfileCase,
	options,
	spec,
	trace,
	query,
	oracles,
	labels,
	test_mask,
	identity,
	repetition,
	*,
	trace_pass=False,
):
	folder = Path(options["output"]) / (f"{repetition:02d}-{case.name}" + ("-trace" if trace_pass else ""))
	folder.mkdir(parents=True, exist_ok=False)
	context = mp.get_context("spawn")
	connection, child = context.Pipe()
	process = context.Process(target=_server, args=(child, case, options, spec, str(folder)))
	process.start()
	assert process.pid is not None
	process_group = process.pid
	child.close()
	try:
		ready = _receive(connection, options["startup_timeout"])
		if ready["identity"] != identity:
			raise RuntimeError("Client and storage loaded different graph/features/weights")
		with grpc.insecure_channel(ready["address"], options=message_options()) as channel:
			grpc.channel_ready_future(channel).result(timeout=30)
			stub = OwnedInferenceStub(channel)
			session = stub.Info(wire.Empty(), timeout=30).session
			warmup = options["warmup"]
			exercise(stub, trace[:warmup], case, session, query, prefix="warmup", timeout=options["timeout"])
			connection.send({"op": "start", "trace": trace_pass})
			_receive(connection, options["timeout"])
			client = exercise(
				stub,
				trace[warmup:],
				case,
				session,
				query,
				prefix="measure",
				timeout=options["timeout"],
				capture_queries=True,
			)
			connection.send({"op": "stop"})
			profile = _receive(connection, options["timeout"])
			(folder / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
			metrics, samples = check_query_samples(client.pop("_query_responses"), oracles, case, labels, test_mask)
			torch.save({"nodes": query, "samples": samples}, folder / "query-samples.pt")
			# Validation and its selected reads are outside the measured window.
			reply = stub.Query(wire.QueryRequest(nodes=tensor(query)), timeout=options["timeout"])
			actual = decode_tensor(TensorPayload(reply.values.dtype, tuple(reply.values.shape), reply.values.data))
			expected = oracles[len(trace)]
			torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)
			info = stub.Info(wire.Empty(), timeout=options["timeout"])
			if (
				reply.version != len(trace)
				or info.completed_updates != len(trace)
				or info.HasField("incomplete_version")
			):
				raise RuntimeError("Profiler observed missing, duplicated, or incomplete updates")
			if (
				profile["engine"]["updates"] != client["completed_updates"]
				or profile["engine"]["input_changes"] != client["completed_input_changes"]
			):
				raise RuntimeError("Client and internal completed-work counters differ")
			result = {
				"case": case.name,
				"method": profile["method"],
				"repetition": repetition,
				"identity": identity,
				"trace_digest": trace_digest(trace),
				"valid": profile["valid"],
				"trace_pass": trace_pass,
				"settings": asdict(case),
				**client,
				"enabled_switches": ready["switches"],
				"profile": profile,
				"engine_seconds": profile["engine"]["completion_window_seconds"],
				"query_accuracy": metrics,
				"correctness": {
					"max_abs_error": float((actual - expected).abs().max()),
					"prediction_disagreement": float((actual.argmax(1) != expected.argmax(1)).float().mean()),
				},
			}
			(folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
		connection.send({"op": "close"})
		process.join(30)
		if process.is_alive() or process.exitcode != 0:
			raise RuntimeError(f"Profiling server did not shut down normally; inspect {folder / 'server.log'}")
		return result
	finally:
		connection.close()
		if process.is_alive() or process.exitcode != 0:
			try:
				os.killpg(process_group, signal.SIGTERM)
			except ProcessLookupError:
				if process.is_alive():
					process.terminate()
			process.join(10)
			with suppress(ProcessLookupError):
				os.killpg(process_group, signal.SIGKILL)
			process.join(10)


ModelChoice = StrEnum("ModelChoice", {str(value).replace("-", "_").upper(): value for value in MODEL_NAMES})  # type: ignore[misc]  # Mypy requires literal Enum members.
QuerySelectionChoice = StrEnum(  # type: ignore[misc]  # Mypy requires literal Enum members.
	"QuerySelectionChoice", {str(value).replace("-", "_").upper(): value for value in ["affected", "test", "uniform"]}
)


_DEFAULT_QUERY_SELECTION = QuerySelectionChoice("affected")


@dataclass
class Options:
	data: Path
	add_inverse_edges: bool
	cases: Path
	output: Path
	checkpoint: Path | None
	model: str | None
	classes: int | None
	hidden: int | None
	layers: int | None
	heads: int | None
	relations: int | None
	pseudo_dim: int | None
	kernels: int | None
	updates: int
	warmup: int
	changes: int
	pool_size: int
	repetitions: int
	queries: int
	query_selection: str
	labels: Path | None
	test_split: Path | None
	threads: int
	seed: int
	mixed: bool
	trace: bool
	trace_tasks: int
	worker_bytes: int
	topology_root: Path | None
	storage_bytes: int
	timeout: float
	startup_timeout: float
	dry_run: bool


app = typer.Typer(help=__doc__, no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def main(
	data: Annotated[Path, typer.Option("--data")],
	cases: Annotated[Path, typer.Option("--cases")],
	output: Annotated[Path, typer.Option("--output")],
	add_inverse_edges: Annotated[
		bool, typer.Option("--add-inverse-edges", help="Include reverse edges, including self loops")
	] = False,
	checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
	model: Annotated[ModelChoice | None, typer.Option("--model")] = None,
	classes: Annotated[int | None, typer.Option("--classes")] = None,
	hidden: Annotated[int | None, typer.Option("--hidden")] = None,
	layers: Annotated[int | None, typer.Option("--layers")] = None,
	heads: Annotated[int | None, typer.Option("--heads")] = None,
	relations: Annotated[int | None, typer.Option("--relations")] = None,
	pseudo_dim: Annotated[int | None, typer.Option("--pseudo-dim")] = None,
	kernels: Annotated[int | None, typer.Option("--kernels")] = None,
	updates: Annotated[int, typer.Option("--updates")] = 12,
	warmup: Annotated[int, typer.Option("--warmup")] = 3,
	changes: Annotated[int, typer.Option("--changes")] = 16,
	pool_size: Annotated[int, typer.Option("--pool-size")] = 128,
	repetitions: Annotated[int, typer.Option("--repetitions")] = 5,
	queries: Annotated[int, typer.Option("--queries")] = 128,
	query_selection: Annotated[QuerySelectionChoice, typer.Option("--query-selection")] = _DEFAULT_QUERY_SELECTION,
	labels: Annotated[
		Path | None, typer.Option("--labels", help="Default: node-label.csv.gz beside raw graph files")
	] = None,
	test_split: Annotated[
		Path | None, typer.Option("--test-split", help="Default: ../split/time/test.csv.gz relative to --data")
	] = None,
	threads: Annotated[int, typer.Option("--threads")] = 4,
	seed: Annotated[int, typer.Option("--seed")] = 71,
	mixed: Annotated[bool, typer.Option("--mixed/--no-mixed")] = True,
	trace: Annotated[bool, typer.Option("--trace/--no-trace")] = False,
	trace_tasks: Annotated[int, typer.Option("--trace-tasks")] = 100,
	worker_bytes: Annotated[int, typer.Option("--worker-bytes")] = 1024**3,
	topology_root: Annotated[Path | None, typer.Option("--topology-root")] = None,
	storage_bytes: Annotated[int, typer.Option("--storage-bytes")] = 2 * 1024**3,
	timeout: Annotated[float, typer.Option("--timeout")] = 120,
	startup_timeout: Annotated[float, typer.Option("--startup-timeout")] = 600,
	dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
):
	args = Options(
		**{**locals(), "model": model.value if model is not None else None, "query_selection": query_selection.value}
	)
	_run(args)


def _run(args: Options) -> None:
	if (
		min(args.updates, args.changes, args.repetitions, args.queries, args.threads, args.worker_bytes) < 1
		or min(args.warmup, args.storage_bytes) < 0
	):
		raise typer.BadParameter("Invalid workload, memory, or thread limits")
	if (
		args.pool_size < args.changes
		or not 1 <= args.trace_tasks <= 1000
		or min(args.timeout, args.startup_timeout) <= 0
	):
		raise typer.BadParameter("Invalid hotspot pool, trace limit, or timeout")
	cases = read_cases(json.loads(args.cases.read_text()))
	if args.dry_run:
		print(
			json.dumps(
				{
					"host_capabilities_required": ["sockets", "process groups", "CUDA for CUDA cases"],
					"settings": [asdict(case) for case in cases],
					"timed_runs": len(cases) * args.repetitions,
					"trace_runs": len(cases) if args.trace else 0,
				},
				indent=2,
			)
		)
		return
	if args.output.exists() and any(args.output.iterdir()):
		raise typer.BadParameter("Output directory must be new or empty; preserve previous raw trials")
	torch.set_num_threads(args.threads)
	graph = load_graph_data(Path(args.data), add_inverse_edges=args.add_inverse_edges)
	features = graph.ndata.pop("feat")
	try:
		spec = ModelSpec.from_options(
			features.shape[1],
			checkpoint=str(args.checkpoint) if args.checkpoint else None,
			name=args.model,
			classes=args.classes,
			hidden=args.hidden,
			layers=args.layers,
			heads=args.heads,
			relations=args.relations,
			pseudo_dim=args.pseudo_dim,
			kernels=args.kernels,
		)
	except ValueError as error:
		raise typer.BadParameter(str(error)) from error
	identity = _identity(graph, features, spec)
	trace = make_trace(
		features,
		updates=args.warmup + args.updates,
		changes=args.changes,
		seed=args.seed,
		pool_size=args.pool_size,
		mixed=args.mixed,
	)
	label_path = args.labels or args.data / "node-label.csv.gz"
	split_path = args.test_split or args.data.parent / "split" / "time" / "test.csv.gz"
	test_nodes = (
		torch.from_numpy(pd.read_csv(split_path, header=None).to_numpy().reshape(-1)).long()
		if split_path.exists()
		else None
	)
	query = _query_nodes(graph, trace, args.queries, args.seed + 1, args.query_selection, test_nodes)
	labels = (
		torch.from_numpy(pd.read_csv(label_path, header=None).to_numpy().reshape(-1)).long()[query]
		if label_path.exists()
		else None
	)
	test_mask = None
	if test_nodes is not None:
		test_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
		test_mask[test_nodes] = True
		test_mask = test_mask[query]
	if any(case.query_every for case in cases) and (len(trace) + 1) * query.numel() * spec.classes * 4 > 64 * 1024**2:
		raise MemoryError("Selected-row reference history exceeds 64 MiB; reduce queries or updates")
	oracles = reference_oracles(
		graph, features, trace, query, spec.build(), every_version=any(case.query_every for case in cases)
	)
	del graph, features
	args.output.mkdir(parents=True, exist_ok=True)
	options = {**vars(args), "data": str(args.data.resolve()), "output": str(args.output.resolve())}

	def attempt(case, repetition, *, trace_pass=False):
		try:
			return run_case(
				case,
				options,
				spec,
				trace,
				query,
				oracles,
				labels,
				test_mask,
				identity,
				repetition,
				trace_pass=trace_pass,
			)
		except Exception:
			failed = {
				"case": case.name,
				"repetition": repetition,
				"valid": False,
				"trace_pass": trace_pass,
				"identity": identity,
				"trace_digest": trace_digest(trace),
				"client_seconds": None,
				"error": traceback.format_exc(),
			}
			suffix = "-trace" if trace_pass else ""
			(args.output / f"{repetition:02d}-{case.name}{suffix}-failure.json").write_text(
				json.dumps(failed, indent=2) + "\n"
			)
			return failed

	runs = []
	for repetition in range(args.repetitions):
		# Rotate and reverse to reduce systematic warm-machine/order bias.
		order = list(cases[repetition % len(cases) :] + cases[: repetition % len(cases)])
		if repetition % 2:
			order.reverse()
		for case in order:
			run = attempt(case, repetition)
			runs.append(run)
			print(
				json.dumps(
					{
						"case": case.name,
						"repetition": repetition,
						"valid": run["valid"],
						"client_seconds": run["client_seconds"],
					}
				),
				flush=True,
			)
	comparisons = []
	for case in cases:
		if case.reference:
			baseline = [run for run in runs if run["case"] == case.reference]
			candidate = [run for run in runs if run["case"] == case.name]
			try:
				comparison = {
					**paired_comparison(baseline, candidate),
					"engine_comparison": paired_comparison(baseline, candidate, metric="engine_seconds"),
					"baseline_method": method_from_switches(baseline[0]["enabled_switches"]),
					"candidate_method": method_from_switches(candidate[0]["enabled_switches"]),
				}
			except ValueError as error:
				comparison = {"verdict": "invalid", "reason": str(error)}
			comparisons.append({"reference": case.reference, "case": case.name, **comparison})
	traces = [attempt(case, 0, trace_pass=True) for case in cases] if args.trace else []
	full_test = (
		args.query_selection == "test"
		and test_nodes is not None
		and labels is not None
		and query.unique().numel() == test_nodes.unique().numel()
	)
	report = {
		"schema_version": 1,
		"dataset": "ogbn-arxiv",
		"model": asdict(spec),
		"identity": identity,
		"trace_digest": trace_digest(trace),
		"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
		"scope": "Local-host RPC client, separate CPU storage process, and process-owned GNN workers",
		"algorithm": "per-case neighborhood recomputation or sparse projected-sum updates",
		"query_selection": args.query_selection,
		"query_rows": query.numel(),
		"unique_query_rows": query.unique().numel(),
		"query_accuracy_scope": "Full official test split at observed graph versions"
		if full_test
		else "Selected rows at observed graph versions; sample accuracy, not full-test accuracy",
		"full_test_coverage": full_test,
		"weights": "trained checkpoint" if spec.checkpoint else "fixed random",
		"metric_notes": [
			"Client RTT includes serialization and ordered waits; internal completed-input throughput is separate.",
			"Model CUDA event time and CPU phases overlap; do not sum them.",
			"P2P bytes count tensor payloads at senders, not NIC traffic. Collective bytes are excluded.",
			"Detailed traces are separate passes. Shared-GPU contention must be recorded by host validation.",
			"Comparison intervals describe these paired trials, not independent cluster repetitions.",
		],
		"runs": runs,
		"comparisons": comparisons,
		"trace_runs": traces,
	}
	(args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
	(args.output / "report.md").write_text(markdown_report(report))
	if any(not run["valid"] for run in [*runs, *traces]):
		raise SystemExit("One or more windows were invalid; inspect raw reports")


if __name__ == "__main__":
	app()
