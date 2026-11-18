"""Opt-in worker phases, CUDA-event operator time, and bounded trace windows."""

import json
import os
import socket
from pathlib import Path
from typing import Callable, List, Optional, Tuple, TypeVar

import torch
from torch import Tensor

from lib.graph.recompute import LayerPlan
from lib.profiling.communication import ProfiledCommManager
from lib.profiling.instrumentation import profile_snapshot, profile_window
from lib.profiling.recorder import MemorySampler
from lib.runtime.aggregates.delta import DeltaHeader
from lib.runtime.aggregates.recipe import RecipeHeader
from lib.runtime.aggregates.reconstruction import ReconstructionHeader
from lib.runtime.protocol import COMPUTE_WITH_AGGREGATE, PROFILE, ComputeHeader
from lib.runtime.worker import OwnedWorker

ModelOutput = TypeVar("ModelOutput")


class ProfiledWorker(OwnedWorker):
	def __init__(self, comm: ProfiledCommManager, *args, profile_dir: Optional[str] = None, **kwargs):
		super().__init__(comm, *args, **kwargs)
		self.metrics_comm = comm
		self.recorder = comm.recorder
		self.profile_dir = Path(profile_dir) if profile_dir is not None else None
		self._rss: Optional[MemorySampler] = None
		self._resident_peak = 0
		self._model_events: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
		self._torch_profile: Optional[torch.profiler.profile] = None
		self._torch_path: Optional[Path] = None
		self._trace_limit = self._trace_models = 0
		self._trace_capped = False
		self._profile_window = 0
		self._incremental_before: dict[str, int] = {}
		self._profile_before: Optional[dict] = None

	def _apply(self, task: LayerPlan, inputs: Tensor) -> Tensor:
		return self._profile_model(task, inputs, super()._apply)

	def _apply_with_aggregate(self, task: LayerPlan, inputs: Tensor) -> Tuple[Tensor, Tensor]:
		result = self._profile_model(task, inputs, super()._apply_with_aggregate)
		if self.recorder.active:
			self.recorder.count(f"model.layer{task.layer}.sum_bytes", result[1].numel() * result[1].element_size())
		return result

	def _apply_delta(self, layer: int, tables: Tuple[Tensor, ...], tail: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor]:
		if not self.recorder.active:
			return super()._apply_delta(layer, tables, tail)
		self.recorder.count(f"model.layer{layer}.executions")
		self.recorder.count(f"model.layer{layer}.targets", tables[1].shape[0])
		self.recorder.count(f"delta.layer{layer}.executions")
		self.recorder.count(f"delta.layer{layer}.input_rows", tables[2].shape[0])
		self.recorder.count(f"delta.layer{layer}.structural_rows", tables[4].shape[0])
		self.recorder.count(f"delta.layer{layer}.routed_edges", tail[2].numel() + tail[4].numel())
		start = end = None
		with self.recorder.span(f"worker.delta_cpu.layer{layer}"):
			if self.comm.device.type == "cuda":
				start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
				start.record()
			values = super()._apply_delta(layer, tables, tail)
			if start is not None and end is not None:
				end.record()
				self._model_events.append((f"worker.delta_cuda.layer{layer}", start, end))
		self._trace_models += 1
		return values

	def _profile_model(
		self, task: LayerPlan, inputs: Tensor, apply: Callable[[LayerPlan, Tensor], ModelOutput]
	) -> ModelOutput:
		if not self.recorder.active:
			return apply(task, inputs)
		self.recorder.count(f"model.layer{task.layer}.executions")
		self.recorder.count(f"model.layer{task.layer}.targets", task.targets.numel())
		self.recorder.count(f"model.layer{task.layer}.sources", task.sources.numel())
		self.recorder.count(f"model.layer{task.layer}.compiled_edges", task.block.num_edges())
		start = end = None
		with self.recorder.span(f"worker.model_cpu.layer{task.layer}"):
			if self.comm.device.type == "cuda":
				start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
				start.record()
			values = apply(task, inputs)
			if start is not None and end is not None:
				end.record()
				self._model_events.append((f"worker.model_cuda.layer{task.layer}", start, end))
		self._trace_models += 1
		return values

	def _flush_cuda(self, *, wait: bool) -> None:
		remaining = []
		for phase, start, end in self._model_events:
			if wait:
				end.synchronize()
			elif not end.query():
				remaining.append((phase, start, end))
				continue
			self.recorder.observe(phase, start.elapsed_time(end) / 1000)
		self._model_events = remaining

	def _inputs(self, *args, **kwargs):
		with self.recorder.span("worker.input_resolution"):
			return super()._inputs(*args, **kwargs)

	def _reconstruct_sage(self, start_layer: int, tables: Tuple[Tensor, ...], degrees: Tensor) -> Tensor:
		if not self.recorder.active:
			return super()._reconstruct_sage(start_layer, tables, degrees)
		layer = start_layer + len(tables) - 1
		self.recorder.count(f"reconstruction.layer{layer}.executions")
		self.recorder.count(f"reconstruction.layer{layer}.rows", degrees.numel())
		self.recorder.count("reconstruction.transforms", len(tables) - 1)
		start = end = None
		with self.recorder.span(f"worker.reconstruction_cpu.layer{layer}"):
			if self.comm.device.type == "cuda":
				start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
				start.record()
			values = super()._reconstruct_sage(start_layer, tables, degrees)
			if start is not None and end is not None:
				end.record()
				self._model_events.append((f"worker.reconstruction_cuda.layer{layer}", start, end))
		self._trace_models += 1
		return values

	def reconstruct(self, command: Tensor) -> None:
		if self.recorder.active:
			header = ReconstructionHeader.decode(command)
			self.recorder.count(f"reconstruction.layer{header.layer}.attempts")
		with self.recorder.span("worker.reconstruct_task"):
			super().reconstruct(command)
		self._after_compute()

	def compute(self, header_tensor: Tensor) -> None:
		if self.recorder.active:
			header = ComputeHeader.decode(header_tensor)
			self.recorder.count(f"tasks.layer{header.layer}.attempts")
		with self.recorder.span("worker.compute_task"):
			super().compute(header_tensor)
		self._after_compute()

	def compute_with_aggregate(self, header_tensor: Tensor) -> None:
		if self.recorder.active:
			header = ComputeHeader.decode(header_tensor, opcode=COMPUTE_WITH_AGGREGATE)
			self.recorder.count(f"tasks.layer{header.layer}.attempts")
		with self.recorder.span("worker.compute_task"):
			super().compute_with_aggregate(header_tensor)
		self._after_compute()

	def compute_from_aggregates(self, command: Tensor) -> None:
		if self.recorder.active:
			header = RecipeHeader.decode(command)
			self.recorder.count(f"tasks.layer{header.layer}.attempts")
			self.recorder.count("tasks.aggregate_input_recipes")
		with self.recorder.span("worker.compute_recipe_task"):
			super().compute_from_aggregates(command)
		self._after_compute()

	def compute_delta(self, command: Tensor) -> None:
		if self.recorder.active:
			header = DeltaHeader.decode(command)
			self.recorder.count(f"tasks.layer{header.layer}.attempts")
		with self.recorder.span("worker.delta_task"):
			super().compute_delta(command)
		self._after_compute()

	def prepare(self, header_tensor: Tensor) -> None:
		with self.recorder.span("worker.prepare_inputs"):
			super().prepare(header_tensor)

	def run_prepared(self, command: Tensor) -> None:
		if self.recorder.active and self._prepared is not None:
			self.recorder.count(f"tasks.layer{self._prepared.header.layer}.attempts")
		with self.recorder.span("worker.compute_task"):
			super().run_prepared(command)
		self._after_compute()

	def _after_compute(self) -> None:
		self._resident_peak = max(self._resident_peak, self.store.bytes_used)
		self._flush_cuda(wait=False)
		if self._torch_profile is not None and self._trace_models >= self._trace_limit:
			self._trace_capped = True
			self._finish_torch_trace()

	def read(self, header: Tensor) -> None:
		with self.recorder.span("worker.read_rows"):
			super().read(header)

	def read_many(self, header: Tensor) -> None:
		with self.recorder.span("worker.gather_rows"):
			self.recorder.count("gather.groups", int(header[4]))
			self.recorder.count("gather.copy_bytes", int(header[5]) * int(header[6]) * 4)
			super().read_many(header)

	def _reply(self, *args, **kwargs) -> None:
		with self.recorder.span("worker.completion_reply"):
			super()._reply(*args, **kwargs)

	def _finish_torch_trace(self) -> None:
		if self._torch_profile is not None:
			profile, self._torch_profile = self._torch_profile, None
			profile.__exit__(None, None, None)
			assert self._torch_path is not None
			profile.export_chrome_trace(str(self._torch_path))

	def handle_extra(self, header: Tensor) -> bool:
		if int(header[0]) != PROFILE:
			return False
		_, action, generation, trace, limit, _, _, _ = header.tolist()
		if generation != self.owner.generation or action not in (1, 2) or trace not in (0, 1):
			raise ValueError("Invalid worker profile command")
		if action == 1:
			if self.recorder.active:
				raise RuntimeError("Worker profile window is already active")
			if trace and (self.profile_dir is None or not 1 <= limit <= 1000):
				raise ValueError("Trace capture requires a directory and a bounded model-task limit")
			self._resident_peak = self.store.bytes_used
			self._incremental_before = (
				{key: getattr(self.incremental, key) for key in ("messages", "reused", "refreshes")}
				if self.incremental is not None
				else {}
			)
			self._profile_before = profile_snapshot(self.profile())
			self._profile_window += 1
			self._model_events.clear()
			self._rss = MemorySampler()
			self._rss.start()
			self._torch_path = None
			self._trace_limit, self._trace_models, self._trace_capped = limit, 0, False
			if self.comm.device.type == "cuda":
				torch.cuda.reset_peak_memory_stats(self.comm.device)
			cpu_trace = None
			if trace:
				assert self.profile_dir is not None
				self.profile_dir.mkdir(parents=True, exist_ok=True)
				prefix = self.profile_dir / f"worker-{self.owner.rank}-g{generation}-w{self._profile_window}"
				self._torch_path = prefix.with_suffix(".torch.json")
				cpu_trace = prefix.with_suffix(".phases.json")
				activities = [torch.profiler.ProfilerActivity.CPU]
				if self.comm.device.type == "cuda":
					activities.append(torch.profiler.ProfilerActivity.CUDA)
				self._torch_profile = torch.profiler.profile(activities=activities)
				self._torch_profile.__enter__()
			self.comm.send(torch.tensor([PROFILE, action, generation, 0]), 0, epoch=self.control_epoch)
			self.recorder.start(cpu_trace)
			return True
		if not self.recorder.active or self._rss is None:
			raise RuntimeError("No active worker profile window")
		# The control-loop receive was posted inside the window. Remove the stop
		# command itself so instrumentation messages do not inflate payload counts.
		self.metrics_comm.note_p2p("recv", 0, header, self.control_epoch, amount=-1)
		self._flush_cuda(wait=True)
		report = self.recorder.stop()
		report["rss"] = self._rss.stop()
		self._rss = None
		self._finish_torch_trace()
		report.update(
			{
				"rank": self.owner.rank,
				"host": socket.gethostname(),
				"pid": os.getpid(),
				"generation": generation,
				"device": str(self.comm.device),
				"resident_peak_bytes": self._resident_peak,
				"resident_current_bytes": self.store.bytes_used,
				"torch_trace": str(self._torch_path) if self._torch_path is not None else None,
				"trace_capped": self._trace_capped,
			}
		)
		report["operator_incremental"] = (
			{
				**{key: getattr(self.incremental, key) - before for key, before in self._incremental_before.items()},
				"history_bytes": self.incremental.bytes_used,
			}
			if self.incremental is not None
			else None
		)
		worker_profile = profile_snapshot(self.profile())
		report["policy_window"] = (
			profile_window(self._profile_before, worker_profile)
			if self._profile_before is not None and isinstance(worker_profile, dict)
			else None
		)
		report["worker_profile"] = worker_profile
		report["cuda"] = (
			None
			if self.comm.device.type != "cuda"
			else {
				"device_name": torch.cuda.get_device_name(self.comm.device),
				"peak_allocated_bytes": torch.cuda.max_memory_allocated(self.comm.device),
				"peak_reserved_bytes": torch.cuda.max_memory_reserved(self.comm.device),
			}
		)
		payload = json.dumps(report, allow_nan=False).encode()
		self.comm.send(torch.tensor([PROFILE, action, generation, len(payload)]), 0, epoch=self.control_epoch)
		self.comm.send(torch.frombuffer(bytearray(payload), dtype=torch.uint8), 0, epoch=self.control_epoch)
		return True

	def run(self) -> None:
		try:
			super().run()
		finally:
			if self._rss is not None:
				self._rss.stop()
			self._finish_torch_trace()
			if self.recorder.active:
				self.recorder.stop()
