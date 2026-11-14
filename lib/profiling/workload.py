"""Portable recordings of the exact requests used by streaming experiments."""

import hashlib
from base64 import b64decode, b64encode
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

from lib.models.spec import ModelSpec
from lib.rpc import inference_pb2 as wire
from lib.rpc.service import _from_wire

MAX_TRACE_BYTES = 256 * 1024**2


@dataclass(frozen=True)
class RecordedWorkload:
	spec: ModelSpec
	input_identity: str
	producer_source: str
	warmup: tuple[wire.Update, ...]
	steps: tuple[tuple[wire.Update, wire.Tensor], ...]
	verification_nodes: Tensor
	initialization: str
	seed: int
	repetition: int
	atol: float
	rtol: float

	@property
	def digest(self) -> str:
		value = hashlib.sha256()
		for request in self.warmup:
			value.update(request.SerializeToString(deterministic=True))
		for request, query in self.steps:
			value.update(request.SerializeToString(deterministic=True))
			nodes = _from_wire(query, MAX_TRACE_BYTES)
			value.update(nodes.numpy().tobytes())
		return value.hexdigest()

	def save(self, path: Path) -> None:
		warmup = [value.SerializeToString(deterministic=True) for value in self.warmup]
		steps = [
			(request.SerializeToString(deterministic=True), query.SerializeToString(deterministic=True))
			for request, query in self.steps
		]
		if sum(map(len, warmup)) + sum(len(request) + len(query) for request, query in steps) > MAX_TRACE_BYTES:
			raise ValueError("Recorded workload exceeds its trace budget")
		payload = {
			"schema": 1,
			"spec": asdict(replace(self.spec, checkpoint=None)),
			"checkpoint_required": self.spec.checkpoint is not None,
			"input_identity": self.input_identity,
			"producer_source": self.producer_source,
			# PyTorch 2.4's restricted loader accepts strings but not pickled bytes.
			"warmup": [b64encode(value).decode("ascii") for value in warmup],
			"steps": [
				(b64encode(request).decode("ascii"), b64encode(query).decode("ascii")) for request, query in steps
			],
			"verification_nodes": self.verification_nodes,
			"initialization": self.initialization,
			"seed": self.seed,
			"repetition": self.repetition,
			"atol": self.atol,
			"rtol": self.rtol,
			"trace_digest": self.digest,
		}
		with path.open("xb") as stream:
			torch.save(payload, stream)

	@classmethod
	def load(cls, path: Path, *, checkpoint: Path | None = None) -> "RecordedWorkload":
		if path.stat().st_size > 1024**3:
			raise ValueError("Recorded workload file exceeds the input budget")
		payload = torch.load(path, map_location="cpu", weights_only=True)
		if payload["schema"] != 1:
			raise ValueError("Unsupported recorded workload schema")
		if payload["checkpoint_required"] != (checkpoint is not None):
			raise ValueError(
				"Supply the original checkpoint for a trained workload, and none for a random-weight workload"
			)
		warmup = [b64decode(value, validate=True) for value in payload["warmup"]]
		steps = [
			(b64decode(request, validate=True), b64decode(query, validate=True)) for request, query in payload["steps"]
		]
		if sum(map(len, warmup)) + sum(len(request) + len(query) for request, query in steps) > MAX_TRACE_BYTES:
			raise ValueError("Recorded workload exceeds its trace budget")
		result = cls(
			spec=ModelSpec(**(payload["spec"] | {"checkpoint": str(checkpoint) if checkpoint else None})),
			input_identity=payload["input_identity"],
			producer_source=payload["producer_source"],
			warmup=tuple(wire.Update.FromString(value) for value in warmup),
			steps=tuple((wire.Update.FromString(request), wire.Tensor.FromString(query)) for request, query in steps),
			verification_nodes=payload["verification_nodes"],
			initialization=payload["initialization"],
			seed=payload["seed"],
			repetition=payload["repetition"],
			atol=payload["atol"],
			rtol=payload["rtol"],
		)
		if result.verification_nodes.dtype != torch.int64 or result.verification_nodes.ndim != 1:
			raise ValueError("Verification node IDs must be an int64 vector")
		if result.digest != payload["trace_digest"]:
			raise ValueError("Recorded workload digest does not match its requests")
		return result
