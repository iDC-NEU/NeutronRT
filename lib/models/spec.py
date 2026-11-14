"""Reproducible model construction without sending graph or feature storage."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

from lib.models import MODEL_NAMES, GNNModel


@dataclass(frozen=True)
class ModelSpec:
	name: str
	features: int
	hidden: int
	classes: int
	layers: int = 2
	heads: int = 2
	seed: int = 42
	checkpoint: Optional[str] = None
	relations: int = 4
	pseudo_dim: int = 2
	kernels: int = 3

	@classmethod
	def from_options(cls, features: int, *, checkpoint: Optional[str] = None, **options) -> "ModelSpec":
		config = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_config"] if checkpoint else {}
		keys = {
			"name": "model_name",
			"hidden": "num_hiddens",
			"classes": "num_classes",
			"layers": "num_layers",
			"heads": "num_heads",
			"relations": "relations",
			"pseudo_dim": "pseudo_dim",
			"seed": "seed",
			"kernels": "kernels",
		}
		defaults: dict[str, Any] = {"name": "sage", "hidden": 64}
		values: dict[str, Any] = {name: config[key] for name, key in keys.items() if key in config}
		values = defaults | values | {name: value for name, value in options.items() if value is not None}
		if "classes" not in values:
			raise ValueError("Pass --classes or a checkpoint containing the output dimension")
		return cls(features=features, checkpoint=checkpoint, **values)

	def __post_init__(self) -> None:
		if self.name not in MODEL_NAMES:
			raise ValueError(f"Unknown model {self.name}; choose from {', '.join(MODEL_NAMES)}")
		if (
			min(
				self.features,
				self.hidden,
				self.classes,
				self.layers,
				self.heads,
				self.relations,
				self.pseudo_dim,
				self.kernels,
			)
			< 1
		):
			raise ValueError("Model dimensions must be positive")

	@property
	def widths(self) -> Tuple[int, ...]:
		return (self.features, *(self.hidden for _ in range(self.layers - 1)), self.classes)

	def build(self) -> GNNModel:
		# Rebuilding a worker must reproduce its weights without changing the
		# coordinator's random stream. Only the worker moves the model to CUDA.
		with torch.random.fork_rng(devices=[]):
			torch.manual_seed(self.seed)
			model = GNNModel(
				self.name,
				self.features,
				self.hidden,
				self.classes,
				self.layers,
				aggregator_type="mean",
				num_heads=self.heads,
				relations=self.relations,
				pseudo_dim=self.pseudo_dim,
				kernels=self.kernels,
			).eval()
		if self.checkpoint is not None:
			checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
			expected = {
				"model_name": self.name,
				"num_features": self.features,
				"num_hiddens": self.hidden,
				"num_classes": self.classes,
				"num_layers": self.layers,
			}
			if any(checkpoint["model_config"].get(key) != value for key, value in expected.items()):
				raise ValueError("Checkpoint model configuration differs from the worker specification")
			model.load_state_dict(checkpoint["state_dict"])
		for layer in model.layers:
			if hasattr(layer, "set_allow_zero_in_degree"):
				layer.set_allow_zero_in_degree(True)
		return model
