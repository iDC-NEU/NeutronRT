"""Export one immutable topology base for shared, file-backed CPU storage."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	from dgl import DGLGraph


@dataclass(frozen=True)
class BaseTopology:
	folder: Path
	nodes: int
	edges: int

	def files(self, direction: str) -> tuple[Path, Path]:
		if direction not in ("in", "out"):
			raise ValueError("Unknown adjacency direction")
		return self.folder / f"{direction}-offsets.npy", self.folder / f"{direction}-neighbors.npy"

	@classmethod
	def export(cls, graph: "DGLGraph", folder: Path) -> "BaseTopology":
		"""Export one temporary sparse view at a time, preserving the caller's layout.

		The caller owns the fresh directory and its cleanup. This preprocessing is
		outside serving throughput measurements and must be reported separately.
		"""
		if graph.device.type != "cpu" or str(graph.idtype) != "torch.int64" or not graph.is_homogeneous:
			raise ValueError("Mapped topology requires a homogeneous CPU int64 graph")
		folder.mkdir(parents=True, exist_ok=True)
		if any(folder.iterdir()):
			raise ValueError("Topology export requires an empty owned directory")
		base = cls(folder.resolve(), graph.num_nodes(), graph.num_edges())
		for direction, layout in (("in", "csc"), ("out", "csr")):
			view = graph.formats(layout)
			for path, values in zip(base.files(direction), view.adj_tensors(layout)[:2], strict=True):
				array = values.numpy()
				if array.dtype != np.int64 or not array.flags.c_contiguous:
					raise ValueError("Topology export requires contiguous int64 adjacency buffers")
				np.save(path, array, allow_pickle=False)
			del view, values, array
		return base
