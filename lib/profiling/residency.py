"""Read published logical footprints after a measurement window, without copies."""

from typing import Dict, TypedDict

from lib.graph.ownership import MissingState, Owner
from lib.runtime.data_plane import RemoteEmbeddingPlane


class ReplicaBytes(TypedDict):
	rank: int
	generation: int
	device: str
	h_bytes: int
	aggregate_bytes: int


def retained_state(plane: RemoteEmbeddingPlane) -> dict:
	"""Logical replicas by state kind; allocator/RSS peaks have separate scopes."""
	with plane._lock:
		owners = (plane.storage.owner, *plane.remotes)
		rows: Dict[Owner, ReplicaBytes] = {
			owner: {
				"rank": owner.rank,
				"generation": owner.generation,
				"device": "cpu" if owner == plane.storage.owner else str(plane.remotes[owner].device),
				"h_bytes": 0,
				"aggregate_bytes": 0,
			}
			for owner in owners
		}
		for key, (count, width) in plane._shapes.items():
			for owner in owners:
				try:
					location = plane.directory.locate(key, preferred=owner)
				except MissingState:
					break
				if location.owner == owner:
					if key.kind == "h":
						rows[owner]["h_bytes"] += count * width * 4
					else:
						rows[owner]["aggregate_bytes"] += count * width * 4
		return {
			"scope": "End-of-window published float32 replica bytes; logical shapes, not allocator or RSS peaks",
			"owners": list(rows.values()),
			"storage_resident_bytes": plane.storage.store.bytes_used,
		}
