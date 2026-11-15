"""Host network-counter snapshots for measured runtime windows."""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Iterable

import psutil


@dataclass(frozen=True)
class NetworkSnapshot:
	"""Cumulative host counters captured at one point in time."""

	host: str
	interfaces: dict[str, dict[str, int]]


def snapshot(interfaces: Iterable[str] | None = None) -> NetworkSnapshot:
	selected = None if interfaces is None else frozenset(interfaces)
	values: dict[str, dict[str, int]] = {}
	for name, counter in psutil.net_io_counters(pernic=True).items():
		if selected is not None and name not in selected:
			continue
		values[name] = {
			"bytes_sent": int(counter.bytes_sent),
			"bytes_recv": int(counter.bytes_recv),
			"packets_sent": int(counter.packets_sent),
			"packets_recv": int(counter.packets_recv),
			"errin": int(counter.errin),
			"errout": int(counter.errout),
			"dropin": int(counter.dropin),
			"dropout": int(counter.dropout),
		}
	return NetworkSnapshot(socket.gethostname(), values)


def delta(before: NetworkSnapshot, after: NetworkSnapshot) -> dict:
	if before.host != after.host:
		raise ValueError("Network snapshots belong to different hosts")
	keys = (
		"bytes_sent",
		"bytes_recv",
		"packets_sent",
		"packets_recv",
		"errin",
		"errout",
		"dropin",
		"dropout",
	)
	interfaces = {}
	for name in sorted(set(before.interfaces) | set(after.interfaces)):
		old, new = before.interfaces.get(name, {}), after.interfaces.get(name, {})
		interfaces[name] = {key: int(new.get(key, 0) - old.get(key, 0)) for key in keys}
	totals = {key: sum(value[key] for value in interfaces.values()) for key in keys}
	return {
		"host": after.host,
		"interfaces": interfaces,
		"totals": totals,
		"scope": (
			"Operating-system interface counters during the measurement window; "
			"includes unrelated host traffic and excludes startup traffic outside the window"
		),
	}
