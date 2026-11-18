"""RPC admission and on-demand reads for the versioned embedding service."""

import logging
from concurrent.futures import TimeoutError
from functools import wraps
from time import monotonic, perf_counter_ns
from typing import Optional, cast

import grpc
import torch

from lib.graph.admission import AdmissionFull, UpdateAdmission, UpdateKind
from lib.graph.on_demand import OnDemand
from lib.graph.ownership import MissingState, Owner
from lib.graph.serving import EmbeddingService
from lib.graph.tickets import CompletionTicket, SessionMismatch, TicketRegistry
from lib.rpc import inference_pb2 as wire
from lib.rpc.inference_pb2_grpc import OwnedInferenceServicer
from lib.rpc.tensors import TensorPayload, decode_tensor, encode_tensor
from lib.runtime.transport import ModelExecutionError, TransportFailure


def _rpc(method):
	@wraps(method)
	def invoke(self, request, context):
		try:
			return method(self, request, context)
		except SessionMismatch as error:
			context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
		except (AdmissionFull, MemoryError) as error:
			context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(error))
		except (ValueError, TypeError) as error:
			context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
		except TimeoutError:
			context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "Wait deadline elapsed; accepted work remains admitted")
		except (TransportFailure, ModelExecutionError, MissingState, OSError) as error:
			context.abort(grpc.StatusCode.UNAVAILABLE, str(error))
		except KeyError:
			context.abort(grpc.StatusCode.NOT_FOUND, "Completion ticket was released, expired, or is unknown")
		except RuntimeError as error:
			context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
		except Exception:
			logging.exception("Owned inference RPC failed")
			context.abort(grpc.StatusCode.INTERNAL, "Inference request failed; inspect the server log")

	return invoke


def _from_wire(value: wire.Tensor, limit: int) -> torch.Tensor:
	payload = TensorPayload(value.dtype, tuple(value.shape), value.data)
	if len(payload.data) > limit:
		raise AdmissionFull("RPC tensor exceeds the request byte limit")
	return decode_tensor(payload, max_bytes=limit)


def _to_wire(value: torch.Tensor) -> wire.Tensor:
	payload = encode_tensor(value)
	return wire.Tensor(dtype=cast(wire.Tensor.DType, payload.dtype), shape=payload.shape, data=payload.data)


def _ticket(entry: CompletionTicket) -> wire.Ticket:
	return wire.Ticket(session=entry.session, number=entry.number, request_id=entry.request_id)


class OwnedServicer(OwnedInferenceServicer):
	"""Sync/Async share one queue; queries preempt at ready chunk boundaries.

	Binary inputs remain private immutable views until their admitted work finishes.
	The query consumer serializes selected CPU rows while leases remain held. No
	transport, CUDA device, or communicator choices appear in the frontend contract.
	"""

	def __init__(
		self,
		service: EmbeddingService,
		query_owner: Owner,
		*,
		admission: Optional[UpdateAdmission] = None,
		demand: Optional[OnDemand] = None,
		max_pending: int = 32,
		max_bytes: int = 64 * 1024**2,
		max_tickets: int = 1024,
		ticket_retention: float = 300,
	) -> None:
		if max_bytes < 0 or max_pending < 1 or max_tickets < 1 or ticket_retention < 0:
			raise ValueError("Invalid RPC byte, admission, or ticket limits")
		if admission is not None and admission.service is not service:
			raise ValueError("Admission belongs to another embedding service")
		if demand is not None and demand.service is not service:
			raise ValueError("On-demand scheduler belongs to another embedding service")
		self.service, self.query_owner = service, query_owner
		self.max_bytes = max_bytes
		self._owns_admission, self._owns_demand = admission is None, demand is None
		self.admission = admission or UpdateAdmission(service, max_pending=max_pending, max_bytes=max_bytes)
		try:
			self.demand = demand or OnDemand(service, max_bytes=max_bytes)
		except BaseException:
			if self._owns_admission:
				self.admission.close()
			raise
		self.tickets = TicketRegistry(self.admission, capacity=max_tickets, retention=ticket_retention)
		self._started = monotonic()

	def _admit(self, request: wire.Update, *, strict: bool = False) -> CompletionTicket:
		kinds = {
			wire.Update.FEATURES: UpdateKind.FEATURES,
			wire.Update.INSERT: UpdateKind.INSERT,
			wire.Update.DELETE: UpdateKind.DELETE,
		}
		if request.kind not in kinds:
			raise ValueError("Unknown update kind")
		first = _from_wire(request.first, self.max_bytes)
		second = _from_wire(request.second, self.max_bytes - first.numel() * first.element_size())
		attributes = None
		if request.HasField("attributes"):
			if request.kind != wire.Update.INSERT:
				raise ValueError("Edge attributes are supported only for INSERT")
			remaining = self.max_bytes - sum(value.numel() * value.element_size() for value in (first, second))
			attributes = _from_wire(request.attributes, remaining)
			if attributes.dtype != torch.float32 or attributes.ndim != 2 or attributes.shape[0] != first.numel():
				raise ValueError("Edge attributes must be a float32 matrix with one row per edge")
		if first.ndim != 1 or first.dtype != torch.int64:
			raise ValueError("Update node IDs must be one-dimensional int64")
		if request.kind == wire.Update.FEATURES:
			if second.dtype != self.service.features.dtype or second.shape != (first.numel(), self.service.widths[0]):
				raise ValueError("Feature rows must match the model input width and dtype")
			if first.unique().numel() != first.numel():
				raise ValueError("Feature update IDs must be unique")
		elif second.dtype != torch.int64 or second.shape != first.shape:
			raise ValueError("Edge endpoints must be equal-length int64 vectors")
		for nodes in (first,) if request.kind == wire.Update.FEATURES else (first, second):
			if nodes.numel() and (nodes.min() < 0 or nodes.max() >= self.service.catalog.nodes):
				raise ValueError("Update node ID out of range")
		return self.tickets.submit(
			kinds[request.kind],
			first,
			second,
			request_id=request.request_id,
			session=request.session,
			strict=strict,
			attributes=attributes,
		)

	def _wait(self, entry: CompletionTicket, context, *, poll: bool = False) -> wire.Completion:
		if poll and not entry.future.done():
			return wire.Completion(ticket=_ticket(entry), ready=False, **self._telemetry(entry, observed=True))
		try:
			result = entry.result(timeout=context.time_remaining())
		except BaseException:
			# A failed or timed-out wait still records the server reply boundary.
			# The RPC error semantics remain unchanged; lifecycle data is retained
			# for a later profile/status read while the admission record is retained.
			self.admission.profile(entry.number, observed_at_ns=perf_counter_ns())
			raise
		return wire.Completion(
			ticket=_ticket(entry),
			ready=True,
			version=result.version,
			affected=result.affected,
			input_changes=result.input_changes,
			model_tasks=result.tasks,
			input_version=result.input_version,
			**self._telemetry(entry, observed=True),
		)

	def _telemetry(self, entry: CompletionTicket, *, observed: bool) -> dict:
		profile = self.admission.profile(entry.number, observed_at_ns=perf_counter_ns() if observed else None)
		if profile is None:
			return {}
		fields = {
			"admitted_at_ns": profile["admitted_at_ns"],
			"execution_started_at_ns": profile["execution_started_at_ns"],
			"staged_at_ns": profile["staged_at_ns"],
			"publication_observed_at_ns": profile["publication_observed_at_ns"],
			"completion_at_ns": profile["completion_at_ns"],
			"server_reply_prepared_at_ns": profile["server_reply_prepared_at_ns"],
			"queue_wait_ns": profile["queue_wait_ns"],
			"execution_ns": profile["execution_ns"],
			"delayed_publication_wait_ns": profile["delayed_publication_wait_ns"],
		}
		return {name: value for name, value in fields.items() if value is not None} | {
			"timestamp_scope": profile["clock_scope"],
			"publication_source": profile["publication_source"] or "",
			"state": profile["state"],
		}

	@_rpc
	def Sync(self, request, context):
		return self._wait(self._admit(request, strict=True), context)

	@_rpc
	def Async(self, request, context):
		return _ticket(self._admit(request))

	@_rpc
	def Wait(self, request, context):
		entry = self.tickets.get(request.ticket.session, request.ticket.number, request.ticket.request_id)
		return self._wait(entry, context, poll=request.poll)

	@_rpc
	def Release(self, request, context):
		self.tickets.release(request.session, request.number, request.request_id)
		return wire.Empty()

	@_rpc
	def Query(self, request, context):
		nodes = _from_wire(request.nodes, self.max_bytes)
		layer = request.layer if request.HasField("layer") else None
		future = self.demand.submit(
			nodes, self.query_owner, _to_wire, layer=layer, borrow=True, max_staleness=request.max_staleness
		)
		reply = future.result(timeout=context.time_remaining())
		return wire.QueryResponse(
			values=cast(wire.Tensor, reply.value),
			version=reply.version,
			max_age=reply.max_age,
			min_fresh_version=reply.version - reply.max_age,
		)

	@_rpc
	def Info(self, request, context):
		service, counters = self.service, self.admission.counters
		reply = wire.RuntimeInfo(
			session=self.tickets.session,
			nodes=service.catalog.nodes,
			widths=service.widths,
			graph_version=service.catalog.version,
			completed_updates=service.counters.updates,
			completed_input_changes=service.counters.input_changes,
			pending_updates=counters.pending,
			failed_updates=counters.failed,
			recovery_rows=service.counters.recovery_rows,
			recovery_seconds=service.counters.recovery_seconds,
			uptime_seconds=monotonic() - self._started,
			admitted_updates=counters.admitted,
			pending_update_bytes=counters.pending_bytes,
			max_staleness=service.deferred.max_versions if service.deferred is not None else 0,
		)
		pending = service._pending_update
		if pending is not None:
			reply.incomplete_version = pending.version
		return reply

	def close(self) -> None:
		try:
			if self._owns_admission:
				self.admission.close()
		finally:
			if self._owns_demand:
				self.demand.close()
