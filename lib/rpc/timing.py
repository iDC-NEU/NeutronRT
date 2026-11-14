"""Opt-in unary RPC timing, shared by existing and owned inference clients.

Elapsed intervals use each host's monotonic clock. The client/server difference
includes transport, scheduling and serialization; it is not one-way latency.
Server dispatch spans RPC interception to handler entry, including request body
receipt, decoding and executor queueing; handler time excludes response encoding.
"""

from threading import Lock
from time import perf_counter_ns

import grpc
from attrs import asdict, define, field, validators

from lib.profiling.experiment import latency_summary

HANDLER_NS = "nrt-handler-ns"
DISPATCH_NS = "nrt-dispatch-ns"


@define(frozen=True)
class RpcSample:
	method: str
	client_started_ns: int
	client_finished_ns: int
	server_handler_ns: int | None
	server_dispatch_ns: int | None
	request_bytes: int
	response_bytes: int | None
	status: str

	@property
	def seconds(self):
		return (self.client_finished_ns - self.client_started_ns) / 1e9


@define
class RpcTimings(grpc.UnaryUnaryClientInterceptor):
	capacity: int = field(default=100_000, validator=validators.ge(1))
	samples: list[RpcSample] = field(factory=list, init=False)
	dropped: int = field(default=0, init=False)
	_lock: Lock = field(factory=Lock, init=False)
	_generation: int = field(default=0, init=False)

	def clear(self):
		with self._lock:
			self.samples.clear()
			self.dropped = 0
			self._generation += 1

	def intercept_unary_unary(self, continuation, details, request):
		with self._lock:
			generation = self._generation
		started = perf_counter_ns()
		call = continuation(details, request)

		def completed(result):
			finished = perf_counter_ns()
			try:
				response = result.result()
			except (grpc.RpcError, grpc.FutureCancelledError):
				response = None
			metadata = dict(result.trailing_metadata() or ())
			try:
				duration = int(metadata[HANDLER_NS]) if HANDLER_NS in metadata else None
				dispatch = int(metadata[DISPATCH_NS]) if DISPATCH_NS in metadata else None
			except (ValueError, TypeError):
				duration = dispatch = None
			sample = RpcSample(
				details.method,
				started,
				finished,
				duration,
				dispatch,
				request.ByteSize(),
				response.ByteSize() if response is not None else None,
				result.code().name,
			)
			with self._lock:
				if generation != self._generation:
					return
				if len(self.samples) < self.capacity:
					self.samples.append(sample)
				else:
					self.dropped += 1

		call.add_done_callback(completed)
		return call

	def report(self):
		with self._lock:
			samples, dropped = list(self.samples), self.dropped
		methods = {}
		for method in sorted({sample.method for sample in samples}):
			rows = [sample for sample in samples if sample.method == method]
			server = [sample.server_handler_ns / 1e9 for sample in rows if sample.server_handler_ns is not None]
			dispatch = [sample.server_dispatch_ns / 1e9 for sample in rows if sample.server_dispatch_ns is not None]
			residual = [
				sample.seconds - sample.server_handler_ns / 1e9
				for sample in rows
				if sample.server_handler_ns is not None
			]
			methods[method] = {
				"client": latency_summary([sample.seconds for sample in rows]),
				"server_handler": latency_summary(server),
				"server_dispatch": latency_summary(dispatch),
				"transport_queue_serialization_residual": latency_summary(residual),
				"errors": sum(sample.status != "OK" for sample in rows),
			}
		return {
			"samples": [asdict(sample) for sample in samples],
			"methods": methods,
			"dropped": dropped,
			"scope": "Client RTT and server handler intervals; residual includes queues/codecs and transport",
		}


class ServerTiming(grpc.ServerInterceptor):
	def intercept_service(self, continuation, details):
		accepted = perf_counter_ns()
		handler = continuation(details)
		if handler is None or handler.unary_unary is None:
			return handler

		def invoke(request, context):
			started = perf_counter_ns()
			try:
				return handler.unary_unary(request, context)
			finally:
				metadata = tuple(context.trailing_metadata() or ())
				context.set_trailing_metadata(
					(
						*metadata,
						(HANDLER_NS, str(perf_counter_ns() - started)),
						(DISPATCH_NS, str(started - accepted)),
					)
				)

		return grpc.unary_unary_rpc_method_handler(
			invoke,
			request_deserializer=handler.request_deserializer,
			response_serializer=handler.response_serializer,
		)
