from concurrent.futures import ThreadPoolExecutor
from threading import Event

import grpc
import pytest

from lib.rpc import inference_pb2 as wire
from lib.rpc.timing import RpcTimings, ServerTiming


@pytest.mark.host
def test_unary_timing_preserves_response_errors_and_trailers():
	def echo(request, context):
		context.set_trailing_metadata((("existing", "kept"),))
		if request.number:
			context.abort(grpc.StatusCode.INVALID_ARGUMENT, "bad request")
		return request

	with ThreadPoolExecutor(max_workers=2) as executor:
		server = grpc.server(executor, interceptors=[ServerTiming()])
		server.add_generic_rpc_handlers(
			(
				grpc.method_handlers_generic_handler(
					"test",
					{
						"Echo": grpc.unary_unary_rpc_method_handler(
							echo,
							request_deserializer=wire.Ticket.FromString,
							response_serializer=wire.Ticket.SerializeToString,
						),
					},
				),
			)
		)
		port = server.add_insecure_port("127.0.0.1:0")
		server.start()
		try:
			timings = RpcTimings(capacity=2)
			with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
				call = grpc.intercept_channel(channel, timings).unary_unary(
					"/test/Echo",
					request_serializer=wire.Ticket.SerializeToString,
					response_deserializer=wire.Ticket.FromString,
				)
				response, result = call.with_call(wire.Ticket(request_id="test"))
				assert response.request_id == "test"
				assert dict(result.trailing_metadata())["existing"] == "kept"
				with pytest.raises(grpc.RpcError) as error:
					call(wire.Ticket(number=1))
				assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
				call(wire.Ticket())
			report = timings.report()
			assert report["dropped"] == 1
			assert report["methods"]["/test/Echo"]["errors"] == 1
			assert all(
				s.server_handler_ns is not None and s.seconds * 1e9 >= s.server_handler_ns for s in timings.samples
			)
			timings.clear()
			assert timings.report()["samples"] == []
		finally:
			server.stop(0).wait()


@pytest.mark.host
def test_future_stays_nonblocking_and_clear_discards_old_window():
	entered, release = Event(), Event()

	def wait(request, context):
		entered.set()
		assert release.wait(5)
		return request

	with ThreadPoolExecutor(max_workers=1) as executor:
		server = grpc.server(executor, interceptors=[ServerTiming()])
		server.add_generic_rpc_handlers(
			(
				grpc.method_handlers_generic_handler(
					"test",
					{
						"Wait": grpc.unary_unary_rpc_method_handler(
							wait,
							request_deserializer=wire.Empty.FromString,
							response_serializer=wire.Empty.SerializeToString,
						),
					},
				),
			)
		)
		port = server.add_insecure_port("127.0.0.1:0")
		server.start()
		try:
			timings = RpcTimings()
			with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
				call = grpc.intercept_channel(channel, timings).unary_unary(
					"/test/Wait",
					request_serializer=wire.Empty.SerializeToString,
					response_deserializer=wire.Empty.FromString,
				)
				future = call.future(wire.Empty(), timeout=5)
				assert entered.wait(2)
				assert not future.done()
				timings.clear()
				release.set()
				future.result(timeout=5)
			assert timings.report()["samples"] == []
		finally:
			release.set()
			server.stop(0).wait()
