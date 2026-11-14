from unittest.mock import Mock

import pytest
import torch

from lib.dist.comm_manager import CommManager
from lib.graph.ownership import Location
from lib.runtime.protocol import ERROR, READ, READY, pack_compute
from lib.runtime.transport import ModelExecutionError, OwnedTransport, RemoteEndpoint, TransportFailure
from tests.runtime.test_protocol import _attempt, _task


def _transport(reply=None):
	attempt = _attempt(1)
	packet = pack_compute(_task(1), attempt, width=4, features=torch.zeros(3, 4))
	manager = Mock(spec=CommManager)
	manager.rank, manager.device = 0, torch.device("cpu")
	manager.empty.side_effect = lambda shape, **kwargs: torch.empty(shape, **kwargs)
	events = []
	manager.send.side_effect = lambda *args, **kwargs: events.append("send")
	manager.isend_tensors.return_value.wait.side_effect = lambda: events.append("body-ready")
	values = [READY, attempt.number, attempt.owner.generation, 7, 7, 1, 33, 2, 4, 32] if reply is None else reply

	def receive(target, *args, **kwargs):
		events.append("post-receive")
		request = Mock()

		def wait():
			events.append("wait")
			target.copy_(torch.tensor(values).reshape(target.shape))

		request.wait.side_effect = wait
		return request

	manager.irecv.side_effect = receive
	endpoint = RemoteEndpoint(attempt.owner, torch.device("cpu"), 1024)
	return OwnedTransport(manager, 12, [endpoint]), packet, events


def test_completion_uses_control_epoch_after_all_inputs_are_supplied():
	transport, packet, events = _transport()
	owner = next(iter(transport.endpoints))
	result = transport.compute(owner, packet, output_width=4, supply_inputs=lambda: events.append("inputs"))
	assert events.index("inputs") < events.index("wait")
	assert transport.comm.send.call_args.kwargs == {"epoch": 12}
	assert transport.comm.irecv.call_args.kwargs == {"epoch": 12}
	assert transport.comm.isend_tensors.call_args.kwargs == {}
	assert result.handle == 7 and result.shape == (2, 4)
	assert transport.endpoints[owner].bytes_used == 32


@pytest.mark.parametrize(
	"index,value", [(0, 1), (1, 10), (2, 4), (3, -1), (4, 8), (5, 2), (6, 34), (7, 3), (8, 5), (9, -1)]
)
def test_stale_or_malformed_completions_cannot_publish_a_ready_result(index, value):
	reply = [READY, 9, 3, 7, 7, 1, 33, 2, 4, 32]
	reply[index] = value
	transport, packet, _ = _transport(reply)
	owner = next(iter(transport.endpoints))
	with pytest.raises(RuntimeError):
		transport.compute(owner, packet, output_width=4, supply_inputs=lambda: None)
	assert transport.endpoints[owner].bytes_used == 0


@pytest.mark.parametrize("code,exception", [(1, MemoryError), (2, ModelExecutionError)])
def test_worker_errors_remain_distinct_from_publication(code, exception):
	transport, packet, _ = _transport([ERROR, 9, 3, code, 7, 1, 33, 2, 4, 0])
	owner = next(iter(transport.endpoints))
	with pytest.raises(exception):
		transport.compute(owner, packet, output_width=4, supply_inputs=lambda: None)


def test_selected_read_posts_its_receive_before_releasing_the_command_sequence():
	transport, _, events = _transport([1, 2, 3, 4, 5, 6, 7, 8])
	owner = next(iter(transport.endpoints))
	rows = torch.tensor([5, 1])
	pending = transport.read(Location(owner, 7), rows, 4)
	assert events == ["send", "send", "post-receive"]
	header = transport.comm.send.call_args_list[0].args[0]
	assert header.tolist() == [READ, 0, owner.generation, 7, 0, 2, 4, 0]
	assert transport.comm.send.call_args_list[1].args[0] is rows
	assert transport.comm.irecv.call_args.kwargs == {"name": "owned-rows"}
	torch.testing.assert_close(pending.result(), torch.arange(1, 9, dtype=torch.float32).reshape(2, 4))


@pytest.mark.parametrize("phase", ["send", "body", "reply", "read"])
def test_failed_io_is_distinguished_from_model_errors_and_bad_metadata(phase):
	transport, packet, _ = _transport()
	owner = next(iter(transport.endpoints))
	error = RuntimeError("Connection closed by peer")
	if phase == "send":
		transport.comm.send.side_effect = error
	elif phase == "body":
		transport.comm.isend_tensors.return_value.wait.side_effect = error
	else:
		transport.comm.irecv.side_effect = lambda *args, **kwargs: Mock(wait=Mock(side_effect=error))
	with pytest.raises(TransportFailure):
		if phase == "read":
			transport.read(Location(owner, 7), torch.tensor([0]), 4).result()
		else:
			transport.compute(owner, packet, output_width=4, supply_inputs=lambda: None)
