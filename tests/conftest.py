import pytest
import torch

_HOST_MODULES = {
	"tests/runtime/test_service.py",
	"tests/runtime/test_rpc.py",
	"tests/runtime/test_runtime.py",
	"tests/runtime/test_deferred_runtime.py",
	"tests/profiling/test_runtime.py",
	"tests/profiling/test_communication.py",
	"tests/dist/test_comm_epochs.py",
	"tests/dist/test_exchange_rows.py",
	"tests/dist/test_transport_abort.py",
}
_HOST_FUNCTIONS = {
	"tests/dist/test_transport.py": {
		"test_cpu_transport_roundtrip",
		"test_cpu_gpu_and_gpu_peer_transport",
		"test_single_p2p_payload_preserves_mixed_types_cpu_gpu_and_peers",
	},
	"tests/runtime/test_worker.py": {"test_process_workers_publish_handles_fetch_peers_and_defer_writeback"},
	"tests/profiling/test_worker.py": {"test_worker_windows_count_real_work_cuda_events_and_payloads_without_warmup"},
}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
	"""Keep socket/process-group checks explicit; fake/local logic remains runnable."""
	for item in items:
		module = item.path.relative_to(config.rootpath).as_posix()
		function = item.name.split("[", 1)[0]
		if module in _HOST_MODULES or function in _HOST_FUNCTIONS.get(module, set()):
			item.add_marker(pytest.mark.host)


@pytest.fixture
def random_seed():
	torch.manual_seed(42)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(42)
