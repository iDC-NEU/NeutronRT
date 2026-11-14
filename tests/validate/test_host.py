from scripts.validate.host import select_devices, stages, unavailable


def test_physical_device_selection_preserves_order_and_refuses_unknown_or_tight_memory():
	listing = (
		"0, gpu0, card, 4000, 16000, 90\n1, gpu1, card, 1200, 16000, 0\n"
		"2, gpu2, card, N/A, 16000, 0\n3, gpu3, card, 5000, 16000, 0\n"
	)
	assert select_devices(["3", "0", "1", "2", "7"], listing) == ("3", "0")
	assert select_devices(["0"], "not a GPU listing") == ()


def test_one_host_plan_includes_real_gates_and_keeps_performance_after_correctness(tmp_path):
	plan = stages(tmp_path / "data", tmp_path / "output", products=True)
	by_name = {stage.name: stage for stage in plan}
	assert by_name["profiles"].dependencies == ("tests", "arxiv-rpc")
	assert "--repetitions" in by_name["profiles"].command and "--trace" in by_name["profiles"].command
	assert by_name["peer-profiles"].peers and by_name["forced-rdma"].rdma
	assert by_name["concurrent-sustained"].peers
	assert by_name["concurrent-sustained"].dependencies == ("tests", "arxiv-rpc")
	command = by_name["concurrent-sustained"].command
	assert command[command.index("--updates") + 1] == "120"
	assert dict(by_name["forced-rdma"].environment)["NCCL_P2P_DISABLE"] == "1"
	context = {"host_available_bytes": 64 * 1024**3}
	assert unavailable(by_name["profiles"], {}, ["0"], context, True) == "Prerequisite validation did not pass"
	passed = {"tests": {"status": "passed"}, "arxiv-rpc": {"status": "passed"}}
	assert unavailable(by_name["profiles"], passed, ["0"], context, True) is None
	assert unavailable(by_name["profiles"], passed, [], context, True)
	assert unavailable(by_name["peer-profiles"], passed, ["0"], context, True)
	assert unavailable(by_name["forced-rdma"], passed, ["0", "1"], context, False)
	assert unavailable(by_name["products-native"], passed, ["0"], {"host_available_bytes": 1024**3}, True)


def test_local_distribution_gates_run_without_gpu_or_rdma_configuration(tmp_path):
	by_name = {stage.name: stage for stage in stages(tmp_path / "data", tmp_path / "output")}
	for name in (
		"arxiv-storage",
		"cpu-accuracy",
		"cpu-storage-profiles",
		"arxiv-retained",
		"cpu-retained-profiles",
		"arxiv-aggregate-only",
		"cpu-aggregate-only-profiles",
		"cpu-aggregate-only-accuracy",
		"arxiv-delta",
		"cpu-delta-profiles",
		"cpu-delta-accuracy",
		"arxiv-row-refresh",
		"cpu-row-refresh-profiles",
		"cpu-row-refresh-accuracy",
		"cpu-row-refresh-drift",
		"arxiv-aggregate-delta",
		"cpu-aggregate-delta-accuracy",
		"cpu-aggregate-delta-drift",
		"cpu-aggregate-delta-profiles",
	):
		stage = by_name[name]
		assert not stage.gpu and not stage.peers and not stage.rdma
		passed = {dependency: {"status": "passed"} for dependency in stage.dependencies}
		assert unavailable(stage, passed, [], {"host_available_bytes": 64 * 1024**3}, False) is None
	assert by_name["arxiv-storage"].dependencies == ("tests",)
	assert "arxiv-storage" in by_name["cpu-accuracy"].dependencies
	assert by_name["cpu-retained-profiles"].dependencies == ("tests", "arxiv-retained")
	assert "--retain-aggregates" in by_name["arxiv-retained"].command
	assert by_name["cpu-aggregate-only-profiles"].dependencies == ("tests", "arxiv-aggregate-only")
	assert "--aggregate-only" in by_name["arxiv-aggregate-only"].command
	accuracy = by_name["cpu-aggregate-only-accuracy"]
	assert accuracy.dependencies == ("tests", "arxiv-aggregate-only")
	assert accuracy.command[accuracy.command.index("--queries") + 1] == "48603"
	assert accuracy.command[accuracy.command.index("--query-selection") + 1] == "test"
	assert "--incremental-aggregates" in by_name["arxiv-delta"].command
	assert by_name["cpu-delta-profiles"].dependencies == ("tests", "arxiv-delta")
	assert by_name["cpu-delta-accuracy"].dependencies == ("tests", "arxiv-delta")
	assert "--row-refresh" in by_name["arxiv-row-refresh"].command
	assert by_name["cpu-row-refresh-profiles"].dependencies == ("tests", "arxiv-row-refresh")
	assert "--row-refresh" in by_name["cpu-row-refresh-drift"].command
	assert "--aggregate-only" in by_name["arxiv-aggregate-delta"].command
	assert "--incremental-aggregates" in by_name["arxiv-aggregate-delta"].command
	assert "--aggregate-only" in by_name["cpu-aggregate-delta-drift"].command
	assert set(by_name["cpu-aggregate-delta-profiles"].dependencies) == {
		"tests",
		"arxiv-aggregate-delta",
		"cpu-aggregate-delta-accuracy",
		"cpu-aggregate-delta-drift",
	}


def test_products_topology_probe_requires_arxiv_storage_logic_and_probe(tmp_path):
	by_name = {stage.name: stage for stage in stages(tmp_path / "data", tmp_path / "output", products=True)}
	stage = by_name["products-storage"]
	assert set(stage.dependencies) == {"tests", "arxiv-storage", "arxiv-storage-probe"}
	assert not stage.gpu and stage.minimum_host_bytes >= 32 * 1024**3
	model = by_name["products-model"]
	assert set(model.dependencies) == {"tests", "arxiv-storage", "products-storage"}
	assert not model.gpu and not model.peers and not model.rdma
	assert model.command[model.command.index("--dataset") + 1] == "ogbn-products"
	assert model.command[model.command.index("--models") + 1] == "sage"
	assert "--checkpoint" not in model.command
