"""Keep model/graph code independent of process orchestration and transport."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
	"package,forbidden",
	[
		("models", ("lib.runtime", "lib.rpc", "lib.profiling", "lib.dist")),
		("graph", ("lib.runtime", "lib.rpc", "lib.profiling", "lib.dist")),
		("dist", ("lib.models", "lib.graph", "lib.storage", "lib.runtime", "lib.rpc", "lib.profiling")),
	],
)
def test_core_packages_do_not_depend_on_orchestration(package, forbidden):
	violations = []
	for path in (ROOT / "lib" / package).rglob("*.py"):
		for node in ast.walk(ast.parse(path.read_text())):
			if isinstance(node, ast.ImportFrom):
				modules = [node.module or ""]
			elif isinstance(node, ast.Import):
				modules = [alias.name for alias in node.names]
			else:
				continue
			for module in modules:
				if any(module == prefix or module.startswith(prefix + ".") for prefix in forbidden):
					violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {module}")
	assert not violations, "\n".join(violations)


def test_removed_runtime_has_no_forwarding_modules():
	removed = (
		"lib/dist/inference",
		"lib/dist/scheduler.py",
		"lib/dist/batcher.py",
		"lib/dist/worker.py",
		"lib/dist/server.py",
		"lib/dist/shared.py",
		"lib/dist/neutronrt.proto",
		"lib/graph/graph_manager.py",
		"lib/models/convs.py",
		"lib/utils/cache.py",
		"scripts/serve_inference.py",
		"scripts/rpc_performance.py",
		"scripts/draw.py",
		"scripts/draw_perf.py",
		"scripts/logparser.py",
		"scripts/commde.py",
		"scripts/workload.py",
		"scripts/throughput.py",
		"scripts/summarize.py",
		"scripts/partition.py",
		"scripts/overlaps.py",
		"scripts/influence.py",
		"lib/dataset/prepared.py",
	)
	assert not [name for name in removed if (ROOT / name).exists()]


def test_entrypoints_use_typer_without_argparse_adapters():
	violations = []
	for path in (ROOT / "scripts").rglob("*.py"):
		if path.name == "__init__.py":
			continue
		nodes = tuple(ast.walk(ast.parse(path.read_text())))
		if not any(
			isinstance(node, ast.Call)
			and isinstance(node.func, ast.Attribute)
			and isinstance(node.func.value, ast.Name)
			and node.func.value.id == "typer"
			and node.func.attr in ("Typer", "run")
			for node in nodes
		):
			violations.append(f"{path.name}: missing native Typer entrypoint")
		for node in nodes:
			if isinstance(node, ast.Import) and any(item.name == "argparse" for item in node.names):
				violations.append(path.name)
			if isinstance(node, ast.ImportFrom) and node.module == "argparse":
				violations.append(path.name)
	assert not violations, violations


def test_validation_and_evaluation_commands_have_one_canonical_package_location():
	root = ROOT / "scripts"
	assert not list(root.glob("validate_*.py"))
	assert not list(root.glob("evaluate_*.py"))
	assert not (root / "distributed_experiment.py").exists()
	assert not (root / "profile_inference.py").exists()
	assert not (root / "performance_report.py").exists()
