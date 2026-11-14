import pandas as pd
import torch
from typer.testing import CliRunner

from lib.dataset.raw_ogb import RawOGBDataset, has_raw_ogb
from lib.models.spec import ModelSpec
from scripts.train_baseline import app


def _raw(root):
	directory = root / "arxiv"
	raw = directory / "raw"
	raw.mkdir(parents=True)
	for name, rows in {
		"edge": [[0, 1], [1, 2], [2, 3]],
		"num-node-list": [[4]],
		"num-edge-list": [[3]],
		"node-feat": [[0.1, 0.2], [0.3, 0.1], [0.2, 0.5], [0.5, 0.2]],
		"node-label": [[0], [1], [0], [1]],
		"node_year": [[2010], [2011], [2012], [2013]],
	}.items():
		pd.DataFrame(rows).to_csv(raw / f"{name}.csv.gz", index=False, header=False)
	split = directory / "split" / "time"
	split.mkdir(parents=True)
	for name, rows in {"train": [0, 1], "valid": [2], "test": [3]}.items():
		pd.DataFrame(rows).to_csv(split / f"{name}.csv.gz", index=False, header=False)
	return directory


def test_raw_loader_uses_official_layout_and_keeps_processed_copies_out_of_storage(tmp_path):
	directory = _raw(tmp_path)
	data = RawOGBDataset("ogbn-arxiv", tmp_path)
	assert data.num_classes == 40 and len(data) == 1 and has_raw_ogb("ogbn-arxiv", tmp_path)
	assert data[0].num_nodes() == 4 and data[0].num_edges() == 3
	assert data[0].ndata["feat"].dtype == torch.float32
	assert data[0].ndata["label"].tolist() == [0, 1, 0, 1]
	assert data.get_idx_split()["test"].tolist() == [3]
	assert sorted(path.name for path in directory.iterdir()) == ["raw", "split"]


def test_small_training_command_exports_a_checkpoint_that_replacement_workers_can_load(tmp_path):
	_raw(tmp_path)
	checkpoint, report = tmp_path / "baseline.pt", tmp_path / "baseline.json"
	result = CliRunner().invoke(
		app,
		[
			"--root",
			str(tmp_path),
			"--device",
			"cpu",
			"--epochs",
			"1",
			"--hidden",
			"4",
			"--output",
			str(checkpoint),
			"--report",
			str(report),
		],
	)
	assert result.exit_code == 0, result.output

	model = ModelSpec("sage", 2, 4, 40, checkpoint=str(checkpoint)).build()
	assert not model.training and checkpoint.is_file() and report.is_file()
