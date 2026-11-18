import zipfile
from unittest.mock import patch

import pytest

from lib.dataset.offline_ogb import papers100m


def _papers_metadata():
	from pathlib import Path

	import ogb.nodeproppred
	import pandas as pd

	return pd.read_csv(Path(ogb.nodeproppred.__file__).parent / "master.csv", index_col=0, keep_default_na=False)[
		"ogbn-papers100M"
	]


def test_incomplete_offline_ogb_never_calls_constructor_or_downloads(tmp_path):
	with patch("lib.dataset.offline_ogb.DglNodePropPredDataset") as reader:
		with pytest.raises(FileNotFoundError, match="release marker"):
			papers100m(tmp_path)
		reader.assert_not_called()


def test_offline_ogb_accepts_complete_raw_files_and_forwards_exact_directory(tmp_path):
	path = tmp_path / "ogbn_papers100M"
	(path / "raw").mkdir(parents=True)
	# Read the installed metadata so the fixture follows the frozen OGB version.
	metadata = _papers_metadata()
	(path / ("RELEASE_v" + str(metadata["version"]) + ".txt")).write_text("fixture")
	with patch("lib.dataset.offline_ogb.DglNodePropPredDataset") as reader:
		with pytest.raises(FileNotFoundError, match="downloaded raw file"):
			papers100m(tmp_path)
		reader.assert_not_called()
		(path / "raw/data.npz").write_bytes(b"fixture")
		(path / "raw/node-label.npz").write_bytes(b"fixture")
		assert papers100m(tmp_path) is reader.return_value
		assert reader.call_args.kwargs["meta_dict"]["dir_path"] == str(path)


def test_offline_ogb_extracts_complete_zip_before_calling_constructor(tmp_path):
	metadata = _papers_metadata()
	prefix = metadata["download_name"]
	archive = tmp_path / "papers100M-bin.zip"
	with zipfile.ZipFile(archive, "w") as stream:
		stream.writestr(f"{prefix}/RELEASE_v{metadata['version']}.txt", "fixture")
		stream.writestr(f"{prefix}/raw/data.npz", b"fixture")
		stream.writestr(f"{prefix}/raw/node-label.npz", b"fixture")

	with patch("lib.dataset.offline_ogb.DglNodePropPredDataset") as reader:
		assert papers100m(tmp_path) is reader.return_value
	assert (tmp_path / "ogbn_papers100M" / "raw/data.npz").read_bytes() == b"fixture"
	assert reader.call_args.kwargs["meta_dict"]["dir_path"] == str(tmp_path / "ogbn_papers100M")


def test_offline_ogb_does_not_publish_incomplete_zip(tmp_path):
	metadata = _papers_metadata()
	prefix = metadata["download_name"]
	with zipfile.ZipFile(tmp_path / "papers100M-bin.zip", "w") as stream:
		stream.writestr(f"{prefix}/raw/data.npz", b"fixture")
		stream.writestr(f"{prefix}/raw/node-label.npz", b"fixture")

	with patch("lib.dataset.offline_ogb.DglNodePropPredDataset") as reader:
		with pytest.raises(FileNotFoundError, match="release marker"):
			papers100m(tmp_path)
		reader.assert_not_called()
	assert not (tmp_path / "ogbn_papers100M").exists()
