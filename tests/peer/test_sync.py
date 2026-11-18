from unittest.mock import patch

import pytest

from scripts.peer.sync import SyncTarget, sync


def test_working_tree_patch_is_rejected_before_network_if_base_is_not_head():
	with (
		patch(
			"scripts.peer.sync.subprocess.check_output",
			side_effect=["refs/heads/other\n", "other-hash\n", "head-hash\n"],
		),
		patch("scripts.peer.sync.subprocess.run") as run,
	):
		with pytest.raises(ValueError, match="equal HEAD"):
			sync(SyncTarget("peer", "/repo", "/new"), "other", working_tree=True)
		run.assert_not_called()


def test_untracked_files_cannot_silently_disappear_from_working_tree_sync():
	with (
		patch(
			"scripts.peer.sync.subprocess.check_output",
			side_effect=["refs/heads/current\n", "head\n", "head\n", "scripts/new.py\n"],
		),
		patch("scripts.peer.sync.subprocess.run") as run,
	):
		with pytest.raises(ValueError, match="git add -N"):
			sync(SyncTarget("peer", "/repo", "/new"), "HEAD", working_tree=True)
		run.assert_not_called()
