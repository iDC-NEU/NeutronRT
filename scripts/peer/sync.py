"""Copy a committed revision into a fresh remote worktree, preserving existing checkouts."""

import hashlib
import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer
from attrs import define


@define(frozen=True)
class SyncTarget:
	host: str
	repository: str
	destination: str


def sync(target: SyncTarget, revision: str, *, working_tree: bool = False) -> dict:
	ref = subprocess.check_output(["git", "rev-parse", "--symbolic-full-name", revision], text=True).strip()
	if not ref.startswith("refs/heads/"):
		raise ValueError("Sync a local branch so the bundle retains a named revision")
	commit = subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
	if working_tree:
		head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
		if head != commit:
			raise ValueError("--working-tree requires the selected revision to equal HEAD")
		untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], text=True)
		if untracked.strip():
			raise ValueError("Mark intended new files with git add -N before working-tree sync; untracked files remain")
	patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"]) if working_tree else b""
	branch = f"sync/{ref.removeprefix('refs/heads/')}/{commit[:12]}-{uuid4().hex[:8]}"
	# No checkout, reset, clean, stash, or force update touches the remote's current branch.
	remote = """
import json,pathlib,subprocess,sys,tempfile
repository,destination,commit,branch,source_ref=sys.argv[1:]
if pathlib.Path(destination).exists():
 raise SystemExit('Destination already exists; choose a new worktree path')
with tempfile.TemporaryDirectory(prefix='neutronrt-sync-') as folder:
 bundle=pathlib.Path(folder)/'revision.bundle'
 with bundle.open('xb') as output:
  import shutil
  shutil.copyfileobj(sys.stdin.buffer,output)
 subprocess.run(['git','-C',repository,'bundle','verify',str(bundle)],check=True,stdout=sys.stderr)
 subprocess.run(['git','-C',repository,'fetch',str(bundle),source_ref],check=True,stdout=sys.stderr)
 subprocess.run(['git','-C',repository,'worktree','add','-b',branch,destination,commit],check=True,stdout=sys.stderr)
 actual=subprocess.check_output(['git','-C',destination,'rev-parse','HEAD'],text=True).strip()
 assert actual==commit
 print(json.dumps({'destination':destination,'branch':branch,'commit':actual}))
"""
	with tempfile.TemporaryDirectory(prefix="neutronrt-sync-") as folder:
		bundle = Path(folder) / "revision.bundle"
		subprocess.run(["git", "bundle", "create", str(bundle), ref], check=True)
		with bundle.open("rb") as stream:
			result = subprocess.run(
				[
					"ssh",
					"-oBatchMode=yes",
					"-oConnectTimeout=15",
					target.host,
					shlex.join(["python3", "-c", remote, target.repository, target.destination, commit, branch, ref]),
				],
				stdin=stream,
				stdout=subprocess.PIPE,
				text=False,
				check=True,
			)
	if patch:
		subprocess.run(
			[
				"ssh",
				"-oBatchMode=yes",
				"-oConnectTimeout=15",
				target.host,
				shlex.join(["git", "-C", target.destination, "apply", "-"]),
			],
			input=patch,
			check=True,
		)
	return {
		"host": target.host,
		**json.loads(result.stdout),
		"working_tree_patch_sha256": hashlib.sha256(patch).hexdigest() if patch else None,
	}


def main(
	host: Annotated[str, typer.Option(help="SSH host from command line or SSH config")],
	repository: Annotated[str, typer.Option(help="Existing remote Git repository")],
	destination: Annotated[str, typer.Option(help="New remote worktree path")],
	revision: str = "HEAD",
	working_tree: bool = False,
):
	"""Sync code into a new worktree; optionally apply the HEAD diff (including intent-to-add files)."""
	typer.echo(
		json.dumps(sync(SyncTarget(host, repository, destination), revision, working_tree=working_tree), indent=2)
	)


if __name__ == "__main__":
	typer.run(main)
