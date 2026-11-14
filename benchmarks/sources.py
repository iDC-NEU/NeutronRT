"""Materialize the exact upstream revisions used in baseline comparisons."""

import io
import json
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PinnedSource:
	name: str
	revision: str
	repository: str

	@classmethod
	def load(cls, name: str) -> "PinnedSource":
		manifest = Path(__file__).with_name("sources.json")
		values = json.loads(manifest.read_text())
		record = values["icde"] if name == "icde" else next(item for item in values["references"] if item["id"] == name)
		return cls(name, record["revision"], record.get("repository", f"git:{record.get('branch', '')}"))

	def snapshot(self, checkout: Path, destination: Path) -> Path:
		"""Reject a changed checkout, then export committed files only."""

		def git(*arguments: str) -> bytes:
			return subprocess.check_output(["git", "-C", str(checkout), *arguments], stderr=subprocess.PIPE)

		if git("rev-parse", "HEAD").decode().strip() != self.revision:
			raise ValueError(f"{self.name} requires upstream revision {self.revision}")
		if git("status", "--porcelain", "--untracked-files=all"):
			raise ValueError(f"{self.name} checkout has local changes; use an unchanged pinned checkout")
		archive = git("archive", "--format=tar", self.revision)
		destination.mkdir(parents=True, exist_ok=False)
		with tarfile.open(fileobj=io.BytesIO(archive)) as source:
			source.extractall(destination, filter="data")
		return destination
