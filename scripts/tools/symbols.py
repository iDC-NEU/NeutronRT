"""Find and extract Python definitions from a Git working tree."""

import argparse
import ast
import io
import json
import os
import re
import subprocess
import sys
import tokenize
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from math import log1p, sqrt
from pathlib import Path

Definition = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
Candidate = tuple["Symbol", str, str]  # Symbol, docstring, and source.


@dataclass(frozen=True)
class Symbol:
	name: str
	qualified_name: str
	kind: str
	path: str
	start_line: int
	end_line: int
	definition_line: int

	@property
	def line_count(self) -> int:
		return self.end_line - self.start_line + 1


def _definitions(node: ast.AST, scope: str = "", in_class: bool = False) -> Iterator[tuple[Definition, str, str]]:
	if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
		scope = f"{scope}.{node.name}" if scope else node.name
		if isinstance(node, ast.ClassDef):
			kind = "class"
		elif in_class:
			kind = "method"
		else:
			kind = "function"
		yield node, scope, kind
		in_class = kind == "class"
	for child in ast.iter_child_nodes(node):
		yield from _definitions(child, scope, in_class)


def _decorator_lines(source: str) -> list[int]:
	lines = []
	at_statement_start = True
	for token in tokenize.generate_tokens(io.StringIO(source).readline):
		if token.type == tokenize.NEWLINE:
			at_statement_start = True
		elif token.type not in (tokenize.NL, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
			if at_statement_start and token.type == tokenize.OP and token.string == "@":
				lines.append(token.start[0])
			at_statement_start = False
	return lines


def _symbols(relative_path: str, source: str, include_docstrings: bool) -> Iterator[tuple[Symbol, str]]:
	tree = ast.parse(source, filename=relative_path)
	lines = source.split("\n")
	decorator_lines = None
	for node, qualified_name, kind in _definitions(tree):
		start = node.lineno
		if node.decorator_list:
			# A parenthesized decorator's expression can start below its @.
			start = node.decorator_list[0].lineno
			if not lines[start - 1].lstrip().startswith("@"):
				if decorator_lines is None:
					decorator_lines = _decorator_lines(source)
				start = decorator_lines[bisect_right(decorator_lines, start) - 1]
		assert node.end_lineno is not None
		symbol = Symbol(node.name, qualified_name, kind, relative_path, start, node.end_lineno, node.lineno)
		docstring = ast.get_docstring(node, clean=False) if include_docstrings else None
		yield symbol, docstring or ""


def _fuzzy_rank(query: str, name: str) -> tuple[int, float] | None:
	if query in name:
		return (1, len(name) - len(query))
	remaining = iter(name)
	if all(character in remaining for character in query):
		return (2, len(name) - len(query))
	if len(query) >= 4:
		similarity = SequenceMatcher(None, query, name, autojunk=False).ratio()
		if similarity >= 0.8:
			return (3, 1 - similarity)
	return None


def _name_rank(query: str, name: str, qualified_name: str, fuzzy: bool) -> tuple[int, float] | None:
	if query == qualified_name:
		return (0, 0)
	if query == name:
		return (1, len(qualified_name) - len(name))
	if not fuzzy:
		return None
	rank = _fuzzy_rank(query, name)
	if rank is not None:
		return (rank[0] + 1, rank[1])
	if name == qualified_name:
		return None
	rank = _fuzzy_rank(query, qualified_name)
	return (rank[0] + 4, rank[1]) if rank is not None else None


def _terms(text: str) -> Counter[str]:
	# Split snake_case, camelCase, and acronyms such as RPCService.
	text = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", text)
	text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
	return Counter(re.findall(r"[^\W_]+", text.casefold()))


def _text_scores(query: str, candidates: list[Candidate], ranker: str) -> Iterator[float]:
	documents = [_terms(f"{symbol.qualified_name} {docstring}") for symbol, docstring, _ in candidates]
	frequencies: Counter[str] = Counter()
	for document in documents:
		frequencies.update(document.keys())
	query_terms = _terms(query)
	count = len(documents)
	if ranker == "bm25":
		k1, b = 1.2, 0.75
		average_length = sum(sum(document.values()) for document in documents) / count or 1.0
		idf = {term: log1p((count - frequencies[term] + 0.5) / (frequencies[term] + 0.5)) for term in query_terms}
		for document in documents:
			normalization = k1 * (1 - b + b * sum(document.values()) / average_length)
			yield sum(
				idf[term] * (k1 + 1) * document[term] / (document[term] + normalization)
				for term in query_terms
				if term in document
			)
	else:
		idf = {term: 1 + log1p(count) - log1p(frequency) for term, frequency in frequencies.items()}
		query_vector = {term: weight * idf[term] for term, weight in query_terms.items() if term in idf}
		query_norm = sqrt(sum(weight**2 for weight in query_vector.values()))
		for document in documents:
			if not query_norm or not document:
				yield 0.0
				continue
			vector = {term: weight * idf[term] for term, weight in document.items()}
			norm = sqrt(sum(weight**2 for weight in vector.values()))
			yield sum(weight * vector.get(term, 0.0) for term, weight in query_vector.items()) / (query_norm * norm)


def _rank_candidates(
	query: str, candidates: list[Candidate], mode: str | None, pattern: re.Pattern[str] | None, ranker: str
) -> list[Candidate]:
	use_text = mode is None and ranker != "fuzzy"
	scores = _text_scores(query, candidates, ranker) if use_text and candidates else (0.0 for _ in candidates)
	folded_query = query.casefold()
	matches = []
	for candidate, score in zip(candidates, scores, strict=True):
		symbol = candidate[0]
		if pattern is not None:
			rank = (0, 0.0) if pattern.search(symbol.name) or pattern.search(symbol.qualified_name) else None
		elif mode == "exact":
			rank = _name_rank(query, symbol.name, symbol.qualified_name, fuzzy=False)
		else:
			rank = _name_rank(
				folded_query, symbol.name.casefold(), symbol.qualified_name.casefold(), fuzzy=not use_text
			)
			if rank is None and score > 0:
				rank = (2, -score)
		if rank is not None:
			matches.append((rank, symbol.path, symbol.start_line, candidate))
	matches.sort(key=lambda match: match[:3])
	return [candidate for _, _, _, candidate in matches]


def _git(directory: Path, *args: str) -> bytes:
	result = subprocess.run(["git", "-C", str(directory), *args], capture_output=True, check=False)
	if result.returncode:
		raise ValueError(os.fsdecode(result.stderr).strip())
	return result.stdout


def _positive_int(value: str) -> int:
	number = int(value)
	if number < 1:
		raise argparse.ArgumentTypeError("expected a positive integer")
	return number


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""Ranking modes
  fuzzy   Use for partial identifiers, abbreviations, and small typos.
          Short names rank by substring, ordered characters, then spelling
          similarity. Partial qualified-name matches follow.
  bm25    Use for keywords in identifiers and docstrings. Rare terms carry
          more weight, repeated terms saturate, and length is normalized.
  vector  Use for several keywords. Rank by cosine similarity of sparse
          TF-IDF word vectors built from identifiers and docstrings.

All rankers put an exact qualified name first, then exact short names
with shorter enclosing names first. BM25 and vector compute term statistics
after path and kind filtering. Ties sort by path and starting line.
--exact uses case-sensitive name ranking. --regex sorts by path and line.
An explicit --ranker applies only to ordinary searches.""",
	)
	parser.add_argument("query", help="symbol name or search pattern")
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument(
		"-r", "--regex", action="store_const", const="regex", dest="mode", help="Python regular expression"
	)
	mode.add_argument(
		"-x", "--exact", action="store_const", const="exact", dest="mode", help="case-sensitive exact name"
	)
	parser.add_argument("--ranker", choices=("fuzzy", "bm25", "vector"), help="ranking method (default: fuzzy)")
	parser.add_argument("-k", "--topk", "--top-k", type=_positive_int, help="limit candidate count")
	parser.add_argument("--root", type=Path, default=Path.cwd(), help="start Git root discovery here")
	parser.add_argument(
		"--path", action="append", default=[], help="restrict to a repository-relative file or directory"
	)
	parser.add_argument("--kind", choices=("function", "method", "class"), help="restrict to a definition kind")
	output = parser.add_mutually_exclusive_group()
	output.add_argument("--show", type=_positive_int, metavar="N", help="print only the source for candidate N")
	output.add_argument("--json", action="store_true", help="emit candidates as JSON")
	args = parser.parse_args(argv)
	if not args.query.strip():
		parser.error("query must not be blank")
	if args.mode and args.ranker:
		parser.error("--ranker applies only to ordinary searches")
	ranker = args.ranker or "fuzzy"
	try:
		pattern = re.compile(args.query) if args.mode == "regex" else None
		root = Path(os.fsdecode(_git(args.root, "rev-parse", "--show-toplevel")).removesuffix("\n"))
		paths = _git(
			root,
			"--literal-pathspecs",
			"ls-files",
			"-z",
			"--cached",
			"--others",
			"--exclude-standard",
			"--",
			*args.path,
		)
	except (OSError, ValueError, re.error) as error:
		parser.error(str(error))

	candidates: list[Candidate] = []
	for relative_path in dict.fromkeys(os.fsdecode(paths).split("\0")):
		path = root / relative_path
		if path.suffix not in (".py", ".pyi") or path.is_symlink():
			continue
		try:
			with tokenize.open(path) as file:
				source = file.read()
			for symbol, docstring in _symbols(relative_path, source, include_docstrings=ranker != "fuzzy"):
				if args.kind and symbol.kind != args.kind:
					continue
				candidates.append((symbol, docstring, source if args.show else ""))
		except FileNotFoundError:
			continue  # A tracked file can be deleted from the working tree.
		except (OSError, SyntaxError, UnicodeError, LookupError, tokenize.TokenError) as error:
			print(f"symbols: {relative_path}: {error}", file=sys.stderr)

	matches = _rank_candidates(args.query, candidates, args.mode, pattern, ranker)[: args.topk]
	if args.show:
		if args.show > len(matches):
			print(f"symbols: candidate {args.show} is unavailable ({len(matches)} matches)", file=sys.stderr)
			return 1
		symbol, _, source = matches[args.show - 1]
		lines = source.split("\n")
		print("\n".join(lines[symbol.start_line - 1 : symbol.end_line]))
	elif args.json:
		print(json.dumps([dict(asdict(symbol), line_count=symbol.line_count) for symbol, _, _ in matches], indent=2))
	else:
		for number, (symbol, _, _) in enumerate(matches, 1):
			location = f"{symbol.path}:{symbol.start_line}-{symbol.end_line}"
			print(f"{number:>3}  {symbol.qualified_name}  {location}  {symbol.line_count} lines  [{symbol.kind}]")
	return 0 if matches else 1


if __name__ == "__main__":
	raise SystemExit(main())
