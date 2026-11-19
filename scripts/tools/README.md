# Project navigation tools

## Symbol navigator

Run from the repository root with Python 3.11 or newer and Git.

```bash
# Rank the five best matches for a partial name.
python -m scripts.tools.symbols incremental -k 5

# Print only the source of the first candidate from that search.
python -m scripts.tools.symbols incremental -k 5 --show 1

# Search qualified names with a regular expression inside lib/models.
python -m scripts.tools.symbols --regex '^IncrementalLayers\.' --path lib/models --kind method

# Retrieve a specific method by its qualified name.
python -m scripts.tools.symbols --exact IncrementalLayers.apply --show 1

# Find likely callers across the repository for the selected definition.
python -m scripts.tools.symbols --exact _fresh --path scripts/evaluate/resident.py --callers 1

# Preview a selected definition and show up to five likely callers.
python -m scripts.tools.symbols --exact _fresh --path scripts/evaluate/resident.py --pick 1 --lines 12 --limit 5

# Export candidates for another tool.
python -m scripts.tools.symbols incremental --topk 5 --json

# Search names and docstrings by keyword.
python -m scripts.tools.symbols 'incremental message cache' --ranker bm25 -k 5

# Score keyword overlap with sparse TF-IDF vectors.
python -m scripts.tools.symbols 'incremental message cache' --ranker vector -k 5
```

Each result has a candidate number, qualified name, repository-relative path,
inclusive line range, line count, and definition kind. Extraction preserves the
original indentation and includes decorators and the definition body.
Use `--show N` with the same query and filters to select candidate N.
Use `--callers N` with the same query and filters to list likely call sites
across the repository. Caller lookup uses static matches for imports, names
and recognizable receivers. Dynamic dispatch may require a follow-up search.

All three rankers are case-insensitive and prioritize exact qualified names,
then exact short names with shorter enclosing names first. For example, a
top-level `service` ranks ahead of `Worker.service` for the query `service`.

| Ranker | When to use it | How it scores other matches |
| --- | --- | --- |
| `fuzzy` (default) | Partial names, abbreviations, and typos | Substrings, ordered characters, then spelling similarity. Short-name matches precede partial qualified-name matches. |
| `bm25` | Keywords you expect in names or docstrings | Rare terms carry more weight. Repeated terms saturate and document length is normalized. |
| `vector` | Multiword queries where term overlap and document focus matter | Cosine similarity on sparse TF-IDF word vectors built from qualified names and docstrings. |

BM25 and vector search split identifiers such as `RPCService` and
`incremental_cache` into words and compute term statistics after path and kind
filtering. Search with `--help` for ranking and tie-break rules. Regex and
exact search are case-sensitive by default. Use `(?i)` in a regex for
case-insensitive matching. Path filters can be repeated.

The navigator reads tracked and unignored untracked Python files (`.py` and
`.pyi`) from the working tree on each query. It supports functions, async
functions, methods, nested definitions, and classes. Files with read or syntax
errors produce diagnostics on stderr. Exit status is 0 for results, 1 for
no matches or an unavailable candidate, and 2 for invalid arguments or Git
discovery errors. To run from elsewhere, invoke the script by path and pass
`--root /path/to/NeutronRT`.
