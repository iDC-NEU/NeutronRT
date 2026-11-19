# NeutronRT

Real-time GNN inference with versioned embeddings, graph updates and distributed
workers. Run the commands below from the repository root. Datasets, checkpoints,
logs and generated results are local inputs and are not included in this release.

## Install

Use Python 3.12 and matching PyTorch/DGL 2.4 builds. For CPU development:

```bash
conda create -n graph python=3.12 -y
conda activate graph
conda install 'dglteam/label/th24_cpu::dgl=2.4.0.th24.cpu=py312_0'
python -m pip install 'torch==2.4.0+cpu' --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev]' grpcio-tools==1.75.1
pre-commit install
```

For CUDA 12.4, install the corresponding backends before the project:

```bash
conda install 'dglteam/label/th24_cu124::dgl=2.4.0.th24.cu124'
python -m pip install 'torch==2.4.0+cu124' --index-url https://download.pytorch.org/whl/cu124
```

## Test

The quick check below uses small CPU logic tests and starts no server or GPU job:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/test_architecture.py tests/graph/test_methods.py \
  tests/evaluate/test_distributed.py -q
```

For the wider CPU logic suite, use the same environment variables with
`python -m pytest tests/ -m 'not host' -q`. The full suite includes process/host
checks; run it separately when the necessary resources are available.

```bash
ruff check lib/ benchmarks/ scripts/ tests/
ruff format --check lib/ benchmarks/ scripts/ tests/
mypy -p lib
python -m scripts.generate_inference_proto
```

The schema generator uses `grpcio-tools==1.75.1`. Native sources and the RPC schema
are checked as wheel resources by `tests/test_distribution.py`.

## Run

Use a prepared graph directory and a matching checkpoint:

```bash
python -m scripts.simple --data /path/to/arxiv/raw --workers cuda:0 \
  --checkpoint /path/to/sage-arxiv.pt --method NrtInc
python -m scripts.benchmark --address localhost:50051 --output /tmp/benchmark.json
```

Use `--method RTEC-NS --fanout 5` for sampling; any positive fanout applies at
every layer. CLI list options repeat once per value, such as `--workers cuda:0
--workers cuda:1`. Use each command's `--help` for its complete options.

Reusable profiling inputs are in `examples/profiles/`. Inspect a plan without
executing it:

```bash
python -m scripts.evaluate.profile --data /path/to/arxiv/raw \
  --checkpoint /path/to/sage-arxiv.pt --cases examples/profiles/default.json \
  --output /tmp/profile-plan --dry-run
```

`examples/profiles/deferred-cpu.json` provides exact execution and version windows
of 0, 1, 2, 4 and 8. A version window does not impose a numerical error budget.
Wait for every admitted update, including final deferred publication, before
stopping a measurement; unfinished service publication invalidates the profile.

Distributed trial reports include `workload_counts` for warmup, measured and total
requests, with per-request sizes. Edge counts refer to requested directed pairs;
deleting a pair removes all matching parallel edges, and an absent pair is a no-op.

### Completion and profile observability

Admitted updates retain optional server-monotonic lifecycle metadata: input and
publication versions, admission, executor start, staging, publication/completion,
and the server reply-preparation boundary. Queue wait, admission-side execution,
and deferred-publication wait are separate fields; these timestamps are not mixed
with client clocks. A client that needs receive time must record its own local
timestamp. Profile windows can attach an `UpdateAdmission` observer to reject
pending or failed admitted work at the measurement boundary.

Remote storage profiles sample RSS through a low-rate endpoint-only command. Reports
distinguish sampled RSS peaks from start/end RSS and end-of-window logical overlay
bytes, and retain sample counts, interval, errors, and sampler shutdown state.

### Remote mapped storage

Remote mapped storage uses a persistent base exported from the exact graph used by
the evaluator. After loading that graph in the `graph` environment, prepare a base
directory once:

```python
from pathlib import Path
from lib.storage.base import BaseTopology

BaseTopology.export(graph, Path("/data/neutronrt-base"))
```

Make an equivalent export available to each storage server and start each server
with its local `--base` path. Paths may differ between machines; the evaluator
and servers verify the graph content and dimensions. Pass the evaluator's local
path with `--topology mapped --topology-base /data/neutronrt-base` and repeat
`--storage-endpoint HOST:PORT` once per storage partition. The exported base is
reused and remains on disk when the evaluator exits.

## Layout

Dedicated standard and published-system baselines are documented in
[`benchmarks/README.md`](benchmarks/README.md). They execute separately from
NeutronRT; framework-hosted Full/NS/UER are mechanism ablations, not reproductions
of external systems. Source pins and comparison roles are in `benchmarks/sources.json`.

- `lib/`: models, graph/storage, runtime, RPC, communication and profiling.
- `scripts/`: launch, preparation, validation and evaluation commands.
- `tests/`: logic, protocol and host checks.
- `examples/profiles/`: portable reusable configurations.
- `benchmarks/`: baseline adapters and public source-revision pins.
