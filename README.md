# NeutronRT

Real-time GNN inference with versioned embeddings, graph updates and distributed
workers. Run the commands below from the repository root. Datasets, checkpoints,
logs and generated results are local inputs and are not included in this release.

## Install

Use Python 3.12 and matching PyTorch/DGL 2.4 builds. For CPU development:

```bash
conda create -n graph python=3.12 -y
conda activate graph
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/repo.html
pip install -e '.[dev]' grpcio-tools==1.75.1
pre-commit install
```

For CUDA 12.4, install the corresponding backends before the project:

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
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

## Layout

- `lib/`: models, graph/storage, runtime, RPC, communication and profiling.
- `scripts/`: launch, preparation, validation and evaluation commands.
- `tests/`: logic, protocol and host checks.
- `examples/profiles/`: portable reusable configurations.
- `benchmarks/`: baseline adapters and public source-revision pins.
