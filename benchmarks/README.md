# Dedicated baseline implementations

The standalone adapters run outside NeutronRT's planner, scheduler and
embedding service. They share recorded model inputs and evaluation utilities.
Benchmark seeds determine the recorded workloads. Graph preparation and
online serving use the graph and live inputs, while fresh oracle outputs
are computed for evaluation.

| Comparison | Execution source | Evaluation scope |
| --- | --- | --- |
| RTEC-FULL (DGL) | Official DGL 2.4.0 native full-neighbor sampler and layerwise inference; dedicated `dgl.py` / `streaming.py` runner | Mean GraphSAGE; exact full-neighbor reference; matched measurements remain cohort-specific |
| RTEC-NS5 (DGL) | Same official DGL source and runner, fanout 5 at every layer | Approximate sampling with its own numerical quality measurements |
| RTEC-UER | Pinned layerwise `Cache` adaptation over native DGL one-layer blocks | Exact full-neighbor reuse; lambdaGrapher-inspired, not official DGL UER or a reproduction of the full Knative/HAG system |
| InkStream | Pinned author release and its unchanged PyG patch in an isolated environment | Verification-only; unchanged mean-SAGE gate fails at final output, so no timing cohort is admitted |
| Ripple / RIPPLE++ | Pinned author repository and unchanged GC-mean processors | Stateful GC-mean compatibility replay is implemented; it is a separate operator cohort, not ordinary trained GraphSAGE; no matched performance cohort is accepted yet |
| Helios | Pinned Graph-Learn `dynamic_graph_service` source identified by the PPoPP 2025 paper | Explicitly unsupported for the current uniform-sampling cohort: timestamp-top-k is not the matched sampling law, and no full-system build/replay passed admission |
| NrtInc | This repository's incremental GraphSAGE implementation and the exact measured source snapshot | Incremental reference with a recorded runtime boundary and source fingerprint |

`sources.json` is the source/role registry. `sources.py` snapshots clean pinned
checkouts without changing algorithms. `inkstream.py` and the verification mode
of `ripple.py` provide numerical gates; `external_ripple_replay.py` executes the
author's Ripple processor for a stateful GC-mean compatibility replay. A fixture
checks numerical compatibility before any matched performance measurement.

The dedicated DGL entry point consumes an existing recording:

```bash
python -m scripts.evaluate.dgl --data /prepared/graph --workload /recording.pt \
  --checkpoint /matched/checkpoint.pt --method RTEC-Full --device cuda:0 \
  --output-device cpu --batch-size 8192 --threads 4 --output /new/result
```

Run GPU trials through neu-sbox using the declared hardware profile. External
artifacts use their pinned environments. For paired results, match the
operator equations, weights and update order. Report initialization, index
maintenance, and output publication according to the timing boundary, along
with sampling quality, supported mutations and device placement.

## Resident replay

`python -m scripts.evaluate.resident` replays the same `RecordedWorkload` on
one explicitly admitted CUDA device. `NrtInc` uses the reusable resident
GraphSAGE runtime in `lib/graph/resident_runtime.py`; Full, NS and UER use the
native DGL adapters with graph, features, model and output state on that same
device. The command requires `--capacity-bytes` and fails closed when admission
is not possible. It never constructs `ManagedRuntime`, RPC or distributed
workers. Reports include device placement, capacity, allocator peaks, exclusive
update phases and final independent CPU-oracle quality.

Framework-hosted ablations and dedicated adapters have separate backend and
cohort identities in their reports.

Run an external verification gate or the admitted Ripple GC-mean replay with an
explicit environment executable. Helios is accepted as a source-status entry but
returns `unsupported` until its exact sampler/system semantics pass admission:

```bash
python -m scripts.evaluate.external --artifact ripple --source /clean/pinned/ripple \
  --python /isolated/environment/bin/python --device cpu --timeout 300 \
  --output /new/verification-result
```

Reports contain source/dependency identity and structured status. Ripple replay
uses an explicit zero-self, zero-bias GraphSAGE compatibility checkpoint; ordinary
trained GraphSAGE checkpoints are rejected rather than converted by discarding
their self terms. Its standalone timings are not comparable to the standard
GraphSAGE cohort, and a matched cohort still requires correctness/resource gates.
