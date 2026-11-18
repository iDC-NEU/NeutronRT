# Dedicated baseline implementations

Baselines execute outside NeutronRT's planner, scheduler and embedding service.
Share reproducible inputs and evaluation utilities; do not execute a baseline by
changing a NeutronRT method switch and present that result as an external system.

Fixed benchmark seeds enable repeatable comparisons. Public graph preparation and
serving must work without those seeds or future traces. Seeded recordings and
oracle outputs are harness artifacts, not inputs to public preprocessing, cache
warm-up or online planning. Reuse prepared graph storage across unrelated traces.

| Comparison | Execution source | Current scope |
| --- | --- | --- |
| RTEC-FULL (DGL) | Official DGL 2.4.0 native full-neighbor sampler and layerwise inference; dedicated `dgl.py` / `streaming.py` runner | Mean GraphSAGE; exact full-neighbor reference; matched measurements remain cohort-specific |
| RTEC-NS5 (DGL) | Same official DGL source and runner, fanout 5 at every layer | Approximate sampling; report measured error separately; do not merge with Full/UER quality claims |
| RTEC-UER | User's pinned layerwise `Cache` adaptation over native DGL one-layer blocks | Exact full-neighbor reuse; lambdaGrapher-inspired, not official DGL UER or a reproduction of the full Knative/HAG system |
| InkStream | Pinned author release and its unchanged PyG patch in an isolated environment | Verification-only; unchanged mean-SAGE gate fails at final output, so no timing cohort is admitted |
| Ripple / RIPPLE++ | Pinned author repository and unchanged GC-mean processors | Stateful GC-mean compatibility replay is implemented; it is a separate operator cohort, not ordinary trained GraphSAGE; no matched performance cohort is accepted yet |
| Helios | Pinned Graph-Learn `dynamic_graph_service` source identified by the PPoPP 2025 paper | Explicitly unsupported for the current uniform-sampling cohort: timestamp-top-k is not the matched sampling law, and no full-system build/replay passed admission |
| NrtInc | This repository's incremental GraphSAGE implementation and the exact measured source snapshot | Main system; report runtime boundary and source fingerprint, not as an author-artifact reproduction |

`sources.json` is the source/role registry. `sources.py` snapshots clean pinned
checkouts without changing algorithms. `inkstream.py` and the verification mode
of `ripple.py` provide numerical gates; `external_ripple_replay.py` executes the
author's Ripple processor for a stateful GC-mean compatibility replay. A fixture
gate is not a matched performance cohort. An unsuccessful numerical gate remains
visible and cannot be replaced by timings from a silently corrected implementation.

The dedicated DGL entry point consumes an existing recording:

```bash
python -m scripts.evaluate.dgl --data /prepared/graph --workload /recording.pt \
  --checkpoint /matched/checkpoint.pt --method RTEC-Full --device cuda:0 \
  --output-device cpu --batch-size 8192 --threads 4 --output /new/result
```

This is a command template, not a submitted experiment or frozen cohort. GPU and
substantial work must use neu-sbox and the repository's fixed hardware profiles.
External artifacts use their own pinned environments without modifying `graph`.
Include initialization, state/index maintenance and final output visibility in
the declared timing boundaries. Report sampling quality, unsupported mutations,
parallel-edge semantics and CPU/GPU placement. Match actual operator equations,
weights and update order before producing paired speedups.

## Resident replay

`python -m scripts.evaluate.resident` replays the same `RecordedWorkload` on
one explicitly admitted CUDA device. `NrtInc` uses the reusable resident
GraphSAGE runtime in `lib/graph/resident_runtime.py`; Full, NS and UER use the
native DGL adapters with graph, features, model and output state on that same
device. The command requires `--capacity-bytes` and fails closed when admission
is not possible. It never constructs `ManagedRuntime`, RPC or distributed
workers. Reports include device placement, capacity, allocator peaks, exclusive
update phases and final independent CPU-oracle quality.

Historical framework method IDs and outputs remain valid as historical ablation
data. New dedicated results require new backend/cohort identities and output paths.

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
