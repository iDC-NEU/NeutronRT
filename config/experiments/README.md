# Experiment manifests

Manifests specify the command, software environment, input paths, scheduler
resources, and time limits for a run. Review input paths before using a
manifest on another host.

## Dataset preparation

`datasets.json` groups full graphs into small (arXiv, Products, Reddit) and large
(Papers100M, Twitter, Friendster) datasets. `prepare-topology-available.json`
exports mapped CSR and CSC bases. The Products export already includes reverse
edges. `prepare-topology-papers100m.json` exports the base after offline
Papers100M preparation.

`scripts.validate.dataset_inventory` checks paths, manifests and file sizes.
`validate-dataset-inventory.json` runs the full checksum audit through neu-sbox.
`scripts.download_datasets --help` lists dataset preparation commands.

The continuation manifests resume dataset preparation after prerequisite tasks
finish. `six-dataset-preparation-continuation.json` checks the six-dataset
inventory and prepares Papers100M from the local archive. The Papers100M-only
workflow uses `papers100m-continuation.json`. Both plans use persistent state
to track submissions. Missing raw files stop preparation.

## Submission and results

With the experiment Python environment active, submit a manifest from the
repository root:

```bash
python -m scripts.evaluate.run_manifest config/experiments/arxiv-deferred-cpu.json \
  --output logs/experiments/arxiv-deferred-cpu-run-01 --submit
```

`--submit` takes resource limits from `hardware.json` and freezes the source,
configuration and declared input digests under `logs/submissions/`. The worker
checks that snapshot and the actual allocation before starting. CPU allocations
are scheduler quotas. GPU allocations are resolved to NVIDIA UUIDs and used to
set the child process's `CUDA_VISIBLE_DEVICES`.

Each run uses a fresh output path. The submission journal reserves that path
before contacting the scheduler and records the task ID in `logs/task-ledger/`.
Check the journal before retrying an interrupted submission.

Trial manifests can supply a child-written `phase_file` with the method,
phase and cohort identity. Timeout records preserve the observed phase.
The final renderer requires known phase and cohort metadata for censored
trials.

## Comparison plans

- `common-baselines-gpu.json` runs 40 trials in one GPU allocation. arXiv and
  Products each have four methods and five paired repetitions. Each trial
  has a 7,200-second wall limit and a 900-second startup limit. The startup
  deadline ends when the managed runtime starts. Products uses seed-42
  model weights and checks sampled numerical agreement.
- `arxiv-full-test-quality-cpu.json` runs six cases with five repetitions
  each against a trained checkpoint and all 48,603 official arXiv test rows.
  The report records test coverage and publication drain.
