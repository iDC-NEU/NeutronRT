# Fixed experiment manifests

`prepare-topology-available.json` exports persistent mapped CSR/CSC bases for the
five available datasets now; task `bade4642e8e1`, fixed16 CPU /256 G /0 GPU.
Products uses reverse-edge expansion as in the current comparison. New inference
inputs must not reverse that serialized graph a second time. Papers100M has a
separate `prepare-topology-papers100m.json` manifest after offline preparation.

Dataset tiers are fixed in `datasets.json`: **small = arXiv, Products, Reddit**;
**large = Papers100M, Twitter, Friendster**. Every entry denotes a full graph.
`scripts.validate.dataset_inventory` checks prepared paths/manifests/file sizes
without loading graphs; `--full` performs checksum reads through neu-sbox using
`validate-dataset-inventory.json`. Metadata presence is distinct from full
integrity, model correctness and runtime memory feasibility.

`six-dataset-preparation-continuation.json` is the current preparation chain:
wait for the existing Papers100M download and historical GPU cohort, then submit
`prepare-six-datasets.json` once (16 CPU /256 G RAM /0 GPU). That task prepares
Papers100M from the completed ZIP offline, reuses the other five datasets and
audits full file checksums. It supersedes the pending Papers100M-only continuation;
do not launch both. Keep the same continuation state and submitted task ID.

`hardware.json` freezes aggregate scheduler CPU/RAM/GPU requests. CPU quotas are not physical-core affinity. Inventory describes available hardware, not reserved capacity. Device UUIDs and transport selection are captured per trial; do not pool A5000 and A6000 cohorts.

Submissions made with `run_manifest --submit` now copy code (including intended untracked source), configuration and declared input hashes into `logs/submissions/`. The worker validates that snapshot before running; queued work never rereads the editable original manifest. The output name is reserved before scheduler submission to prevent duplicate jobs after interruption. Inspect `logs/task-ledger/` before retrying ambiguous submissions.

`arxiv-deferred-cpu.json` fixes all CLI arguments, checkpoint, environment and wall timeout for the first resumed experiment. Submit from an explicit project directory (the scheduler does not inherit the client working directory):

```bash
neu-sbox --json submit --cpu 16 --mem 32 --device-num 0 -- bash -lc 'cd /home/lhr/NeutronRT && exec /home/hdd/lvhr/miniconda/envs/graph/bin/python -m scripts.evaluate.run_manifest config/experiments/arxiv-deferred-cpu.json --output /home/lhr/NeutronRT/logs/experiments/NEW-UNIQUE-RUN'
```

Save the returned task ID before `neu-sbox wait ID`. Never resubmit an active task. Change output paths for every new run; keep source/runtime unchanged while a cohort runs. Algorithm or hardware changes create a new versioned cohort. Existing logs and generated manuscript artifacts must remain intact.

General dataset tool: `python -m scripts.download_datasets --help`. Peer checkout sync: `python -m scripts.peer.sync --help`. All expensive preparation must run in its declared neu-sbox profile. Existing dataset paths containing `revision` are preserved to avoid moving live downloads.

Use `run_manifest --submit` to derive scheduler flags directly from the frozen hardware profile and save the returned task ID in `logs/task-ledger/`. Example: `graph/bin/python -m scripts.evaluate.run_manifest config/experiments/arxiv-deferred-cpu.json --output logs/experiments/UNIQUE-RUN --submit`. This performs a real submission.

`papers100m-continuation.json` waits for the existing transfer and CPU cohort to terminate before submitting `papers100m-prepare.json`. The continuation monitor uses persistent state and a one-hour maximum window; resume the same state after a timeout, never submit a second preparation job. Missing raw files stop the continuation rather than triggering a new download.

For final-evidence censoring, supply a `phase_file` written atomically by the child (e.g. `{output}/phase.json`) with its actual `method`, `phase`, and complete `cohort_identity`. The wrapper preserves unknown phases and does not infer RTEC-Full from a timeout. A non-timeout exit137 remains a failed run. Final plotting rejects unknown/incomplete censoring; legacy inspection requires its explicit diagnostic flag.

GPU workers resolve their actual inherited sandbox and allocated device minors, map them to NVIDIA UUIDs, and set child `CUDA_VISIBLE_DEVICES` explicitly. Do not assume the scheduler supplies that variable or use physical ordinal0 for an automatically allocated GPU. The frozen CPU/RAM/GPU envelope is checked against the actual allocation before child work starts.

`common-baselines-gpu.json` launches the frozen40-trial plan in `common-baselines-gpu-trials.json`: arXiv then Products, all four methods and five paired repetitions, serially in one16CPU/64G/1GPU allocation. Each full-graph trial has its own7200s wall and900s startup deadline; failures retain logs/status and do not suppress later trials. All allocated GPU UUIDs are recorded, and the same single GPU remains allocated throughout. Products uses random seed42 weights and sampled correctness, not task accuracy. The900s startup boundary ends when ManagedRuntime enters successfully, before model-state initialization; the separate600s request timeout applies thereafter. New outputs only; never resubmit an existing output.

`arxiv-full-test-quality-cpu.json` is a separate prepared quality cohort: six cases × five repetitions, trained checkpoint, and all 48,603 official test rows. It uses the fixed16CPU/32G/0GPU profile and a14,400-second cap. It is not a replacement for the completed128-query cohort; require full-test coverage and completed publication/drain in its report. Submit only after the final combined code regression and the active GPU cohort finish.
