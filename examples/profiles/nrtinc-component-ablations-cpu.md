# NrtInc Component Ablations

`nrtinc-component-ablations-cpu.json` is a prepared CPU profile matrix. It defines
matched reference pairs for operator incrementality, pipeline, retained and
sparse aggregates, row refresh, refresh interval, dispatch, deferred publication,
admission limits, and concurrent compute. The final case exercises the supported
pipeline + locality/load dispatch + fixed-time deferred interaction.

For a matched comparison, use the same graph, checkpoint, updates, queries,
software environment and hardware budget in every case. Profile reports
record the effective switches, policy settings and counter deltas that
identify which paths the workload exercised.

The current RPC transfers complete feature rows before worker lookup.
`ProfileCase` rejects the `worker_feature_cache` option until the transfer
protocol can verify cache hits and send only missing rows.

The refresh-interval pair compares `delta_max_steps` values of 8 and 32.
The `refreshes`, `fallback_reason_counts`, and row-refresh counters show
how often the workload triggered each path.
