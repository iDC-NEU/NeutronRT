# NrtInc Component Ablations

`nrtinc-component-ablations-cpu.json` is a prepared CPU profile matrix. It defines
matched reference pairs for operator incrementality, pipeline, retained and
sparse aggregates, row refresh, refresh interval, dispatch, deferred publication,
admission limits, and concurrent compute. The final case exercises the supported
pipeline + locality/load dispatch + fixed-time deferred interaction.

The file is configuration only. It does not claim that any case has run or passed
correctness, resource, or performance admission. Before measuring, use one frozen
workload, checkpoint, graph, trace, query stream, environment, and hardware budget
for every case. Inspect each profile's effective switches, policy settings, and
counter deltas; requested flags alone do not establish that a component ran.

`worker_feature_cache` is intentionally excluded. The current RPC sends complete
feature rows before worker lookup, and its version/generation-certified miss-only
protocol is not implemented. `ProfileCase` fails closed if this switch is enabled.
Do not describe a post-transfer local copy cache as reduced network fetch.

The refresh-interval pair changes `delta_max_steps` (8 versus 32); it is a
refresh-policy sensitivity, not a forced-fallback test. Interpret `refreshes`,
`fallback_reason_counts`, and row-refresh counters from actual execution. A zero
counter means the fixed workload did not exercise that branch, not that the
component is ineffective.
