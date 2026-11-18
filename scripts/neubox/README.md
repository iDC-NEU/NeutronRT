# Neu Box task inspection

Use the frozen graph Python from the project root:

```bash
python -m scripts.neubox.tasks --user lhr --status running --topk 10
python -m scripts.neubox.tasks --task-id ID --health --json
python -m scripts.neubox.tasks --task-id ID --watch --seconds 3600 --json
```

`--hash` remains an alias for `--task-id`; repeat it to watch several known tasks. JSON parsing preserves multiline commands and handles both `queue` and `tasks` response keys. Service failures propagate, each CLI request has a timeout, and monitor expiry exits124 without cancelling remote work. Start long watches as harness background terminals; check health and reuse the same task IDs after expiry.

## Session-owned watchdog and hourly timer

Use `watchdog` to discover all active tasks belonging to the experiment owner,
including new tasks submitted while it is running. Historical terminal tasks and
other users are excluded by default (`--user '*'` explicitly selects everyone).
Repeat `--host` for local and SSH schedulers:

```bash
python -m scripts.neubox.watchdog --host local --host gpunode2 --user lhr \
  --output logs/task-waits/session-watchdog --seconds 3600 --interval 60
python -m scripts.neubox.timer --interval 3600 \
  --state logs/task-waits/session-watchdog/state.json \
  --output logs/task-waits/session-timer-001
```

Run these as two background tool terminals owned by the current Codex session.
Both infer `CODEX_THREAD_ID` and bind to the matching ancestor `codex resume`
process, including its Linux start-time token to detect PID reuse. They stop
when that session process exits; an idle turn does not stop them. An unknown
owner is an error, not permission to launch a detached monitor. The scripts
never submit, cancel, or signal neu-sbox jobs. Scheduler allocations and jobs
remain alive independently of Codex and its observers.

The watchdog checks status automatically, writes `state.json`, appends every
scan to `events.jsonl`, and saves full latest worker results/logs under
`task-logs/`. Scheduler errors remain explicit and prevent false completion.
When all discovered tasks are terminal it queues one session callback; failed
or canceled tasks count as done but not successful. Queue receipts and callback
output are saved before and after notification. Queue acceptance is recorded;
actual delivery is established when this session receives the message.

The separate timer queues an hourly health-check callback even if monitoring
has failed. Both processes are bounded to about one hour. On the timer callback,
read the saved logs, check observer/process health once, resume the watchdog
using the same output if needed, and start the next one-hour timer in a fresh
output directory. A file lock prevents duplicate watchdogs on the same state.
On a completion callback, inspect outcomes and submit any authorized next stage,
then resume the watchdog for that stage. Do not manually poll between callbacks.
