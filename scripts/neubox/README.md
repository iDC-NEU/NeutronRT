# Neu Box task inspection

Run the task inspector from the repository root.

```bash
python -m scripts.neubox.tasks --user YOUR_USER --status running --topk 10
python -m scripts.neubox.tasks --task-id ID --health --json
python -m scripts.neubox.tasks --task-id ID --watch --seconds 3600 --json
```

`--hash` is an alias for `--task-id`. Repeat the task option to watch several
jobs. JSON output preserves multiline commands. The monitor returns status 124
when its watch window expires. Scheduler jobs continue independently, and the
same task IDs can be used in a new watch.

## Session-owned watchdog and hourly timer

Use `watchdog` to discover all active tasks belonging to the experiment owner,
including new tasks submitted while it is running. Historical terminal tasks and
other users are excluded by default (`--user '*'` explicitly selects everyone).
Repeat `--host` for local and SSH schedulers:

```bash
python -m scripts.neubox.watchdog --host local --host GPU_HOST --user YOUR_USER \
  --output logs/task-waits/session-watchdog --seconds 3600 --interval 60
python -m scripts.neubox.timer --interval 3600 \
  --state logs/task-waits/session-watchdog/state.json \
  --output logs/task-waits/session-timer-001
```

Start the watchdog and timer as background processes in a Codex session.
They use `CODEX_THREAD_ID` and the matching `codex resume` process to identify
the owner. The Linux process start time distinguishes the owner from a reused
PID. Both monitors exit when the owner process exits. Neu Box jobs retain their
scheduler allocations independently.

The watchdog writes a status snapshot to `state.json`, scan history to
`events.jsonl`, and worker results and logs to `task-logs/`. It queues a
callback when all discovered tasks reach a terminal state. Scheduler errors
keep completion pending. Callback receipts record whether the queue accepted
the message.

The timer queues a health-check callback after one hour. Each monitor window
is bounded. Reuse the watchdog output directory to continue tracking the
same tasks and choose a new output directory for the next timer. A file lock
allows one watchdog per state directory.
