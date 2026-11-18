# Profile-based coding sessions

A model override does not load a configuration profile. Use a persistent CLI session:

```bash
/home/hdd/lvhr/miniconda/envs/graph/bin/python -m scripts.coding.run task.md --profile lab --cwd /tmp/isolated-checkout --output logs/coding/unique-run
```

An optional terminal callback queues one bounded completion message to an explicit
parent thread and records `callback.stdout`, `callback.stderr`, and
`callback-receipt.json` beside the session state:

```bash
/home/hdd/lvhr/miniconda/envs/graph/bin/python -m scripts.coding.run task.md --profile lab --output logs/coding/unique-run --notify-thread PARENT_THREAD_ID
```

The receipt records queue exit status and warnings only; it does not claim that a
queued message resumed or woke the parent session. Callback failure never replaces
the original task exit code, and a session never notifies itself.

Launch that command with a short initial tool yield, without shell `&`. The wrapper consumes the JSON event stream, saves the thread ID in `session.json`, and returns when the turn completes. Terminal exit alone does not wake an idle Codex session; a completion wrapper must queue a bounded message to the owning thread with `codex queue --resume`. Do not repeatedly poll logs. Each run is bounded to one hour. The wrapper permits at most two concurrent lab invocations; count any separately launched lab sessions too.

Send instructions to the persistent session using its recorded ID:

```bash
codex queue --thread SESSION_ID --message "Additional instructions"
```

Queue uses the session's model. Do not add profile overrides to queue when talking to the running local daemon. After completion, continue with the wrapper's `--resume SESSION_ID` and a NEW output directory, retaining `--profile lab`. Do not use `--ephemeral`: it prevents persistent resume/queue discovery. The CLI exec stream is turn-based; use the app-server protocol if a fully bidirectional event client is needed. No approval or sandbox bypass flags are used.

Official references:
- https://learn.chatgpt.com/docs/non-interactive-mode
- https://learn.chatgpt.com/docs/config-file/config-advanced
- https://learn.chatgpt.com/docs/cli/reference
