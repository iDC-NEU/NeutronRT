# Profile-based coding sessions

The wrapper starts a persistent Codex CLI session using the named configuration profile.

```bash
python -m scripts.coding.run task.md --profile lab --cwd /tmp/isolated-checkout --output logs/coding/unique-run
```

An optional terminal callback queues one bounded completion message to an explicit
parent thread and records `callback.stdout`, `callback.stderr`, and
`callback-receipt.json` beside the session state:

```bash
python -m scripts.coding.run task.md --profile lab --output logs/coding/unique-run --notify-thread PARENT_THREAD_ID
```

The receipt records the queue exit status and warnings. The wrapper preserves
the task exit code independently of the callback and rejects self-notification.

The wrapper saves the thread ID in `session.json` and streams JSON events for
up to one hour. The `lab` profile allows two concurrent sessions launched
through this wrapper.

Send instructions to the persistent session using its recorded ID:

```bash
codex queue --thread SESSION_ID --message "Additional instructions"
```

Queued messages use the running session's model. To continue a completed turn,
pass `--resume SESSION_ID` with the original profile and a fresh output directory.

Official references:
- https://learn.chatgpt.com/docs/non-interactive-mode
- https://learn.chatgpt.com/docs/config-file/config-advanced
- https://learn.chatgpt.com/docs/cli/reference
