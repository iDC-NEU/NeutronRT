# Neu Box task inspection

Use the frozen graph Python from the project root:

```bash
python -m scripts.neubox.tasks --user lhr --status running --topk 10
python -m scripts.neubox.tasks --task-id ID --health --json
python -m scripts.neubox.tasks --task-id ID --watch --seconds 3600 --json
```

`--hash` remains an alias for `--task-id`; repeat it to watch several known tasks. JSON parsing preserves multiline commands and handles both `queue` and `tasks` response keys. Service failures propagate, each CLI request has a timeout, and monitor expiry exits124 without cancelling remote work. Start long watches as harness background terminals; check health and reuse the same task IDs after expiry.
