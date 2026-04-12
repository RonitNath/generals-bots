# Match Analysis

## Logged Match Runner

Use:

```bash
uv run python examples/run_logged_match.py \
  --agent-a material \
  --agent-b backdoor \
  --output logs/material-vs-backdoor \
  --gui
```

## Artifacts

A match directory contains:

- `metadata.json`
- `turns.jsonl`
- `events.jsonl`
- `summary.json`
- `keyframes/`

## What Each File Is For

### `metadata.json`

Contains:

- players
- seed
- env config
- map fairness report

Use this first. If the map is flagged as biased, do not over-interpret the match.

### `turns.jsonl`

One record per turn, including:

- pre/post state stats
- chosen actions
- agent debug snapshots
- anomaly detections
- anomaly score for the turn

### `events.jsonl`

Sparse event stream only:

- city captures
- general reveals
- land swings
- anomaly events
- keyframe writes

This is the fastest file to scan for “what happened.”

### `summary.json`

Contains:

- winner
- turn count
- cumulative anomaly scores
- anomaly counts by type
- worst turns
- keyframe/event totals

## Keyframes

Each keyframe is stored as:

- JSON state snapshot
- PNG render

Capture policy:

- periodic
- city capture
- general reveal
- land swing
- anomaly
- game end

Open one with:

```bash
uv run python examples/view_keyframe.py --input logs/.../keyframes/0025__periodic.json
```

## Debugging Sequence

When a match looks wrong:

1. Read `metadata.json` fairness warnings.
2. Read `summary.json` worst turns.
3. Inspect `events.jsonl` around the suspicious window.
4. Open the matching keyframes.
5. Read the agent debug snapshot for that turn from `turns.jsonl`.
