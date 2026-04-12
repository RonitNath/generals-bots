# Agent Development

## Agent Interface

Every agent subclasses `generals.agents.Agent` and implements:

```python
def act(self, observation, key) -> jnp.ndarray:
    ...
```

Return action format:

```text
[pass, row, col, direction, split]
```

Where:

- `pass`: `1` means do nothing
- `row`, `col`: source tile
- `direction`: `0=up`, `1=down`, `2=left`, `3=right`
- `split`: `1` means half move, `0` means all-but-one

Optional debugging hook:

```python
def get_debug_snapshot(self) -> dict | None:
    ...
```

Use this for structured explanations of why the agent chose a move.

## Custom Agent Loading

Use the shared LAN client without editing it:

```bash
uv run python examples/lan_client.py \
  --host <server-ip> \
  --agent-custom path/to/agent.py:make_agent \
  --name MyBot
```

Supported target forms:

- `python.module:factory`
- `/path/to/file.py:factory`

## Built-In Heuristic Agents

Current heuristic roster:

- `material`: material advantage, city timing, favorable trades
- `scout`: early fog exploration and pressure toward inferred enemy position
- `backdoor`: deep incursions and indirect attack routes
- `defense`: defensive shell and counterattack bias
- `surround`: multi-angle pressure and flank shaping

These are intended as sparring partners and debugging references, not “final” strong bots.

## Recommended Loop

1. Start from `examples/custom_agent.py` or a copy of one heuristic agent.
2. Add `get_debug_snapshot()` early.
3. Run short logged matches first.
4. Inspect anomalies and keyframes before tuning blindly.
5. Only then move to long runs or LAN series.
