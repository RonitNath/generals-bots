# generals-bots

`generals-bots` is a JAX-based Generals-style simulator that we are using as the base for a local bot competition setup.

The practical goal of this repo is:
- run the authoritative game server on one machine,
- connect two laptops over LAN,
- let each laptop run its own agent,
- and watch the matches on the server display.

It still contains the original high-performance simulator core, but the main workflow in this repo is now local head-to-head bot development and competition.

## What This Repo Is For

This repo supports three related workflows:

1. Build and test agents against the local JAX simulator.
2. Run LAN matches where a central server hosts the game and two remote clients submit actions.
3. Use the same agent interface for experiments, benchmarks, and future RL training.

The server is authoritative. Clients do not simulate the game independently. They receive observations, choose actions, and send those actions back to the server.

## Repo Layout

The core modules are:

- `generals/core/`: game rules, state, observations, actions, rewards, grid generation
- `generals/agents/`: agent interface, built-in baseline agents, agent loading utilities
- `generals/lan/`: TCP server/client protocol for remote bot play over LAN
- `generals/remote/`: live generals.io integration
- `generals/gui/`: local pygame-based visualization/debugging
- `examples/`: runnable scripts for local use
- `tests/`: core correctness and performance tests

Important entry points:

- `examples/lan_server.py`
- `examples/lan_client.py`
- `examples/custom_agent.py`

## Installation

This repo uses `uv`.

```bash
git clone https://github.com/RonitNath/generals-bots.git
cd generals-bots
uv sync --extra dev
```

Useful commands:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
```

There is also a small `Makefile` wrapper:

```bash
make test
make lan_server
make lan_client
```

## Core Concepts

### Environment API

The public environment is `GeneralsEnv`.

```python
from generals import GeneralsEnv

env = GeneralsEnv(grid_dims=(10, 10), truncation=500)
pool, state = env.reset(key)
timestep, state = env.step(state, actions, pool)
```

Notes:

- `reset()` returns `(pool, state)`, not just a single state.
- `pool` is a pre-generated state pool used for cheap auto-resets.
- `step()` requires the pool: `env.step(state, actions, pool)`.

### Observation

Agents receive an `Observation` containing:

- visible armies
- owned / opponent / neutral cells
- mountains, cities, generals
- fog masks
- scalar stats such as total army, land count, and timestep

### Action Format

Actions are integer arrays:

```text
[pass, row, col, direction, split]
```

Where:

- `pass`: `1` means do nothing
- `row`, `col`: source cell
- `direction`: `0=up`, `1=down`, `2=left`, `3=right`
- `split`: `1` means send half, `0` means send all but one

## LAN Competition Workflow

This is the main workflow this repo is being shaped around.

### Topology

- One machine runs the server.
- That machine owns the game state.
- Two laptops connect as clients.
- Each laptop runs its own bot implementation.

### Start The Server

Run this on the host machine:

```bash
uv run python examples/lan_server.py --host 0.0.0.0 --port 5555 --grid 15 --games 10
```

Common flags:

- `--host`: interface to bind to
- `--port`: TCP port for clients
- `--grid`: square grid size
- `--truncation`: max turns before draw
- `--timeout`: action timeout per player
- `--fps`: local visualization tick rate
- `--games`: number of games in the series

### Connect A Client

Run this on each laptop:

```bash
uv run python examples/lan_client.py --host <server-lan-ip> --port 5555 --agent expander --name AliceBot
```

Built-in options:

- `--agent expander`
- `--agent random`

Each client receives:

- game start messages
- per-turn observations
- final results and running match score

The server rotates player sides between games for fairness.

## Writing Your Own Agent

The shared LAN client script supports loading custom agents directly, so each person can work on their own agent code without modifying shared infrastructure.

### Agent Interface

Agents subclass `generals.agents.Agent` and implement:

```python
def act(self, observation, key) -> jnp.ndarray:
    ...
```

The method must return an action in the standard 5-integer format.

### Custom Agent Factory

Your module should expose either:

- an `Agent` subclass, or
- a factory function returning an `Agent`

The simplest pattern is:

```python
from generals.agents import Agent

class MyAgent(Agent):
    def act(self, observation, key):
        ...

def make_agent(name: str) -> Agent:
    return MyAgent(id=name)
```

See `examples/custom_agent.py` for a concrete template.

### Run A Custom Agent

You can load from either a Python module or a file path.

Module form:

```bash
uv run python examples/lan_client.py \
  --host <server-lan-ip> \
  --agent-custom my_package.my_agent:make_agent \
  --name AliceBot
```

File form:

```bash
uv run python examples/lan_client.py \
  --host <server-lan-ip> \
  --agent-custom examples/custom_agent.py:make_agent \
  --name AliceBot
```

The `--agent-custom` target must be one of:

- `python.module:factory`
- `/path/to/file.py:factory`

## Minimal Local Example

If you just want to run the simulator locally in one process:

```python
import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv, get_observation
from generals.agents import ExpanderAgent, RandomAgent

env = GeneralsEnv(grid_dims=(10, 10), truncation=500)
agent_0 = RandomAgent()
agent_1 = ExpanderAgent()

key = jrandom.PRNGKey(42)
pool, state = env.reset(key)

done = False
while not done:
    obs_0 = get_observation(state, 0)
    obs_1 = get_observation(state, 1)

    key, k1, k2 = jrandom.split(key, 3)
    actions = jnp.stack([
        agent_0.act(obs_0, k1),
        agent_1.act(obs_1, k2),
    ])

    timestep, state = env.step(state, actions, pool)
    done = bool(timestep.terminated or timestep.truncated)
```

## Tests

Current test coverage includes:

- `tests/test_game_jax.py`: core game mechanics and JIT behavior
- `tests/test_grid_generation_performance.py`: grid generation validity and performance
- `tests/test_performance.py`: vectorized environment benchmark path

Run everything with:

```bash
uv run --extra dev pytest
```

## Current State Of The Project

This repo is in active transition from “RL simulator package” toward “shared LAN bot arena.”

That means:

- the JAX simulator core is the stable foundation,
- LAN workflow is being made first-class,
- spectator/UI work may evolve separately from the core server/client path,
- and the recommended collaboration pattern is to keep platform work separate from agent-strategy work.

## Recommended Collaboration Model

If two people are building agents against this repo:

1. Keep the shared platform code in the main repo.
2. Let each person keep their own agent module or branch.
3. Use `--agent-custom` so shared scripts stay unchanged.
4. Run `uv run --extra dev pytest` before merging platform changes.
5. Treat server/protocol changes as shared infrastructure and agent logic as independent work.

## Live generals.io Support

The repo also still includes live generals.io integration through `generals.remote`, but that is not the primary workflow for the LAN competition setup.

Example:

```bash
uv run python examples/client_example.py --user_id <user-id> --lobby_id <lobby-id>
```

## License

MIT
