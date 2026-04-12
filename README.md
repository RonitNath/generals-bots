<div align="center">

![Gameplay GIF](https://raw.githubusercontent.com/strakam/generals-bots/master/generals/assets/gifs/wider_gameplay.gif)

## **Generals.io Bots**

[Installation](#-installation) • [Getting Started](#-getting-started) • [Environment](#-environment) • [Deployment](#-deployment)
</div>

A high-performance JAX-based simulator for [generals.io](https://generals.io), designed for reinforcement learning research.

**Highlights:**
* ⚡ **10M+ steps/second** — fully JIT-compiled JAX simulator with vectorized `vmap` for massive parallelism
* 🎯 **Pure functional design** — immutable state, reproducible trajectories
* 🚀 **Live deployment** — deploy agents to [generals.io](https://generals.io) servers
* 🎮 **Built-in GUI** — visualize games and debug agent behavior

> [!Note]
> This repository is based on the [generals.io](https://generals.io) game.
> The goal is to provide a fast bot development platform for reinforcement learning research.

## 📦 Installation

```bash
git clone https://github.com/strakam/generals-bots
cd generals-bots
uv sync --extra dev
```

## 🌱 Getting Started

### Basic Game Loop

```python
import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv, get_observation
from generals.agents import RandomAgent, ExpanderAgent

# Create environment (customize grid size and truncation)
env = GeneralsEnv(grid_dims=(10, 10), truncation=500)

# Create agents
agent_0 = RandomAgent()
agent_1 = ExpanderAgent()

# Initialize
key = jrandom.PRNGKey(42)
pool, state = env.reset(key)

# Game loop
while True:
    # Get observations
    obs_0 = get_observation(state, 0)
    obs_1 = get_observation(state, 1)

    # Get actions
    key, k1, k2 = jrandom.split(key, 3)
    action_0 = agent_0.act(obs_0, k1)
    action_1 = agent_1.act(obs_1, k2)
    actions = jnp.stack([action_0, action_1])

    # Step environment (auto-resets from pre-generated pool)
    timestep, state = env.step(state, actions, pool)

    if timestep.terminated or timestep.truncated:
        break

print(f"Winner: Player {int(timestep.info.winner)}")
```

### ⚡Vectorized Parallel Environments

Run **thousands** of games in parallel using `jax.vmap`:

```python
import jax
import jax.random as jrandom
from generals import GeneralsEnv, get_observation

# Create single environment
env = GeneralsEnv(grid_dims=(10, 10), truncation=500)

# Generate state pool once, then create per-env starting states
NUM_ENVS = 1024
key = jrandom.PRNGKey(0)
key, pool_key = jrandom.split(key)
pool, _ = env.reset(pool_key)  # generates shared pool

keys = jrandom.split(key, NUM_ENVS)
states = jax.vmap(env.init_state)(keys)  # Batched states

# Step all environments in parallel (auto-resets from pool)
# ... get batched observations and actions ...
timesteps, states = jax.vmap(lambda s, a: env.step(s, a, pool))(states, actions)
```

See `examples/vectorized_example.py` for a complete example.

## 🌍 Environment

### Observation

Each player receives an `Observation` with these fields:

| Field | Shape | Description |
|-------|-------|-------------|
| `armies` | `(H, W)` | Army counts in visible cells |
| `generals` | `(H, W)` | Mask of visible generals |
| `cities` | `(H, W)` | Mask of visible cities |
| `mountains` | `(H, W)` | Mask of visible mountains |
| `owned_cells` | `(H, W)` | Mask of cells you own |
| `opponent_cells` | `(H, W)` | Mask of opponent's visible cells |
| `neutral_cells` | `(H, W)` | Mask of neutral visible cells |
| `fog_cells` | `(H, W)` | Mask of fog (unexplored) cells |
| `structures_in_fog` | `(H, W)` | Mask of cities/mountains in fog |
| `owned_land_count` | scalar | Total cells you own |
| `owned_army_count` | scalar | Total armies you have |
| `opponent_land_count` | scalar | Opponent's cell count |
| `opponent_army_count` | scalar | Opponent's army count |
| `timestep` | scalar | Current game step |

### Action

Actions are arrays of 5 integers: `[pass, row, col, direction, split]`

| Index | Field | Values |
|-------|-------|--------|
| 0 | `pass` | `1` to pass, `0` to move |
| 1 | `row` | Source cell row |
| 2 | `col` | Source cell column |
| 3 | `direction` | `0`=up, `1`=down, `2`=left, `3`=right |
| 4 | `split` | `1` to send half army, `0` to send all-1 |

Use `compute_valid_move_mask` to get legal moves:

```python
from generals import compute_valid_move_mask

mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
# mask shape: (H, W, 4) - True where move from (i,j) in direction d is valid
```

## 🚀 Deployment

Deploy agents to live [generals.io](https://generals.io) servers:

```python
from generals.remote import autopilot
from generals.agents import ExpanderAgent

agent = ExpanderAgent()
autopilot(agent, user_id="your_user_id", lobby_id="your_lobby")
```

Register at [generals.io](https://generals.io) to get your user ID.

## 🖥️ LAN Battles

For a local bot arena, run the server on the TV machine and connect two laptops as clients over the same network.

Start the server:

```bash
uv run python examples/lan_server.py --host 0.0.0.0 --port 5555 --grid 15 --games 10
```

Connect a client from each laptop:

```bash
uv run python examples/lan_client.py --host <server-lan-ip> --port 5555 --agent expander --name AliceBot
uv run python examples/lan_client.py --host <server-lan-ip> --port 5555 --agent random --name BobBot
```

The server runs the authoritative environment and rotates player sides each game for fairness. Clients only receive observations and send back actions, which makes it a good fit for developing agents independently on separate laptops.

## 🤖 Building Agents

Each laptop can run a custom agent module without editing the shared LAN client script.

Create an agent factory that returns an `Agent`:

```python
from generals.agents import Agent

class MyAgent(Agent):
    def act(self, observation, key):
        ...

def make_agent(name: str) -> Agent:
    return MyAgent(id=name)
```

Then connect it to the server:

```bash
uv run python examples/lan_client.py \
  --host <server-lan-ip> \
  --agent-custom examples/custom_agent.py:make_agent \
  --name AliceBot
```

`--agent-custom` accepts either:
- `python.module:factory`
- `/path/to/file.py:factory`

## 👥 Friend Workflow

Recommended setup for sharing this repo:

1. Commit the shared platform code, including the LAN server, examples, and spectator work.
2. Each of you creates your own agent module locally and connects it with `--agent-custom`.
3. Use `uv run --extra dev pytest` before sharing changes back.
4. Keep spectator/UI work and agent-strategy work in separate branches so they do not conflict.

## 📄 Citation

```bibtex
@misc{generals_rl,
      author    = {Matej Straka, Martin Schmid},
      title     = {Artificial Generals Intelligence: Mastering Generals.io with Reinforcement Learning},
      year      = {2025},
      eprint    = {2507.06825},
      archivePrefix = {arXiv},
      primaryClass = {cs.LG},
}
```
