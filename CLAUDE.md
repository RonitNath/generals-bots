# Generals Bots

JAX-based simulator for [generals.io](https://generals.io), optimized for reinforcement learning research. Achieves 10M+ steps/second via JIT compilation with massive parallelism (1000s of simultaneous games via `jax.vmap`).

Fork of [strakam/generals-bots](https://github.com/strakam/generals-bots) with LAN multiplayer support.

## Quick Start

```bash
uv sync                    # install deps
uv run python3 examples/simple_example.py         # single game
uv run python3 examples/visualization_example.py   # with GUI
```

On NixOS, use `nix develop` first (provides native libs for JAX/pygame).

## Architecture

```
generals/
├── core/           # JAX game engine (pure functional, immutable NamedTuple state)
│   ├── game.py         # GameState, step(), get_observation(), combat logic
│   ├── env.py          # GeneralsEnv: reset()/step() with pre-computed state pool
│   ├── grid.py         # Procedural map generation with connectivity guarantees
│   ├── observation.py  # 14-field Observation NamedTuple (fog of war applied)
│   ├── action.py       # Action format [pass, row, col, direction, split], valid move masks
│   ├── rewards.py      # Reward shaping: win/lose, composite, ratio-based
│   ├── config.py       # Constants, enums
│   └── rendering.py    # JaxGameAdapter: GameState → GUI protocol
├── agents/         # Agent implementations
│   ├── agent.py        # Abstract Agent base class: act(obs, key) → action
│   ├── random_agent.py # Random valid moves
│   ├── expander_agent.py # Greedy territorial expansion
│   ├── strategic_agent.py # StrategicAgent base + heuristic subclasses
│   ├── graph_search_agent.py # BFS/graph-based agent
│   └── loading.py      # Dynamic agent loading from module:factory or file.py:factory
├── gui/            # PyGame visualization
│   ├── gui.py          # Main GUI class
│   ├── replay_gui.py   # Simple ReplayGUI wrapper for game visualization
│   ├── rendering.py    # Tile/cell rendering
│   ├── event_handler.py
│   └── properties.py
└── lan/            # LAN multiplayer (TCP, no extra deps)
    ├── protocol.py     # 4-byte length prefix + JSON framing, observation serialization
    ├── server.py       # LANServer: hosts game engine, accepts 2 TCP clients
    ├── client.py       # LANClient: wraps any Agent, connects to server
    ├── client_cli.py   # generals-client entry point
    └── server_cli.py   # generals-server entry point
```

## Key Concepts

### Game State & Environment
- `GameState`: immutable NamedTuple (armies, ownership, generals, cities, mountains, etc.)
- `GeneralsEnv.reset(key)` → `(pool, state)` — pool is 10K pre-generated states for cheap auto-reset
- `GeneralsEnv.step(state, actions, pool)` → `(TimeStep, GameState)` — actions shape (2, 5)

### Observations
14-channel player view with fog of war: armies, generals, cities, mountains, owned/opponent/neutral/fog cells, structures_in_fog, land/army counts, timestep. Convert to tensor with `obs.as_tensor()` → (14, H, W).

### Actions
5-int array: `[pass, row, col, direction, split]`. Direction: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT. Use `compute_valid_move_mask(obs)` → (H, W, 4) boolean mask.

### Writing an Agent
```python
from generals.agents import Agent
from generals.core import action

class MyAgent(Agent):
    def act(self, observation, key):
        valid = action.compute_valid_move_mask(observation)
        # your logic here
        return action.create_action(row=r, col=c, direction=d)
```

## LAN Multiplayer

One machine runs the server (game engine + GUI), two players connect as clients:

```bash
# Server (displays GUI)
uv run python3 examples/lan_server.py --grid 15 --fps 6

# Each player on their laptop
uv run python3 examples/lan_client.py --host <server-ip> --name "MyBot"
```

Games auto-rematch with swapped sides. 2s action timeout (configurable).

## NixOS Deployment

The project includes a `flake.nix` providing a devShell with all native dependencies.
The NixOS server `gateway` has cage (Wayland compositor) + Python + uv installed.
To display the game GUI on the server's physical screen:

```bash
ssh g
cd ~/generals-bots
nix develop
cage -- uv run python3 examples/lan_server.py --grid 15
```

## Development

```bash
uv sync                     # install deps
uv run pytest               # run tests
uv run python3 bench.py     # performance benchmark
```
