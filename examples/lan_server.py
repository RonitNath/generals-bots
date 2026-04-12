"""
Start a LAN game server with web spectator UI.

Both players connect as clients from their own machines.
Open http://localhost:8080 in a browser to watch the game.

Usage:
    uv run python examples/lan_server.py
    uv run python examples/lan_server.py --port 5555 --seed 42 --grid 15 --games 10
"""

import argparse

from generals import GeneralsEnv
from generals.lan import LANServer

parser = argparse.ArgumentParser(description="Generals LAN Server")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Interface to bind on the TV/server machine")
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random per startup)")
parser.add_argument("--grid", type=int, default=15, help="Grid size (square)")
parser.add_argument("--truncation", type=int, default=500)
parser.add_argument("--timeout", type=float, default=2.0, help="Action timeout (seconds)")
parser.add_argument("--fps", type=int, default=6)
parser.add_argument("--games", type=int, default=None, help="Number of games (default: infinite)")
parser.add_argument("--spectator-port", type=int, default=8080, help="HTTP/WebSocket port for spectator UI")
parser.add_argument("--ctl-port", type=int, default=5556, help="TCP control port for remote CLI (default 5556)")
parser.add_argument("--no-spectator", action="store_true", help="Disable web spectator UI")
parser.add_argument("--min-distance", type=int, default=20, help="Min Manhattan distance between generals (default: 20 = opposite sides)")
parser.add_argument("--max-distance", type=int, default=None, help="Max Manhattan distance between generals")
args = parser.parse_args()

env = GeneralsEnv(
    grid_dims=(args.grid, args.grid),
    truncation=args.truncation,
    min_generals_distance=args.min_distance,
    max_generals_distance=args.max_distance,
    pool_size=1,  # LAN plays one game at a time; skip RL's 10K-map pool
)
server = LANServer(
    env,
    host=args.host,
    port=args.port,
    action_timeout=args.timeout,
    fps=args.fps,
    spectator_port=args.spectator_port,
    no_spectator=args.no_spectator,
    ctl_port=args.ctl_port,
)
import time
seed = args.seed if args.seed is not None else int(time.time()) % (2**31)
print(f"Seed: {seed}")
server.run(seed=seed, num_games=args.games)
