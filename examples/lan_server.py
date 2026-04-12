"""
Start a LAN game server with GUI.

Both players connect as clients from their own machines.

Usage:
    python examples/lan_server.py
    python examples/lan_server.py --port 5555 --seed 42 --grid 15
"""

import argparse

from generals import GeneralsEnv
from generals.lan import LANServer

parser = argparse.ArgumentParser(description="Generals LAN Server")
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--grid", type=int, default=15, help="Grid size (square)")
parser.add_argument("--truncation", type=int, default=500)
parser.add_argument("--timeout", type=float, default=2.0, help="Action timeout (seconds)")
parser.add_argument("--fps", type=int, default=6)
parser.add_argument("--games", type=int, default=None, help="Number of games (default: infinite)")
args = parser.parse_args()

env = GeneralsEnv(grid_dims=(args.grid, args.grid), truncation=args.truncation)
server = LANServer(env, port=args.port, action_timeout=args.timeout, fps=args.fps)
server.run(seed=args.seed, num_games=args.games)
