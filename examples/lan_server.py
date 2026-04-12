"""
Start a LAN game server with web spectator UI.

Both players connect as clients from their own machines.
Open http://localhost:8080 in a browser to watch the game.

Usage:
    uv run python examples/lan_server.py
    uv run python examples/lan_server.py --port 5555 --seed 42 --grid 15 --games 10

This is a thin wrapper around the ``generals-server`` entry point.
If generals-bots is installed as a package, use ``generals-server`` directly.
"""

from generals.lan.server_cli import main

if __name__ == "__main__":
    main()
