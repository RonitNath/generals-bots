"""
Connect an agent to a LAN game server.

Usage:
    uv run python examples/lan_client.py --host 192.168.1.10 --name RonitBot
    uv run python examples/lan_client.py --host localhost --agent expander

This is a thin wrapper around the ``generals-client`` entry point.
If generals-bots is installed as a package, use ``generals-client`` directly.
"""

from generals.lan.client_cli import main

if __name__ == "__main__":
    main()
