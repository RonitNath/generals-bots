"""
Connect an agent to a LAN game server.

Usage:
    python examples/lan_client.py --host 192.168.1.10
    python examples/lan_client.py --host localhost --agent expander

Swap the agent below with your own implementation!
"""

import argparse

from generals.agents import ExpanderAgent, RandomAgent
from generals.lan import LANClient

parser = argparse.ArgumentParser(description="Generals LAN Client")
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--agent", type=str, default="expander", choices=["expander", "random"],
                    help="Built-in agent to use (or replace this script with your own)")
parser.add_argument("--name", type=str, default=None, help="Display name for your agent")
args = parser.parse_args()

if args.agent == "expander":
    agent = ExpanderAgent(id=args.name or "Expander")
elif args.agent == "random":
    agent = RandomAgent(id=args.name or "Random")

client = LANClient(agent, host=args.host, port=args.port)
client.run()
