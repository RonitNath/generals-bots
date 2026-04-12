"""
Connect an agent to a LAN game server.

Usage:
    uv run python examples/lan_client.py --host 192.168.1.10 --name RonitBot
    uv run python examples/lan_client.py --host localhost --agent expander

Swap the agent below with your own implementation!
"""

import argparse

from generals.agents import (
    BackdoorAgent,
    ChaosAgent,
    DefenseCounterAgent,
    ExpanderAgent,
    GreedyCityAgent,
    MaterialAdvantageAgent,
    PunishAgent,
    RandomAgent,
    ScoutPressureAgent,
    SniperAgent,
    SurroundPressureAgent,
    SwarmAgent,
    TurtleAgent,
    build_agent,
)
from generals.lan import LANClient

parser = argparse.ArgumentParser(description="Generals LAN Client")
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--agent", type=str, default="expander", choices=["expander", "random", "material", "scout", "backdoor", "defense", "surround", "turtle", "punish", "swarm", "sniper", "greedy_city", "chaos"],
                    help="Built-in agent to use (or replace this script with your own)")
parser.add_argument(
    "--agent-custom",
    type=str,
    default=None,
    help="Custom agent target: 'python.module:factory' or '/path/to/file.py:factory'",
)
parser.add_argument("--name", type=str, default=None, help="Display name for your agent")
parser.add_argument("--seed", type=int, default=None, help="Optional JAX seed for reproducible agent behavior")
args = parser.parse_args()

if args.agent_custom:
    agent = build_agent(args.agent_custom, name=args.name)
elif args.agent == "expander":
    agent = ExpanderAgent(id=args.name or "Expander")
elif args.agent == "random":
    agent = RandomAgent(id=args.name or "Random")
elif args.agent == "material":
    agent = MaterialAdvantageAgent(id=args.name or "MaterialAdvantage")
elif args.agent == "scout":
    agent = ScoutPressureAgent(id=args.name or "ScoutPressure")
elif args.agent == "backdoor":
    agent = BackdoorAgent(id=args.name or "Backdoor")
elif args.agent == "defense":
    agent = DefenseCounterAgent(id=args.name or "DefenseCounter")
elif args.agent == "surround":
    agent = SurroundPressureAgent(id=args.name or "SurroundPressure")
elif args.agent == "turtle":
    agent = TurtleAgent(id=args.name or "Turtle")
elif args.agent == "punish":
    agent = PunishAgent(id=args.name or "Punish")
elif args.agent == "swarm":
    agent = SwarmAgent(id=args.name or "Swarm")
elif args.agent == "sniper":
    agent = SniperAgent(id=args.name or "Sniper")
elif args.agent == "greedy_city":
    agent = GreedyCityAgent(id=args.name or "GreedyCity")
elif args.agent == "chaos":
    agent = ChaosAgent(id=args.name or "Chaos")

client = LANClient(agent, host=args.host, port=args.port)
client.run(seed=args.seed)
