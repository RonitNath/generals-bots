"""
CLI entry point for connecting an agent to a LAN game server.

Installed as ``generals-client`` via pyproject.toml console scripts.
"""

import argparse
import sys

from generals.agents import BUILTIN_AGENTS, build_agent, build_builtin_agent
from generals.lan import LANClient


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Connect an agent to a Generals LAN server",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--agent", type=str, default="expander",
        choices=sorted(BUILTIN_AGENTS),
        help="Built-in agent to use",
    )
    parser.add_argument(
        "--agent-custom", type=str, default=None,
        help="Custom agent: 'module:factory' or '/path/to/file.py:factory'",
    )
    parser.add_argument("--name", type=str, default=None, help="Display name for your agent")
    parser.add_argument("--seed", type=int, default=None, help="JAX random seed")
    args = parser.parse_args(argv)

    if args.agent_custom:
        agent = build_agent(args.agent_custom, name=args.name)
    else:
        agent = build_builtin_agent(args.agent, args.name)

    client = LANClient(agent, host=args.host, port=args.port)
    client.run(seed=args.seed)


if __name__ == "__main__":
    main()
