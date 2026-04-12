"""
LAN game client for connecting an agent to a remote server.

Connects to a LANServer, receives observations, runs the agent,
and sends actions back each turn.

Usage:
    client = LANClient(my_agent, host="192.168.1.10")
    client.run()
"""

import socket
import time

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from generals.agents.agent import Agent

from .protocol import (
    deserialize_observation,
    recv_msg,
    send_msg,
)


class LANClient:
    def __init__(self, agent: Agent, host: str = "localhost", port: int = 5555):
        """
        Args:
            agent: Agent instance to play with.
            host: Server hostname or IP.
            port: Server TCP port.
        """
        self.agent = agent
        self.host = host
        self.port = port

    def run(self, seed: int | None = None):
        """
        Connect to the server and play games until disconnected.

        Args:
            seed: Random seed for agent's JAX key. Defaults to current time.
        """
        if seed is None:
            seed = int(time.time() * 1000) % (2**31)
        key = jrandom.PRNGKey(seed)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        print(f"Connected to {self.host}:{self.port}")

        # Send join
        send_msg(sock, {"type": "join", "agent_id": self.agent.id})

        try:
            while True:
                msg = recv_msg(sock)

                if msg["type"] == "game_start":
                    player_id = msg["player_id"]
                    player_name = msg.get("player_name", self.agent.id)
                    opponent_name = msg.get("opponent_name", "Opponent")
                    grid_dims = msg["grid_dims"]
                    game_num = msg["game_num"]
                    print(
                        f"\nGame {game_num} starting — "
                        f"you are Player {player_id} ({player_name}) vs {opponent_name} "
                        f"on {grid_dims[0]}x{grid_dims[1]}"
                    )
                    self.agent.reset()

                elif msg["type"] == "observation":
                    obs = deserialize_observation(msg["obs"])
                    key, act_key = jrandom.split(key)
                    action = self.agent.act(obs, act_key)
                    send_msg(sock, {
                        "type": "action",
                        "action": np.array(action).tolist(),
                    })

                elif msg["type"] == "game_end":
                    winner = msg["winner"]
                    winner_name = msg.get("winner_name", "???")
                    turns = msg["turns"]
                    score = msg.get("score", {})
                    if winner == player_id:
                        print(f"Game {msg['game_num']} — YOU WON in {turns} turns!")
                    elif winner < 0:
                        print(f"Game {msg['game_num']} — DRAW after {turns} turns.")
                    else:
                        print(f"Game {msg['game_num']} — You lost. Winner: {winner_name} ({turns} turns)")
                    if score:
                        print(f"Match score: {score}")

        except ConnectionError:
            print("Disconnected from server.")
        except KeyboardInterrupt:
            print("\nClient shutting down.")
        finally:
            sock.close()
