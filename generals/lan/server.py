"""
LAN game server for head-to-head agent competition.

Hosts the game engine and GUI. Two clients connect over TCP,
each running their own agent. The server sends observations
and collects actions each turn.

Usage:
    server = LANServer(env)
    server.run(seed=42)
"""

import socket
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.core.env import GeneralsEnv
from generals.core.game import get_observation
from generals.gui.replay_gui import ReplayGUI

from .protocol import (
    PASS_ACTION,
    recv_msg,
    send_msg,
    serialize_observation,
)


class LANServer:
    def __init__(
        self,
        env: GeneralsEnv,
        host: str = "0.0.0.0",
        port: int = 5555,
        action_timeout: float = 2.0,
        fps: int = 6,
    ):
        """
        Args:
            env: Game environment configuration.
            host: Address to bind (0.0.0.0 for all interfaces).
            port: TCP port.
            action_timeout: Seconds to wait for each player's action before substituting a pass.
            fps: GUI frames per second.
        """
        self.env = env
        self.host = host
        self.port = port
        self.action_timeout = action_timeout
        self.fps = fps

    def run(self, seed: int = 42, num_games: int | None = None):
        """
        Accept two client connections, then play games in a loop.

        Args:
            seed: Random seed for game generation.
            num_games: Number of games to play. None = infinite.
        """
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(2)
        print(f"LAN Server listening on {self.host}:{self.port}")

        # Accept two players
        clients, agent_ids = self._accept_players(srv)

        key = jrandom.PRNGKey(seed)
        game_num = 0
        # Track which client is player 0 vs 1 (swap each game for fairness)
        assignment = [0, 1]

        gui = None

        try:
            while num_games is None or game_num < num_games:
                game_num += 1
                key, reset_key = jrandom.split(key)
                pool, state = self.env.reset(reset_key)

                p0_name = agent_ids[assignment[0]]
                p1_name = agent_ids[assignment[1]]
                grid_dims = list(state.armies.shape)

                print(f"\n--- Game {game_num}: {p0_name} (P0) vs {p1_name} (P1) ---")

                # Send game_start to each client
                for i in range(2):
                    send_msg(clients[assignment[i]], {
                        "type": "game_start",
                        "player_id": i,
                        "grid_dims": grid_dims,
                        "game_num": game_num,
                    })

                # Create or reset GUI
                if gui is None:
                    gui = ReplayGUI(state, agent_ids=[p0_name, p1_name], fps=self.fps)
                else:
                    gui._adapter.update_from_state(state)
                    gui.agent_ids = [p0_name, p1_name]

                # Game loop
                terminated = truncated = False
                turn = 0

                while not (terminated or truncated):
                    # Compute and send observations
                    obs_0 = get_observation(state, 0)
                    obs_1 = get_observation(state, 1)

                    try:
                        send_msg(clients[assignment[0]], {
                            "type": "observation",
                            "obs": serialize_observation(obs_0),
                            "turn": turn,
                        })
                        send_msg(clients[assignment[1]], {
                            "type": "observation",
                            "obs": serialize_observation(obs_1),
                            "turn": turn,
                        })
                    except ConnectionError as e:
                        print(f"Client disconnected during send: {e}")
                        break

                    # Collect actions from both players
                    actions = [None, None]
                    for i in range(2):
                        actions[i] = self._recv_action(clients[assignment[i]], agent_ids[assignment[i]])

                    stacked = jnp.stack([jnp.array(a, dtype=jnp.int32) for a in actions])
                    timestep, state = self.env.step(state, stacked, pool)

                    gui.update(state, timestep.info)
                    gui.tick(fps=self.fps)

                    terminated = bool(timestep.terminated)
                    truncated = bool(timestep.truncated)
                    turn += 1

                # Determine winner
                winner_idx = int(timestep.info.winner)
                if winner_idx >= 0:
                    winner_name = [p0_name, p1_name][winner_idx]
                    print(f"Game {game_num} over after {turn} turns. Winner: {winner_name}")
                else:
                    winner_name = None
                    print(f"Game {game_num} truncated after {turn} turns. Draw.")

                # Send game_end
                for i in range(2):
                    try:
                        send_msg(clients[assignment[i]], {
                            "type": "game_end",
                            "winner": winner_idx,
                            "winner_name": winner_name,
                            "turns": turn,
                            "game_num": game_num,
                        })
                    except ConnectionError:
                        pass

                # Pause briefly so players can see the final state
                time.sleep(2)

                # Swap player assignments for fairness
                assignment = [assignment[1], assignment[0]]

        except KeyboardInterrupt:
            print("\nServer shutting down.")
        finally:
            for c in clients:
                c.close()
            srv.close()
            if gui is not None:
                gui.close()

    def _accept_players(self, srv: socket.socket) -> tuple[list[socket.socket], list[str]]:
        """Wait for two clients to connect and send their join messages."""
        clients = []
        agent_ids = []
        for i in range(2):
            print(f"Waiting for player {i + 1}/2...")
            conn, addr = srv.accept()
            conn.settimeout(10.0)
            try:
                msg = recv_msg(conn)
                assert msg["type"] == "join"
                name = msg["agent_id"]
            except Exception:
                name = f"Player {i}"
            conn.settimeout(None)
            clients.append(conn)
            agent_ids.append(name)
            print(f"  Player {i + 1} connected: {name} from {addr[0]}:{addr[1]}")
        return clients, agent_ids

    def _recv_action(self, sock: socket.socket, name: str) -> list[int]:
        """Receive an action from a client, with timeout fallback to pass."""
        sock.settimeout(self.action_timeout)
        try:
            msg = recv_msg(sock)
            if msg["type"] == "action":
                return msg["action"]
        except (socket.timeout, TimeoutError):
            print(f"  Timeout: {name} passed (exceeded {self.action_timeout}s)")
        except ConnectionError:
            print(f"  Disconnect: {name} — substituting pass")
        finally:
            sock.settimeout(None)
        return PASS_ACTION
