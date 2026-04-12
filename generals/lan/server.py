"""
LAN game server for head-to-head agent competition.

Hosts the game engine. Two clients connect over TCP, each running their
own agent. A web-based spectator UI is served on a separate port for
display on a TV or any browser.

Usage:
    server = LANServer(env)
    server.run(seed=42)
"""

import socket
import time
from collections import Counter

import jax.numpy as jnp
import jax.random as jrandom

from generals.core.env import GeneralsEnv
from generals.core.game import get_observation

from .protocol import (
    PASS_ACTION,
    recv_msg,
    send_msg,
    serialize_observation,
)

# Default player colors (RGB)
DEFAULT_COLORS = [[220, 56, 56], [56, 120, 220]]


class LANServer:
    def __init__(
        self,
        env: GeneralsEnv,
        host: str = "0.0.0.0",
        port: int = 5555,
        action_timeout: float = 2.0,
        fps: int = 6,
        spectator_port: int = 8080,
        no_spectator: bool = False,
        colors: list[list[int]] | None = None,
    ):
        """
        Args:
            env: Game environment configuration.
            host: Address to bind (0.0.0.0 for all interfaces).
            port: TCP port for agent connections.
            action_timeout: Seconds to wait for each player's action before substituting a pass.
            fps: Game ticks per second.
            spectator_port: HTTP/WebSocket port for the spectator UI.
            no_spectator: If True, run headless without the spectator UI.
            colors: Player colors as [[r,g,b], [r,g,b]].
        """
        self.env = env
        self.host = host
        self.port = port
        self.action_timeout = action_timeout
        self.fps = fps
        self.spectator_port = spectator_port
        self.no_spectator = no_spectator
        self.colors = colors or DEFAULT_COLORS

    def _get_server_ip(self) -> str:
        """Best-effort LAN IP for display purposes."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "localhost"

    def run(self, seed: int = 42, num_games: int | None = None):
        """
        Accept two client connections, then play games in a loop.

        Args:
            seed: Random seed for game generation.
            num_games: Number of games to play. None = infinite.
        """
        # Start spectator broadcast
        spectator = None
        if not self.no_spectator:
            from generals.spectator import SpectatorBroadcast
            spectator = SpectatorBroadcast(host=self.host, port=self.spectator_port)
            print(f"Spectator UI at http://localhost:{self.spectator_port}")

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(2)
        print(f"LAN Server listening on {self.host}:{self.port}")

        server_ip = self._get_server_ip()

        if spectator:
            spectator.set_lobby([], server_ip)

        # Accept two players
        clients, agent_ids = self._accept_players(srv, spectator, server_ip)

        key = jrandom.PRNGKey(seed)
        game_num = 0
        assignment = [0, 1]
        match_score: Counter[str] = Counter()
        draws = 0
        tick_interval = 1.0 / self.fps

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
                        "player_name": [p0_name, p1_name][i],
                        "opponent_name": [p1_name, p0_name][i],
                        "grid_dims": grid_dims,
                        "game_num": game_num,
                    })

                if spectator:
                    spectator.game_start(state, [p0_name, p1_name], self.colors, game_num)

                # Game loop
                terminated = truncated = False
                turn = 0

                while not (terminated or truncated):
                    tick_start = time.monotonic()

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

                    actions = [None, None]
                    for i in range(2):
                        actions[i] = self._recv_action(clients[assignment[i]], agent_ids[assignment[i]])

                    stacked = jnp.stack([jnp.array(a, dtype=jnp.int32) for a in actions])
                    timestep, state = self.env.step(state, stacked, pool)

                    if spectator:
                        spectator.broadcast_state(state, timestep.info)

                    terminated = bool(timestep.terminated)
                    truncated = bool(timestep.truncated)
                    turn += 1

                    # Pace to target FPS
                    elapsed = time.monotonic() - tick_start
                    remaining = tick_interval - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

                # Determine winner
                winner_idx = int(timestep.info.winner)
                if winner_idx >= 0:
                    winner_name = [p0_name, p1_name][winner_idx]
                    match_score[winner_name] += 1
                    print(f"Game {game_num} over after {turn} turns. Winner: {winner_name}")
                else:
                    winner_name = None
                    draws += 1
                    print(f"Game {game_num} truncated after {turn} turns. Draw.")

                score = {
                    agent_ids[0]: match_score[agent_ids[0]],
                    agent_ids[1]: match_score[agent_ids[1]],
                    "draws": draws,
                }

                print(
                    "Match score: "
                    f"{agent_ids[0]}={match_score[agent_ids[0]]}, "
                    f"{agent_ids[1]}={match_score[agent_ids[1]]}, "
                    f"draws={draws}"
                )

                # Send game_end to clients
                for i in range(2):
                    try:
                        send_msg(clients[assignment[i]], {
                            "type": "game_end",
                            "winner": winner_idx,
                            "winner_name": winner_name,
                            "turns": turn,
                            "game_num": game_num,
                            "score": score,
                        })
                    except ConnectionError:
                        pass

                if spectator:
                    spectator.game_end(winner_idx, winner_name, turn, game_num, score)

                # Countdown between games
                for s in range(5, 0, -1):
                    if spectator:
                        spectator.countdown(s, game_num + 1)
                    time.sleep(1)

                # Swap player assignments for fairness
                assignment = [assignment[1], assignment[0]]

        except KeyboardInterrupt:
            print("\nServer shutting down.")
        finally:
            for c in clients:
                c.close()
            srv.close()
            if spectator:
                spectator.shutdown()

    def _accept_players(self, srv: socket.socket, spectator=None, server_ip: str = "") -> tuple[list[socket.socket], list[str]]:
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
            if spectator:
                spectator.set_lobby(list(agent_ids), server_ip)
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
