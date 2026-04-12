"""
LAN game server for head-to-head agent competition.

Hosts the game engine. Two clients connect over TCP, each running their
own agent. A web-based spectator UI is served on a separate port for
display on a TV or any browser.

CLI commands (type while server is running):
    fps <N>          Set game speed (ticks per second)
    truncation <N>   Set max turns for next game
    end              Force-end the current game (draw)
    help             Show available commands

Usage:
    server = LANServer(env)
    server.run(seed=42)
"""

import json
import os
import queue
import socket
import sys
import threading
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

LEADERBOARD_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "leaderboard.json")


class Leaderboard:
    """Persistent win/loss/draw tracker, saved to JSON."""

    def __init__(self, path: str = LEADERBOARD_PATH):
        self._path = os.path.abspath(path)
        self._data: dict[str, dict] = {}  # {agent_name: {wins, losses, draws, games}}
        self._history: list[dict] = []    # recent game results
        self._load()

    def _load(self):
        try:
            with open(self._path) as f:
                saved = json.load(f)
                self._data = saved.get("agents", {})
                self._history = saved.get("history", [])
        except (FileNotFoundError, json.JSONDecodeError):
            self._data = {}
            self._history = []

    def _save(self):
        with open(self._path, "w") as f:
            json.dump({"agents": self._data, "history": self._history}, f, indent=2)

    def _ensure(self, name: str):
        if name not in self._data:
            self._data[name] = {"wins": 0, "losses": 0, "draws": 0, "games": 0}

    def record(self, p0_name: str, p1_name: str, winner_name: str | None, turns: int):
        self._ensure(p0_name)
        self._ensure(p1_name)
        if winner_name:
            loser = p1_name if winner_name == p0_name else p0_name
            self._data[winner_name]["wins"] += 1
            self._data[winner_name]["games"] += 1
            self._data[loser]["losses"] += 1
            self._data[loser]["games"] += 1
        else:
            self._data[p0_name]["draws"] += 1
            self._data[p0_name]["games"] += 1
            self._data[p1_name]["draws"] += 1
            self._data[p1_name]["games"] += 1

        self._history.append({
            "p0": p0_name, "p1": p1_name,
            "winner": winner_name, "turns": turns,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        # Keep last 100 games
        self._history = self._history[-100:]
        self._save()

    def to_dict(self) -> dict:
        """Leaderboard sorted by wins desc, for broadcast."""
        ranked = sorted(self._data.items(), key=lambda x: (-x[1]["wins"], x[1]["losses"]))
        return {
            "rankings": [{"name": n, **s} for n, s in ranked],
            "last_game": self._history[-1] if self._history else None,
        }

_HELP_TEXT = """
Commands:
  fps <N>          Set game speed (ticks/sec), e.g. 'fps 20'
  truncation <N>   Set max turns for next game, e.g. 'truncation 300'
  end              Force-end the current game as a draw
  kick 1           Kick player 1 (returns to lobby for replacement)
  kick 2           Kick player 2
  kick all         Kick both players (full lobby reset)
  help             Show this message
""".strip()


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
        ctl_port: int = 5556,
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
        self.ctl_port = ctl_port

        # Mutable settings changed via CLI commands
        self._next_truncation = env.truncation
        self._force_end = False
        self._kick: list[int] = []  # indices into clients[] to kick
        self._cmd_queue: queue.Queue[str] = queue.Queue()

    def _start_cli(self):
        """Start stdin reader and TCP control socket."""
        # Stdin reader (for local interactive use)
        def _stdin_reader():
            while True:
                try:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    line = line.strip()
                    if line:
                        self._cmd_queue.put(line)
                except EOFError:
                    break
        threading.Thread(target=_stdin_reader, daemon=True).start()

        # TCP control socket (for remote CLI)
        def _ctl_server():
            ctl = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ctl.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            ctl.bind((self.host, self.ctl_port))
            ctl.listen(4)
            while True:
                conn, addr = ctl.accept()
                threading.Thread(target=self._handle_ctl_client, args=(conn,), daemon=True).start()
        threading.Thread(target=_ctl_server, daemon=True).start()

    def _handle_ctl_client(self, conn: socket.socket):
        """Handle a single control connection. One command per line, responses sent back."""
        try:
            buf = b""
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    cmd = line.decode("utf-8", errors="replace").strip()
                    if cmd:
                        self._cmd_queue.put(cmd)
                        conn.sendall(f"ok: {cmd}\n".encode())
        except (ConnectionError, OSError):
            pass
        finally:
            conn.close()

    def _process_commands(self, spectator=None):
        """Drain command queue, apply changes. Called each tick."""
        while not self._cmd_queue.empty():
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                break
            parts = cmd.split()
            verb = parts[0].lower() if parts else ""

            if verb == "fps" and len(parts) >= 2:
                try:
                    new_fps = max(1, min(120, int(parts[1])))
                    self.fps = new_fps
                    print(f"  >> FPS set to {new_fps}")
                    if spectator:
                        spectator.settings(self.fps, self._next_truncation)
                except ValueError:
                    print("  >> Usage: fps <number>")

            elif verb == "truncation" and len(parts) >= 2:
                try:
                    new_trunc = max(10, int(parts[1]))
                    self._next_truncation = new_trunc
                    print(f"  >> Next game truncation set to {new_trunc}")
                    if spectator:
                        spectator.settings(self.fps, self._next_truncation)
                except ValueError:
                    print("  >> Usage: truncation <number>")

            elif verb == "end":
                self._force_end = True
                print("  >> Force-ending current game")

            elif verb == "kick" and len(parts) >= 2:
                target = parts[1].lower()
                if target == "1":
                    self._kick = [0]
                    self._force_end = True
                    print("  >> Kicking player 1")
                elif target == "2":
                    self._kick = [1]
                    self._force_end = True
                    print("  >> Kicking player 2")
                elif target == "all":
                    self._kick = [0, 1]
                    self._force_end = True
                    print("  >> Kicking all players")
                else:
                    print("  >> Usage: kick 1|2|all")

            elif verb == "help":
                print(_HELP_TEXT)

            else:
                print(f"  >> Unknown command: {cmd}. Type 'help' for options.")

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
        print(f"Control port on {self.host}:{self.ctl_port}")
        print(f"Type 'help' for server commands.\n")

        self._start_cli()

        server_ip = self._get_server_ip()
        key = jrandom.PRNGKey(seed)
        game_num = 0
        leaderboard = Leaderboard()

        try:
            # Outer loop: lobby → games → kick → lobby
            while True:
                if spectator:
                    spectator.set_lobby([], server_ip)
                    spectator.settings(self.fps, self._next_truncation)
                    spectator.leaderboard(leaderboard.to_dict())

                clients, agent_ids = self._accept_players(srv, spectator, server_ip)

                assignment = [0, 1]
                match_score: Counter[str] = Counter()
                draws = 0

                # Inner loop: play games until kick or num_games reached
                while num_games is None or game_num < num_games:
                    game_num += 1

                    # Apply truncation changes between games
                    if self._next_truncation != self.env.truncation:
                        self.env = GeneralsEnv(
                            grid_dims=self.env._fixed_dims,
                            truncation=self._next_truncation,
                            mountain_density_range=(self.env.mountain_density_range
                                                    if hasattr(self.env, 'mountain_density_range')
                                                    else (0.18, 0.26)),
                            min_generals_distance=self.env.min_generals_distance,
                            max_generals_distance=self.env.max_generals_distance,
                        )
                        print(f"  >> Env rebuilt with truncation={self._next_truncation}")

                    self._force_end = False
                    self._kick = []

                    key, reset_key = jrandom.split(key)
                    pool, state = self.env.reset(reset_key)

                    p0_name = agent_ids[assignment[0]]
                    p1_name = agent_ids[assignment[1]]
                    grid_dims = list(state.armies.shape)

                    print(f"\n--- Game {game_num}: {p0_name} (P0) vs {p1_name} (P1) ---")

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

                        self._process_commands(spectator)
                        if self._force_end:
                            print(f"  >> Game {game_num} force-ended at turn {turn}")
                            break

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

                        elapsed = time.monotonic() - tick_start
                        remaining = (1.0 / self.fps) - elapsed
                        if remaining > 0:
                            time.sleep(remaining)

                    # Determine winner
                    winner_idx = int(timestep.info.winner) if not self._kick else -1
                    if winner_idx >= 0:
                        winner_name = [p0_name, p1_name][winner_idx]
                        match_score[winner_name] += 1
                        print(f"Game {game_num} over after {turn} turns. Winner: {winner_name}")
                    else:
                        winner_name = None
                        draws += 1
                        print(f"Game {game_num} {'force-ended' if self._kick else 'truncated'} after {turn} turns. Draw.")

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

                    # Record to persistent leaderboard (skip kicked games)
                    if not self._kick:
                        leaderboard.record(p0_name, p1_name, winner_name, turn)
                        if spectator:
                            spectator.leaderboard(leaderboard.to_dict())

                    # Handle kicks — close sockets and break to lobby
                    if self._kick:
                        for idx in sorted(self._kick, reverse=True):
                            name = agent_ids[idx]
                            print(f"  >> Disconnecting {name}")
                            try:
                                clients[idx].close()
                            except Exception:
                                pass
                            clients.pop(idx)
                            agent_ids.pop(idx)
                        # Close remaining clients too (they'll reconnect)
                        for c in clients:
                            try:
                                c.close()
                            except Exception:
                                pass
                        self._kick = []
                        break  # back to lobby

                    # Countdown between games
                    for s in range(5, 0, -1):
                        if spectator:
                            spectator.countdown(s, game_num + 1)
                        time.sleep(1)

                    assignment = [assignment[1], assignment[0]]

        except KeyboardInterrupt:
            print("\nServer shutting down.")
        finally:
            for c in clients:
                try:
                    c.close()
                except Exception:
                    pass
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
