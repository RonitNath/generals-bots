"""
WebSocket spectator broadcast server.

Runs a WebSocket + HTTP server in a daemon thread. The synchronous game loop
calls broadcast methods to push state to all connected browser spectators.
Also serves the spectator HTML page on the same port.
"""

import asyncio
import json
import threading
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import websockets
import websockets.asyncio.server
from websockets.http11 import Response

from generals.core.game import GameState, GameInfo


def _serialize_game_start(state: GameState, agents: list[str], colors: list[list[int]], game_num: int) -> dict:
    return {
        "type": "game_start",
        "game_num": game_num,
        "grid_dims": list(state.armies.shape),
        "agents": agents,
        "colors": colors,
        "mountains": np.array(state.mountains).astype(int).tolist(),
        "cities": np.array(state.cities).astype(int).tolist(),
        "generals": np.array(state.generals).astype(int).tolist(),
        "general_positions": np.array(state.general_positions).tolist(),
    }


def _serialize_state(state: GameState, info: GameInfo) -> dict:
    return {
        "type": "state",
        "turn": int(state.time),
        "armies": np.array(state.armies).tolist(),
        "ownership": np.array(state.ownership).astype(int).tolist(),
        # Generals and cities can change mid-game (capture), send each frame
        "generals": np.array(state.generals).astype(int).tolist(),
        "cities": np.array(state.cities).astype(int).tolist(),
        "ownership_neutral": np.array(state.ownership_neutral).astype(int).tolist(),
        "stats": [
            {"army": int(info.army[i]), "land": int(info.land[i])}
            for i in range(2)
        ],
        "winner": int(state.winner),
    }


class SpectatorBroadcast:
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self._host = host
        self._port = port
        self._clients: set[websockets.ServerConnection] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

        # Cache for reconnecting browsers
        self._cached_game_start: str | None = None
        self._cached_state: str | None = None
        self._cached_leaderboard: str | None = None

        # Load HTML once
        html_path = Path(files("generals.spectator")) / "index.html"
        self._html = html_path.read_bytes()

        self._start()

    def _start(self):
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start_server())
        self._ready.set()
        self._loop.run_forever()

    async def _start_server(self):
        self._server = await websockets.asyncio.server.serve(
            self._ws_handler,
            self._host,
            self._port,
            process_request=self._http_handler,
        )

    def _http_handler(self, connection, request):
        """Serve HTML for non-WebSocket requests, pass through WebSocket upgrades."""
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None
        return Response(200, "OK", websockets.Headers({
            "Content-Type": "text/html; charset=utf-8",
            "Content-Length": str(len(self._html)),
        }), self._html)

    async def _ws_handler(self, ws: websockets.ServerConnection):
        self._clients.add(ws)
        try:
            # Send cached state on connect (for mid-game reconnects)
            if self._cached_leaderboard:
                await ws.send(self._cached_leaderboard)
            if self._cached_game_start:
                await ws.send(self._cached_game_start)
            if self._cached_state:
                await ws.send(self._cached_state)
            # Keep connection alive, ignore any client messages
            async for _ in ws:
                pass
        finally:
            self._clients.discard(ws)

    def _broadcast(self, msg_json: str):
        if not self._loop or not self._clients:
            return
        self._loop.call_soon_threadsafe(
            asyncio.ensure_future,
            self._broadcast_async(msg_json),
        )

    async def _broadcast_async(self, msg_json: str):
        if not self._clients:
            return
        dead = set()
        for ws in self._clients:
            try:
                await ws.send(msg_json)
            except websockets.ConnectionClosed:
                dead.add(ws)
        self._clients -= dead

    # --- Public API (called from main thread) ---

    def set_lobby(self, players: list[str], server_ip: str = ""):
        msg = json.dumps({"type": "lobby", "players": players, "server_ip": server_ip})
        self._cached_game_start = None
        self._cached_state = None
        # Also cache lobby as the "game_start" for reconnects
        self._cached_game_start = msg
        self._broadcast(msg)

    def game_start(self, state: GameState, agents: list[str], colors: list[list[int]], game_num: int):
        data = _serialize_game_start(state, agents, colors, game_num)
        msg = json.dumps(data)
        self._cached_game_start = msg
        self._cached_state = None
        self._broadcast(msg)

    def broadcast_state(self, state: GameState, info: GameInfo):
        msg = json.dumps(_serialize_state(state, info))
        self._cached_state = msg
        self._broadcast(msg)

    def game_end(self, winner_idx: int, winner_name: str | None, turns: int, game_num: int, score: dict[str, Any]):
        msg = json.dumps({
            "type": "game_end",
            "winner": winner_idx,
            "winner_name": winner_name,
            "turns": turns,
            "game_num": game_num,
            "score": score,
        })
        self._broadcast(msg)

    def leaderboard(self, data: dict):
        msg = json.dumps({"type": "leaderboard", **data})
        self._cached_leaderboard = msg
        self._broadcast(msg)

    def settings(self, fps: int, truncation: int):
        msg = json.dumps({"type": "settings", "fps": fps, "truncation": truncation})
        self._broadcast(msg)

    def countdown(self, seconds: int, next_game: int):
        msg = json.dumps({"type": "countdown", "seconds": seconds, "next_game": next_game})
        self._broadcast(msg)

    def shutdown(self):
        if self._loop and self._loop.is_running():
            async def _close():
                if hasattr(self, '_server'):
                    self._server.close()
                    await self._server.wait_closed()
                self._loop.stop()
            self._loop.call_soon_threadsafe(asyncio.ensure_future, _close())
        if self._thread:
            self._thread.join(timeout=3)
