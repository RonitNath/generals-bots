"""
LAN protocol for message framing and serialization.

Uses 4-byte big-endian length prefix + JSON payload over TCP.
"""

import json
import socket
import struct
from typing import Any

import numpy as np
import jax.numpy as jnp

from generals.core.observation import Observation

# 4-byte big-endian unsigned int for message length
_HEADER_FMT = "!I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from socket, or raise ConnectionError."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_msg(sock: socket.socket, data: dict[str, Any]) -> None:
    """Send a length-prefixed JSON message."""
    payload = json.dumps(data).encode("utf-8")
    header = struct.pack(_HEADER_FMT, len(payload))
    sock.sendall(header + payload)


def recv_msg(sock: socket.socket) -> dict[str, Any]:
    """Receive a length-prefixed JSON message. Raises ConnectionError on disconnect."""
    header = _recv_exactly(sock, _HEADER_SIZE)
    length = struct.unpack(_HEADER_FMT, header)[0]
    payload = _recv_exactly(sock, length)
    return json.loads(payload.decode("utf-8"))


def serialize_observation(obs: Observation) -> dict[str, Any]:
    """Convert an Observation (JAX arrays) to JSON-serializable dict."""
    return {field: np.array(getattr(obs, field)).tolist() for field in Observation._fields}


def deserialize_observation(data: dict[str, Any]) -> Observation:
    """Convert a dict back to an Observation with JAX arrays."""
    return Observation(**{field: jnp.array(data[field]) for field in Observation._fields})


# Pass action: [1, 0, 0, 0, 0]
PASS_ACTION = [1, 0, 0, 0, 0]
