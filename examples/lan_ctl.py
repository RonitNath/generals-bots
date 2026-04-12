"""
Control a running LAN game server.

Usage:
    uv run python examples/lan_ctl.py kick 1
    uv run python examples/lan_ctl.py fps 20
    uv run python examples/lan_ctl.py end
    uv run python examples/lan_ctl.py          # interactive mode
"""

import argparse
import socket
import sys


def send_command(host: str, port: int, cmd: str) -> str:
    """Send a single command and return the response."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    sock.connect((host, port))
    sock.sendall((cmd + "\n").encode())
    resp = sock.recv(4096).decode().strip()
    sock.close()
    return resp


def interactive(host: str, port: int):
    """Interactive REPL sending commands to the server."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    sock.connect((host, port))
    print(f"Connected to {host}:{port}. Type commands (Ctrl-C to quit).")
    try:
        while True:
            cmd = input(">> ").strip()
            if not cmd:
                continue
            sock.sendall((cmd + "\n").encode())
            sock.settimeout(2)
            try:
                resp = sock.recv(4096).decode().strip()
                print(resp)
            except socket.timeout:
                pass
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        sock.close()


parser = argparse.ArgumentParser(description="Generals LAN Server Control")
parser.add_argument("--host", type=str, default="192.168.0.168")
parser.add_argument("--port", type=int, default=5556, help="Control port (default 5556)")
parser.add_argument("command", nargs="*", help="Command to send (omit for interactive mode)")
args = parser.parse_args()

if args.command:
    cmd = " ".join(args.command)
    try:
        resp = send_command(args.host, args.port, cmd)
        print(resp)
    except ConnectionRefusedError:
        print(f"Cannot connect to {args.host}:{args.port} — is the server running?", file=sys.stderr)
        sys.exit(1)
else:
    try:
        interactive(args.host, args.port)
    except ConnectionRefusedError:
        print(f"Cannot connect to {args.host}:{args.port} — is the server running?", file=sys.stderr)
        sys.exit(1)
