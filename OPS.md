# Generals LAN Competition — Operations Guide

## Architecture

```
Laptops (agents)          NixOS Server "gateway" (TV)
┌──────────┐  TCP:5555   ┌─────────────────────────┐
│ Agent A   │────────────▶│ generals-server          │
│ Agent B   │────────────▶│  (headless JAX engine)   │
│ Agent C   │  (queued)   │                          │
│  ...      │             │ :8080 HTTP + WebSocket   │
└──────────┘              └───────────┬──────────────┘
                                      │
                          ┌───────────▼──────────────┐
                          │ cage-tty1 (Chromium)      │
                          │  Kiosk browser on TV      │
                          └──────────────────────────┘
```

- **generals-server** (systemd): Hosts the game engine, accepts 2 TCP agent connections at a time, broadcasts state via WebSocket to the spectator UI. Extra agents wait in a reconnect loop.
- **cage-tty1** (systemd): Wayland compositor running Chromium in kiosk mode on the physical display. Independent of the game server — has its own retry loop.
- **Spectator UI**: Single-page HTML+Canvas app served on `:8080`. Connects via WebSocket, auto-reconnects on disconnect.

## Server Access

```bash
ssh g                          # alias for the NixOS server (192.168.0.168)
```

## Service Management

```bash
# View status
sudo systemctl status generals-server
sudo systemctl status cage-tty1

# Restart game server (agents auto-reconnect within ~5s)
sudo systemctl restart generals-server

# Restart display (if TV goes black / stuck on TTY)
sudo systemctl restart cage-tty1

# View logs
journalctl -u generals-server -f              # follow live
journalctl -u generals-server --since '5 min ago'
journalctl -u cage-tty1 -f
```

## Deploying Code Changes

```bash
# On your laptop:
git push

# On the server (or via ssh):
ssh g "cd ~/generals-bots && git pull && sudo systemctl restart generals-server"
```

The browser auto-reconnects — no need to restart cage-tty1 for code changes.

## Running Agents

From your laptop:

```bash
# Single agent
uv run python examples/lan_client.py --host 192.168.0.168 --agent material

# Multiple agents (server takes 2 at a time, others queue)
for agent in material punish turtle swarm sniper chaos; do
  nohup uv run python examples/lan_client.py --host 192.168.0.168 --agent "$agent" > /dev/null 2>&1 &
done

# Custom agent from a file
uv run python examples/lan_client.py --host 192.168.0.168 --agent-custom ./my_agent.py:MyAgent --name "MyBot"

# Kill all local agents
pkill -f lan_client
```

Available built-in agents: `expander`, `random`, `material`, `scout`, `backdoor`, `defense`, `surround`, `turtle`, `punish`, `swarm`, `sniper`, `greedy_city`, `chaos`.

## Remote Server Control (lan_ctl)

Control the running server without SSH:

```bash
# Single commands
uv run python examples/lan_ctl.py fps 20           # change game speed
uv run python examples/lan_ctl.py truncation 5000   # change max turns
uv run python examples/lan_ctl.py end                # force-end current game
uv run python examples/lan_ctl.py kick 1             # kick player 1
uv run python examples/lan_ctl.py kick 2             # kick player 2
uv run python examples/lan_ctl.py kick all           # kick both (back to lobby)
uv run python examples/lan_ctl.py help                # list commands

# Interactive mode
uv run python examples/lan_ctl.py
>> fps 30
ok: fps → 30
>> end
ok: force-end
```

Default host: `192.168.0.168`, default port: `5556`. Override with `--host` and `--port`.

## Server CLI Arguments

```bash
uv run python examples/lan_server.py \
  --grid 30              # grid size (square) — default 15
  --fps 15               # ticks per second — default 6
  --truncation 10000     # max turns before draw — default 500
  --timeout 2.0          # action timeout (seconds) — default 2.0
  --seed 42              # random seed (default: time-based)
  --games 10             # number of games (default: infinite)
  --min-distance 20      # min Manhattan distance between generals
  --spectator-port 8080  # HTTP/WS port
  --ctl-port 5556        # TCP control port
  --no-spectator          # disable web UI
```

Current production settings: `--grid 30 --fps 15 --truncation 10000`

## Leaderboard

Persistent across server restarts. Stored at `~/.generals/leaderboard.json` on the server.

Tracks: wins, losses, draws, games played, win rate, last 1000 game history per agent.

Displayed in the spectator UI:
- **Lobby**: Full table with all stats
- **In-game**: Compact sidebar (top-right)
- **Last winner** highlighted in green

## Troubleshooting

### TV shows TTY / black screen
```bash
ssh g "sudo systemctl restart cage-tty1"
```
The browser launcher loops, waiting for the server to respond on :8080 before opening Chromium.

### Server stuck "Waiting for players"
No agents are connected. Launch agents from your laptop (see above).

### Server takes minutes to start first game
Normal — JAX JIT-compiles the environment on first `reset()`. Uses ~8GB RAM and 400% CPU for 2-5 minutes on a 30x30 grid. Subsequent games start instantly.

### Agent disconnects crash the server
Fixed: game_start send is now wrapped in try/except. The server breaks back to lobby and waits for new connections. Agents auto-reconnect within 5s.

### "BrokenPipeError" in logs
A client disconnected mid-communication. The server should recover automatically. If it doesn't, `sudo systemctl restart generals-server`.

### Stale browser cache
The server sends `Cache-Control: no-cache, no-store, must-revalidate` headers. If the UI looks wrong after a code update:
```bash
ssh g "sudo systemctl restart cage-tty1"
```

### Map feels repetitive
Maps are generated from a seed. Default seed is `int(time.time())`, so restarting the server gives new maps. The pool contains 10K pre-generated maps that cycle.

## NixOS Configuration

Service config: `/etc/nixos/generals-game.nix`

To apply NixOS config changes:
```bash
ssh g "sudo nixos-rebuild switch"
```

Firewall ports: `5555` (game TCP), `5556` (control TCP), `8080` (spectator HTTP/WS).

## Ports Summary

| Port | Protocol | Purpose |
|------|----------|---------|
| 5555 | TCP | Agent connections (game protocol) |
| 5556 | TCP | Server control (lan_ctl) |
| 8080 | HTTP+WS | Spectator UI (browser) |
