# Generals LAN Competition — Operations Guide

## Architecture

```
Laptops (agents)          NixOS Server "gateway" (192.168.0.168)
┌──────────┐  TCP:5555   ┌─────────────────────────┐
│ Agent A   │────────────▶│ generals-server.service  │
│ Agent B   │────────────▶│  (headless JAX engine)   │
│ Agent C   │  (queued)   │                          │
│  ...      │             │  :8080 HTTP + WebSocket  │
└──────────┘              └───────────┬──────────────┘
              TCP:5556                │
┌──────────┐  (control)   ┌───────────▼──────────────┐
│ lan_ctl   │────────────▶│ cage-tty1.service         │
└──────────┘              │  cage → Chromium kiosk    │
                          │  (Wayland on physical TV) │
                          └──────────────────────────┘
```

Two independent systemd services:
- **generals-server**: Headless Python game engine. TCP port 5555 for agents, 5556 for control, 8080 for spectator WebSocket + HTTP.
- **cage-tty1**: Wayland compositor (`cage`) running a browser launcher script. The script loops: polls `:8080` until the server is up, launches Chromium kiosk, and restarts if the browser exits.

They are intentionally **not** linked via systemd dependencies. cage-tty1 has its own retry loop and survives server restarts independently.

## Server Access

```bash
ssh g                              # ~/.ssh/config alias for 192.168.0.168
```

The user `ronitnath` has passwordless sudo on the server.

## Deploying Code Changes

```bash
# From your laptop:
git push
ssh g "cd ~/generals-bots && git pull && sudo systemctl restart generals-server"
```

The browser auto-reconnects to the WebSocket — no need to restart cage-tty1 for code-only changes. If you changed `index.html`, the server sends `Cache-Control: no-cache` headers so a browser reconnect picks up the new version.

## NixOS Configuration

### Key files

All config lives in `/etc/nixos/` on the server:

| File | Purpose |
|------|---------|
| `configuration.nix` | Base system: networking, users, packages, SSH, flakes |
| `generals-game.nix` | Game server + browser kiosk services |
| `hardware-configuration.nix` | Auto-generated hardware config |
| `wireguard-ingress.nix` | WireGuard / Netbird VPN |
| `victoriametrics-monitoring.nix` | Metrics collection |

### Applying NixOS changes

**Never edit NixOS config files with `sed` or ad-hoc mutations.** NixOS is declarative — the correct workflow is:

```bash
# 1. Edit the config
ssh g "sudo vim /etc/nixos/generals-game.nix"

# 2. Rebuild and switch atomically
ssh g "sudo nixos-rebuild switch"
```

`nixos-rebuild switch` compiles the entire system config, builds derivations, and atomically switches. If it fails, the old generation is untouched. This is the ONLY way to apply config changes — manual service file edits, `sed` patches, or `systemctl edit` are not durable and get overwritten on the next rebuild.

### Current generals-game.nix structure

```nix
{ config, pkgs, ... }:
let
  # Native libs for pip-installed JAX wheels (libstdc++, zlib, glib)
  nativeLibPath = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    pkgs.glib
  ];

  gameServer = pkgs.writeShellScript "generals-server" ''
    cd /home/ronitnath/generals-bots
    export LD_LIBRARY_PATH="${nativeLibPath}:''${LD_LIBRARY_PATH:-}"
    exec ${pkgs.uv}/bin/uv run python3 examples/lan_server.py --grid 15 --fps 15 --truncation 10000
  '';

  browserLauncher = pkgs.writeShellScript "generals-browser" ''
    while true; do
      while ! ${pkgs.curl}/bin/curl -sf http://localhost:8080/ > /dev/null 2>&1; do
        sleep 2
      done
      ${pkgs.chromium}/bin/chromium \
        --kiosk --no-first-run --disable-infobars \
        --disable-translate --noerrdialogs \
        --disable-session-crashed-bubble \
        --disable-features=TranslateUI \
        http://localhost:8080 || true
      sleep 2
    done
  '';
in { ... }
```

### Changing game parameters

To change grid size, fps, truncation, etc., edit the `gameServer` script args in `generals-game.nix`:

```bash
ssh g "sudo vim /etc/nixos/generals-game.nix"
# Edit the --grid, --fps, --truncation args in gameServer
ssh g "sudo nixos-rebuild switch"
# Service auto-restarts with new args
```

Alternatively, use `lan_ctl` for runtime changes to fps/truncation (doesn't survive restarts).

### NixOS patterns to know

- **All binaries must use full Nix store paths** in service scripts: `${pkgs.curl}/bin/curl`, not just `curl`. Nix doesn't put packages on `$PATH` inside systemd units.
- **`LD_LIBRARY_PATH`** is required for pip-installed wheels (JAX, scipy) because they link against `libstdc++.so.6`, `libz.so.1`, etc. which aren't in the default Nix runtime linker path. The `nativeLibPath` variable handles this.
- **`pkgs.writeShellScript`** creates an immutable script in the Nix store. You can't edit it in place — change the Nix expression and rebuild.
- **`services.cage`** is a NixOS module that manages the cage Wayland compositor. It creates `cage-tty1.service`. The `extraArguments = [ "-s" ]` flag makes cage survive the last client closing.
- **`WLR_LIBINPUT_NO_DEVICES = "1"`** prevents cage from failing when no input devices are present (headless TV server).
- **`hardware.graphics.enable = true`** is required for GPU-accelerated Chromium rendering.
- **Firewall is declarative**: `networking.firewall.allowedTCPPorts = [ 5555 5556 8080 ];`
- **`PYTHONUNBUFFERED=1`** in the service Environment ensures Python stdout appears in journald immediately (systemd pipes are fully buffered by default).

### flake.nix (dev environment)

The repo's `flake.nix` provides a devShell for local development. On the NixOS server:

```bash
cd ~/generals-bots
nix develop          # enters shell with python311, uv, SDL2, cage, LD_LIBRARY_PATH
```

The devShell includes SDL2 libs for pygame (local GUI debugging). The server's systemd service doesn't use the flake — it uses `uv run` directly with the `nativeLibPath` from `generals-game.nix`.

## Service Management

```bash
# Status
ssh g "sudo systemctl status generals-server"
ssh g "sudo systemctl status cage-tty1"

# Restart game server (agents auto-reconnect within ~5s)
ssh g "sudo systemctl restart generals-server"

# Restart display (TV shows TTY / black screen / stale UI)
ssh g "sudo systemctl restart cage-tty1"

# Stop everything
ssh g "sudo systemctl stop generals-server cage-tty1"

# Logs (live follow)
ssh g "journalctl -u generals-server -f"
ssh g "journalctl -u cage-tty1 -f"

# Logs (recent)
ssh g "journalctl -u generals-server --since '5 min ago' --no-pager"

# Process-level check (useful during JAX compilation when journald has no output yet)
ssh g "ps -o pid,etime,%cpu,%mem,rss -C python3"
```

### Service behavior

Both services have `Restart = "always"` — they come back automatically after crashes, stops, or reboots. The game server has `RestartSec = 2`, cage has `RestartSec = 3`.

The cage browser launcher is a loop script, not a one-shot browser exec. This means:
1. cage starts the launcher script
2. Script polls `localhost:8080` every 2s until the game server responds
3. Launches Chromium kiosk (blocks until browser exits)
4. If browser crashes or server goes away, loops back to step 2

This makes the display resilient to server restarts without needing systemd dependency wiring.

## Running Agents

From your laptop:

```bash
# Single agent
uv run python examples/lan_client.py --host 192.168.0.168 --agent material

# Multiple agents (server takes 2 at a time, others queue with 5s reconnect)
for agent in material punish turtle swarm sniper chaos; do
  nohup uv run python examples/lan_client.py --host 192.168.0.168 --agent "$agent" > /dev/null 2>&1 &
done

# Custom agent from a file
uv run python examples/lan_client.py --host 192.168.0.168 \
  --agent-custom ./my_agent.py:MyAgent --name "MyBot"

# Kill all local agents
pkill -f lan_client

# Check running agents
ps aux | grep lan_client | grep -v grep
```

Available built-in agents: `expander`, `random`, `material`, `scout`, `backdoor`, `defense`, `surround`, `turtle`, `punish`, `swarm`, `sniper`, `greedy_city`, `chaos`.

Agents auto-reconnect every 5s for up to 5 minutes (60 attempts). If the server restarts, connected agents will reconnect automatically.

## Remote Server Control (lan_ctl)

Control the running server over TCP without SSH:

```bash
# Single commands
uv run python examples/lan_ctl.py fps 20
uv run python examples/lan_ctl.py truncation 5000
uv run python examples/lan_ctl.py end               # force-end current game
uv run python examples/lan_ctl.py kick 1             # kick player 1
uv run python examples/lan_ctl.py kick 2             # kick player 2
uv run python examples/lan_ctl.py kick all           # kick both → lobby
uv run python examples/lan_ctl.py help

# Interactive mode
uv run python examples/lan_ctl.py
>> fps 30
ok: fps → 30
```

Default: `--host 192.168.0.168 --port 5556`.

Runtime fps/truncation changes take effect next game. They don't survive server restarts — for permanent changes, edit `generals-game.nix`.

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
  --no-spectator         # disable web UI
```

Current production args (in `generals-game.nix`): `--grid 15 --fps 15 --truncation 10000`

## Leaderboard

Persistent file at `~/.generals/leaderboard.json` on the server. Survives server restarts, code deploys, and git operations.

Tracks per agent: wins, losses, draws, games played, win rate, last 1000 game history.

Displayed in spectator UI:
- **Lobby**: Full table with all stats
- **In-game**: Compact sidebar (top-right) with W/L/%
- **Last winner** highlighted in green

## Startup Performance

JAX JIT-compiles functions on first call. The server calls `init_state()` (not `reset()` with a 10K vmap pool) so startup is ~1 min CPU / ~1GB RAM for a 30x30 grid. The compilation happens when the first two agents connect and a game starts — the TCP listener and spectator UI are available immediately.

Subsequent games in the same server session start instantly (compiled code is cached in memory).

If startup seems stuck: check `ps -o etime,%cpu,%mem -C python3` — high CPU with increasing elapsed time means JIT compilation is in progress, not a hang.

## Troubleshooting

### TV shows TTY / black screen
```bash
ssh g "sudo systemctl restart cage-tty1"
```

### Server waiting for players but agents are running
Check network connectivity: `ssh g "ss -tlnp | grep 5555"`. If the port isn't listening, the server is still JIT-compiling — wait for it.

### No output in journald
Python stdout is fully buffered when piped (systemd default). The `PYTHONUNBUFFERED=1` env var in the service config fixes this. If you don't see output, check the config has this set, then `sudo nixos-rebuild switch`.

### Stale browser after UI changes
The server sends `Cache-Control: no-cache` headers. The browser reconnects on WebSocket drop. If still stale: `ssh g "sudo systemctl restart cage-tty1"`.

### Map generation feels repetitive
Seed defaults to `int(time.time())` — restarting the server generates fresh maps. Each game gets a unique map derived from the seed via JAX PRNG splitting.

## Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 5555 | TCP | Agent connections (game protocol) |
| 5556 | TCP | Server control (`lan_ctl`) |
| 8080 | HTTP+WS | Spectator UI (browser on TV) |

All three are open in the NixOS firewall via `generals-game.nix`.
