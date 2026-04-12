# Project Overview

This repo is a Generals-style simulator and bot arena built around a JAX game core.

The important split is:

- `generals/core/`: authoritative rules, state, map generation, observation model
- `generals/lan/`: run matches with a central server and remote laptop agents
- `generals/agents/`: agent interface, built-in baselines, heuristic strategy agents
- `generals/analysis/`: fairness checks, match logs, anomalies, keyframes, replay reconstruction
- `generals/spectator/`: browser spectator UI for the TV/server flow

Recommended workflow:

1. Write or tune an agent locally.
2. Run logged matches with `examples/run_logged_match.py`.
3. Inspect fairness warnings, anomalies, and keyframes.
4. Move the agent into LAN play against another laptop bot.

The repo is no longer just “an RL simulator package.” It is a shared bot-development platform with:

- a deterministic local engine
- LAN competition support
- match artifact generation for debugging
- a growing strategy roster for sparring and benchmarking
