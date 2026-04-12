# FFA Plan

This repo is currently a strict 2-player system. Supporting N simultaneous agents in free-for-all mode is a core architecture project, not a small feature.

## Why This Is Large

The current code assumes 2 players in:

- `generals/core/game.py`
- `generals/core/env.py`
- `generals/core/observation.py`
- `generals/lan/`
- `generals/gui/`
- `generals/analysis/`
- examples and tournament tooling

The most important hard-coded assumptions are:

- `ownership` is shaped `(2, H, W)`
- `general_positions` is shaped `(2, 2)`
- `GameInfo.army` and `GameInfo.land` are shaped `(2,)`
- observations expose one opponent aggregate, not many opponents
- rewards and winner logic are 2-player

## Migration Phases

### 1. Core State Generalization

Goal:
- parameterize player count across `GameState`, `GameInfo`, and action handling

Key changes:
- `ownership` becomes `(P, H, W)`
- `general_positions` becomes `(P, 2)`
- `GameInfo.army` and `GameInfo.land` become `(P,)`
- game over state tracks eliminated players and final placement, not just one winner

Acceptance:
- core step logic still works for `P=2`
- tests pass for both `P=2` and small `P>2`

### 2. FFA Combat and Elimination Rules

Goal:
- define how generals are captured and what happens to defeated players

Questions to settle:
- when a general is captured, does the captor inherit all land immediately
- how are ties / simultaneous captures resolved
- what constitutes match end: one survivor or time truncation with ranking

Acceptance:
- elimination semantics are explicit and tested

### 3. Observation Model Redesign

Goal:
- replace the current one-opponent abstraction with multi-opponent visibility

Options:
- per-player opponent channels
- compressed “other players” aggregate plus visible player IDs

Recommendation:
- keep an agent-friendly representation with:
  - owned cells
  - neutral cells
  - visible player-id map
  - per-player stats arrays

Acceptance:
- agents can distinguish opponents rather than treating the whole world as one enemy blob

### 4. Environment and Reward Updates

Goal:
- update `GeneralsEnv` and reward logic for N-player episodes

Key changes:
- actions become `(P, 5)`
- rewards become `(P,)`
- truncation and reset logic handle many players
- ranking-aware rewards or elimination rewards replace win/loss-only reward

Acceptance:
- vectorized env still works
- `P=2` compatibility preserved

### 5. Generator and Spawn Policy

Goal:
- generate FFA maps with realistic spacing and balanced early openings

Key changes:
- spawn selection for `P>2`
- opening-space and city-access fairness across many players
- larger-map defaults for FFA

Acceptance:
- sampled FFA maps avoid immediate spawn crowding
- early reachability is not wildly asymmetric

### 6. LAN / Protocol / Tournament Tooling

Goal:
- let many clients join the same match cleanly

Key changes:
- LAN protocol supports variable player counts
- lobbies and ready-state management support N seats
- tournament runner supports FFA scoring and placement

Acceptance:
- stable N-client local match setup

### 7. Spectator / GUI / Analysis

Goal:
- update all visual and analysis tooling for FFA

Key changes:
- more than two player colors
- scoreboards and elimination displays
- anomaly analysis per player
- tournament summaries based on placement, not only wins

Acceptance:
- local replay/debugging remains usable with 4+ players

## Recommended Order

1. Generalize core state and game logic
2. Redesign observation model
3. Update environment and rewards
4. Add FFA map generation
5. Update LAN and tournament tooling
6. Update spectator and analysis

## Short-Term Guidance

Do not try to “partially” support FFA in the current LAN or GUI layer before the core state model is generalized.

The right approach is:
- preserve a stable 2-player path
- develop FFA on a branch or clearly isolated workstream
- add compatibility tests so `P=2` remains first-class while `P>2` matures
