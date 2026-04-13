# Agent Analysis Runbook

How to evaluate, debug, and iterate on agent behavior.

## Game Sampling Strategy

### Multi-seed statistical runs

A single game is dominated by map layout and spawn position. Never draw conclusions from one game.

- Run **50+ games** per agent pair across different seeds
- Track aggregate **win rate**, **median game length**, and **median land/army at fixed turn milestones** (T25, T50, T100, T200)
- Use the same seed set across agent versions to measure deltas

### Matchup matrix

Don't just test an agent against itself or one opponent.

- Run a **round-robin** across all agent types (GS vs Expander, GS vs Chaos, GS vs MaterialAdvantage, etc.)
- Build a **2D win-rate heatmap** — reveals whether the agent is genuinely strong or just exploiting one opponent's weakness
- Track **win rate by spawn position** (player 0 vs player 1) to detect positional bias

### Turn-bucketed snapshots

Fixed-interval sampling (every 20 turns) misses critical moments. Instead, sample at **game-phase transitions**:

- First territory expansion beyond the general's neighborhood
- First contact with opponent territory
- First city capture (or first failed city attempt)
- First kill attempt (army pushed toward enemy general)
- Army ratio crossing 1.5x in either direction
- Game end

If using fixed intervals for comparison, use T25, T50, T100, T200 — these correspond roughly to opening, early-mid, mid, and late game on a 15x15 grid.

### Replay logs over screenshots

Store the full `GameState` at every turn (or every Nth turn) as serialized JSON via `serialize_game_state()` from `generals/analysis/keyframes.py`. This lets you:

- Rewind and re-render with different visualizations
- Compute derived metrics after the fact without re-running
- Diff states between turns to detect events
- Feed replays into new analysis tools later

For quick iteration, use the ASCII board renderer (render state as text) — it's faster to scan than images and embeds directly into logs.

## Per-Turn Agent Telemetry

Attach to each `act()` call via `get_debug_snapshot()`. The GraphSearchAgent already logs most of these:

| Field | Why it matters |
|---|---|
| **Chosen move kind** (expand/attack/reinforce/scout) | Tracks behavioral phase transitions |
| **Plan mode** (attack/gather/scout/expand/defend) | Shows when the agent's self-assessment changes |
| **Phase and mode** (opening/pressure/kill, ahead/behind/even) | Strategic context for the decision |
| **Top-3 candidate scores with component breakdown** | The score components (base, phase adjustment, path alignment, city investment, etc.) reveal which heuristic is driving decisions. When a bad move wins, you can see exactly which term outbid the right one |
| **Source army size and moving army** | Are we moving big stacks or small ones? |
| **Target and target kind** | What the agent is aiming for |
| **Enemy general estimate** | Is the agent converging on the right location? |
| **Path distance to target** (from BFS distance map) | Is the agent actually making progress turn over turn? |

### Score breakdown is the most important log

When debugging, the single most useful thing is seeing the score breakdown:

```
[12, 2]->[12, 3] (reinforce) score=49.8 base=40.4 phase=0.0 mode=0.0 path=15.0
[12, 2]->[11, 2] (expand)    score=42.0 base=32.0 phase=10.0 mode=0.0 path=0.0
```

This immediately tells you which heuristic is wrong and by how much. Without the component breakdown, you're guessing.

## Per-Turn Game Events

Derive from state diffs between consecutive turns:

| Event | How to detect |
|---|---|
| **Territory gained/lost** | Diff `ownership` arrays between turns |
| **Army destroyed** | Track combat losses: `attacker_army_moved - remaining` |
| **City captured** | Cell changed from neutral/opponent+city to owned+city |
| **General visibility** | `opponent_cells & generals` became true in observation |
| **Fog revealed** | Count of cells where `fog_cells` went from true to false |
| **Army movement** | Diff `armies` array — which cells gained/lost army |

## Aggregate Per-Game Metrics

### Expansion and economy

| Metric | Definition | What it reveals |
|---|---|---|
| **Expansion rate curve** | Land count over time | Is the agent expanding fast enough in the opening? |
| **Time to first city capture** | Turn when first non-general city is owned | Economic development speed |
| **Cities held over time** | Count of owned cities at each turn | Income trajectory |
| **Army production rate** | Delta of total army per turn | Economy health |

### Combat effectiveness

| Metric | Definition | What it reveals |
|---|---|---|
| **Army efficiency** | `enemy_army_destroyed / own_army_produced` | Are armies being used or hoarded? |
| **Kill ratio** | `enemy_army_destroyed / own_army_destroyed` | Combat quality |
| **Time to first enemy contact** | Turn when agent first sees opponent cells | Scouting speed |
| **Time to enemy general discovery** | Turn when enemy general becomes visible | Intel gathering |

### Strategic quality

| Metric | Definition | What it reveals |
|---|---|---|
| **Army concentration index** | Gini coefficient of army distribution across owned cells | High = one big stack (good for attacking), Low = spread thin |
| **Territory compactness** | Ratio of owned cells to convex hull area | Low = scattered outposts, High = dense blob |
| **Frontier ratio** | Frontier cells / total owned cells | How much border are you defending? |
| **General exposure** | BFS distance from own general to nearest enemy cell | How safe is the general? |

## Critical Events to Flag

These indicate bugs or degenerate behavior patterns. Flag them in logs and count per-game:

| Event | Detection rule | Likely cause |
|---|---|---|
| **Oscillation** | Same cell moved to and from within 4 turns | Competing heuristics with similar scores |
| **Idle general** | General army > 8 for 5+ consecutive turns | Missing general evacuation incentive |
| **Failed city attempt** | Moved toward a city, attacked, but didn't capture (army too small) | No army concentration before city assault |
| **Army hoarding** | Any cell with army > 50 not moving for 3+ turns | Gather mode not transitioning to attack |
| **Stagnation** | Land count unchanged for 10+ turns | Agent stuck in reinforce loop |
| **Own-general attack** | Agent moves onto its own general with large army | Bug in general detection (dest_is_general without ownership check) |
| **Suicide scouts** | Army of 1-2 attacks opponent cell with army > 5 | Scouting bonus too high relative to survival |

## Iteration Workflow

1. **Baseline**: Run 50-game matchup against a reference agent (e.g., ExpanderAgent). Record win rate and key metrics.
2. **Identify**: Pick the worst metric or most common critical event.
3. **Diagnose**: Run a single game with full per-turn telemetry. Use ASCII board + score breakdowns to find the exact turn where behavior goes wrong.
4. **Fix**: Adjust the specific scoring term. The score breakdown tells you which component to change and by how much.
5. **Verify**: Re-run the 50-game baseline. Confirm the target metric improved without regressing others.
6. **Deploy**: Push to live server, observe visually for a few games to catch issues the metrics miss.

## Tools Reference

| Tool | Location | Purpose |
|---|---|---|
| `render_state_png()` | `generals/analysis/keyframes.py` | Headless PNG screenshot of game state |
| `serialize_game_state()` | `generals/analysis/keyframes.py` | JSON serialization for replay logs |
| `get_debug_snapshot()` | Agent method | Per-turn decision telemetry |
| `compute_valid_move_mask()` | `generals/core/action.py` | Valid move enumeration |
| ASCII board renderer | Inline (see examples below) | Text-based state visualization |

### ASCII board renderer snippet

```python
import numpy as np

def ascii_board(state):
    armies = np.array(state.armies)
    own0, own1 = np.array(state.ownership[0]), np.array(state.ownership[1])
    generals, cities, mountains = np.array(state.generals), np.array(state.cities), np.array(state.mountains)
    H, W = armies.shape
    lines = []
    for r in range(H):
        row = f"{r:2d}|"
        for c in range(W):
            a = int(armies[r, c])
            if mountains[r, c]:          cell = " ## "
            elif generals[r, c] and own0[r, c]: cell = f"G{a:<3d}"
            elif generals[r, c] and own1[r, c]: cell = f"g{a:<3d}"
            elif cities[r, c] and own0[r, c]:   cell = f"C{a:<3d}"
            elif cities[r, c] and own1[r, c]:   cell = f"c{a:<3d}"
            elif cities[r, c]:           cell = f"${a:<3d}"
            elif own0[r, c]:             cell = f"R{a:<3d}"
            elif own1[r, c]:             cell = f"B{a:<3d}"
            else:                        cell = " .  "
            row += cell
        lines.append(row)
    return "\n".join(lines)
```
