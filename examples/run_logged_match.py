"""
Run a local match, log structured artifacts, and optionally show the pygame GUI.

Examples:
    uv run python examples/run_logged_match.py --agent-a material --agent-b backdoor --gui
    uv run python examples/run_logged_match.py --agent-a material --agent-b scout --output logs/material-vs-scout
"""

import argparse

import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv, get_observation
from generals.agents import (
    BackdoorAgent,
    DefenseCounterAgent,
    ExpanderAgent,
    MaterialAdvantageAgent,
    RandomAgent,
    ScoutPressureAgent,
    SurroundPressureAgent,
    build_agent,
)
from generals.analysis import MatchLogger, analyze_map_fairness
from generals.core.game import get_info


def build_builtin_agent(name: str, display_name: str | None):
    if name == "random":
        return RandomAgent(id=display_name or "Random")
    if name == "expander":
        return ExpanderAgent(id=display_name or "Expander")
    if name == "material":
        return MaterialAdvantageAgent(id=display_name or "Material")
    if name == "scout":
        return ScoutPressureAgent(id=display_name or "Scout")
    if name == "backdoor":
        return BackdoorAgent(id=display_name or "Backdoor")
    if name == "defense":
        return DefenseCounterAgent(id=display_name or "DefenseCounter")
    if name == "surround":
        return SurroundPressureAgent(id=display_name or "SurroundPressure")
    raise ValueError(f"Unknown built-in agent: {name}")


parser = argparse.ArgumentParser(description="Run and log a local Generals match")
parser.add_argument("--grid", type=int, default=15)
parser.add_argument("--truncation", type=int, default=400)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gui", action="store_true", help="Show the pygame GUI while logging")
parser.add_argument("--fps", type=int, default=6)
parser.add_argument("--output", type=str, default="logs/latest-match")
parser.add_argument("--keyframe-every", type=int, default=25)
parser.add_argument("--no-keyframes", action="store_true")
parser.add_argument("--keyframe-on", type=str, default="game_start,periodic,city,general,land_swing,anomaly,game_end")
parser.add_argument("--min-fairness-score", type=float, default=0.65)
parser.add_argument("--max-map-rerolls", type=int, default=20)
parser.add_argument("--min-spawn-distance", type=int, default=8)
parser.add_argument("--min-opening-area", type=int, default=18)

choices = ["random", "expander", "material", "scout", "backdoor", "defense", "surround"]
parser.add_argument("--agent-a", type=str, default="material", choices=choices)
parser.add_argument("--agent-b", type=str, default="backdoor", choices=choices)
parser.add_argument("--agent-a-custom", type=str, default=None)
parser.add_argument("--agent-b-custom", type=str, default=None)
parser.add_argument("--name-a", type=str, default=None)
parser.add_argument("--name-b", type=str, default=None)
args = parser.parse_args()

agent_a = build_agent(args.agent_a_custom, args.name_a) if args.agent_a_custom else build_builtin_agent(args.agent_a, args.name_a)
agent_b = build_agent(args.agent_b_custom, args.name_b) if args.agent_b_custom else build_builtin_agent(args.agent_b, args.name_b)

gui = None
if args.gui:
    from generals.gui import ReplayGUI

env = GeneralsEnv(
    grid_dims=(args.grid, args.grid),
    truncation=args.truncation,
    min_generals_distance=args.min_spawn_distance,
)
key = jrandom.PRNGKey(args.seed)
selected_attempt = 0
fairness_report = None
map_accepted = False
for attempt in range(args.max_map_rerolls + 1):
    key, reset_key = jrandom.split(key)
    pool, state = env.reset(reset_key)
    fairness_report = analyze_map_fairness(state)
    selected_attempt = attempt
    spawn_distance = fairness_report.get("spawn_distance")
    opening_area = fairness_report.get("opening_area_within_4", [0, 0])
    map_accepted = (
        fairness_report["fairness_score"] >= args.min_fairness_score
        and not fairness_report["reject_map"]
        and (spawn_distance is None or spawn_distance >= args.min_spawn_distance)
        and min(opening_area) >= args.min_opening_area
    )
    if map_accepted:
        break

logger = MatchLogger(
    args.output,
    keyframe_every=args.keyframe_every,
    enable_keyframes=not args.no_keyframes,
    keyframe_on={part.strip() for part in args.keyframe_on.split(",") if part.strip()},
)
logger.start_game(
    state,
    [agent_a.id, agent_b.id],
    seed=args.seed,
    env_config={
        "grid": args.grid,
        "truncation": args.truncation,
        "min_fairness_score": args.min_fairness_score,
        "max_map_rerolls": args.max_map_rerolls,
        "min_spawn_distance": args.min_spawn_distance,
        "min_opening_area": args.min_opening_area,
        "generator_min_generals_distance": args.min_spawn_distance,
        "map_reroll_attempts": selected_attempt,
        "map_accepted": map_accepted,
    },
)

if args.gui:
    gui = ReplayGUI(state, agent_ids=[agent_a.id, agent_b.id], fps=args.fps)

agent_a.reset()
agent_b.reset()
terminated = truncated = False
turn = 0

while not (terminated or truncated):
    state_before = state
    obs_a = get_observation(state_before, 0)
    obs_b = get_observation(state_before, 1)

    key, k1, k2 = jrandom.split(key, 3)
    actions = jnp.stack([
        agent_a.act(obs_a, k1),
        agent_b.act(obs_b, k2),
    ])

    info_before = get_info(state_before)
    timestep, state = env.step(state_before, actions, pool)
    logger.log_turn(
        turn,
        state_before=state_before,
        info_before=info_before,
        actions=actions,
        state_after=state,
        info_after=timestep.info,
        agents=[agent_a, agent_b],
    )

    if gui is not None:
        gui.update(state, timestep.info)
        gui.tick(fps=args.fps)

    terminated = bool(timestep.terminated)
    truncated = bool(timestep.truncated)
    turn += 1

winner = int(timestep.info.winner)
winner_name = [agent_a.id, agent_b.id][winner] if winner >= 0 else None
logger.finish_game(winner, winner_name, turn, final_state=state)

if gui is not None:
    gui.close()

print(f"Logged match to {args.output}")
print(
    "Map fairness: "
    f"{fairness_report['fairness_score']:.3f} "
    f"({'accepted' if map_accepted else 'fallback'}) after {selected_attempt} rerolls; "
    f"spawn_distance={fairness_report.get('spawn_distance')}, "
    f"opening_area={fairness_report.get('opening_area_within_4')}"
)
print(f"Winner: {winner_name if winner_name is not None else 'draw'} in {turn} turns")
