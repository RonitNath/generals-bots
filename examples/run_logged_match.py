"""
Run a local match, log structured artifacts, and optionally show the pygame GUI.

Examples:
    uv run python examples/run_logged_match.py --agent-a material --agent-b backdoor --gui
    uv run python examples/run_logged_match.py --agent-a material --agent-b scout --output logs/material-vs-scout
"""

import argparse
import json
import time

import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv, get_observation
from generals.agents import (
    BackdoorAgent,
    ChaosAgent,
    DefenseCounterAgent,
    ExpanderAgent,
    GreedyCityAgent,
    MaterialAdvantageAgent,
    PunishAgent,
    RandomAgent,
    ScoutPressureAgent,
    SniperAgent,
    SurroundPressureAgent,
    SwarmAgent,
    TurtleAgent,
    build_agent,
)
from generals.analysis import MatchLogger, Telemetry, analyze_map_fairness
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
    if name == "turtle":
        return TurtleAgent(id=display_name or "Turtle")
    if name == "punish":
        return PunishAgent(id=display_name or "Punish")
    if name == "swarm":
        return SwarmAgent(id=display_name or "Swarm")
    if name == "sniper":
        return SniperAgent(id=display_name or "Sniper")
    if name == "greedy_city":
        return GreedyCityAgent(id=display_name or "GreedyCity")
    if name == "chaos":
        return ChaosAgent(id=display_name or "Chaos")
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
parser.add_argument("--min-fairness-score", type=float, default=0.72)
parser.add_argument("--max-map-rerolls", type=int, default=8)
parser.add_argument("--min-spawn-distance", type=int, default=10)
parser.add_argument("--min-opening-area", type=int, default=20)
parser.add_argument("--pool-size", type=int, default=32)
parser.add_argument("--spawn-candidates", type=int, default=3)
parser.add_argument("--terrain-candidates", type=int, default=4)

choices = ["random", "expander", "material", "scout", "backdoor", "defense", "surround", "turtle", "punish", "swarm", "sniper", "greedy_city", "chaos"]
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
    pool_size=args.pool_size,
    spawn_candidate_count=args.spawn_candidates,
    terrain_candidate_count=args.terrain_candidates,
)
key = jrandom.PRNGKey(args.seed)
telemetry = Telemetry()
match_start = time.perf_counter()
selected_attempt = 0
fairness_report = None
map_accepted = False
for attempt in range(args.max_map_rerolls + 1):
    key, reset_key = jrandom.split(key)
    attempt_start = time.perf_counter()
    pool, state = telemetry.time_block("reset.env_reset", lambda: env.reset(reset_key))
    telemetry.time_block("reset.block_state", lambda: None, ready_value=state.armies)
    fairness_report = telemetry.time_block("reset.fairness_analysis", lambda: analyze_map_fairness(state))
    selected_attempt = attempt
    spawn_distance = fairness_report.get("spawn_distance")
    opening_area = fairness_report.get("opening_area_within_4", [0, 0])
    map_accepted = (
        fairness_report["fairness_score"] >= args.min_fairness_score
        and not fairness_report["reject_map"]
        and (spawn_distance is None or spawn_distance >= args.min_spawn_distance)
        and min(opening_area) >= args.min_opening_area
    )
    telemetry.record("reset.attempt_total", time.perf_counter() - attempt_start)
    telemetry.add_sample(
        "reset_attempts",
        {
            "attempt": attempt,
            "fairness_score": fairness_report["fairness_score"],
            "spawn_distance": spawn_distance,
            "opening_area": opening_area,
            "accepted": map_accepted,
        },
    )
    if map_accepted:
        break

logger = MatchLogger(
    args.output,
    keyframe_every=args.keyframe_every,
    enable_keyframes=not args.no_keyframes,
    keyframe_on={part.strip() for part in args.keyframe_on.split(",") if part.strip()},
)
telemetry.time_block(
    "logger.start_game",
    lambda: logger.start_game(
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
            "pool_size": args.pool_size,
            "spawn_candidate_count": args.spawn_candidates,
            "terrain_candidate_count": args.terrain_candidates,
            "map_reroll_attempts": selected_attempt,
            "map_accepted": map_accepted,
        },
    ),
)

if args.gui:
    gui = ReplayGUI(state, agent_ids=[agent_a.id, agent_b.id], fps=args.fps)

agent_a.reset()
agent_b.reset()
terminated = truncated = False
turn = 0

while not (terminated or truncated):
    turn_start = time.perf_counter()
    state_before = state
    obs_a = telemetry.time_block("turn.observe_a", lambda: get_observation(state_before, 0))
    obs_b = telemetry.time_block("turn.observe_b", lambda: get_observation(state_before, 1))

    key, k1, k2 = jrandom.split(key, 3)
    action_a = telemetry.time_block("turn.act_a", lambda: agent_a.act(obs_a, k1))
    action_b = telemetry.time_block("turn.act_b", lambda: agent_b.act(obs_b, k2))
    actions = telemetry.time_block("turn.stack_actions", lambda: jnp.stack([action_a, action_b]))

    info_before = telemetry.time_block("turn.get_info_before", lambda: get_info(state_before))
    timestep, state = telemetry.time_block("turn.env_step", lambda: env.step(state_before, actions, pool))
    telemetry.time_block("turn.block_step", lambda: None, ready_value=state.armies)
    telemetry.time_block(
        "turn.logger_log_turn",
        lambda: logger.log_turn(
            turn,
            state_before=state_before,
            info_before=info_before,
            actions=actions,
            state_after=state,
            info_after=timestep.info,
            agents=[agent_a, agent_b],
        ),
    )

    if gui is not None:
        telemetry.time_block("turn.gui_update", lambda: gui.update(state, timestep.info))
        telemetry.time_block("turn.gui_tick", lambda: gui.tick(fps=args.fps))

    terminated = bool(timestep.terminated)
    truncated = bool(timestep.truncated)
    turn_duration = time.perf_counter() - turn_start
    telemetry.record("turn.total", turn_duration)
    telemetry.add_sample(
        "turn_samples",
        {
            "turn": turn,
            "duration_sec": turn_duration,
            "terminated": terminated,
            "truncated": truncated,
        },
    )
    turn += 1

winner = int(timestep.info.winner)
winner_name = [agent_a.id, agent_b.id][winner] if winner >= 0 else None
telemetry.time_block("logger.finish_game", lambda: logger.finish_game(winner, winner_name, turn, final_state=state))
telemetry.record("match.total", time.perf_counter() - match_start)
telemetry.merge(logger.telemetry, prefix="logger")

if gui is not None:
    telemetry.time_block("gui.close", lambda: gui.close())

summary_path = logger.summary_path
summary = json.loads(summary_path.read_text())
summary["runner_telemetry"] = telemetry.snapshot()
summary_path.write_text(json.dumps(summary, indent=2))

print(f"Logged match to {args.output}")
print(
    "Map fairness: "
    f"{fairness_report['fairness_score']:.3f} "
    f"({'accepted' if map_accepted else 'fallback'}) after {selected_attempt} rerolls; "
    f"spawn_distance={fairness_report.get('spawn_distance')}, "
    f"opening_area={fairness_report.get('opening_area_within_4')}"
)
print(f"Winner: {winner_name if winner_name is not None else 'draw'} in {turn} turns")
