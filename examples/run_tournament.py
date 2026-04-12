"""
Run a batch tournament across seeds and aggregate results.

Examples:
    uv run python examples/run_tournament.py --agents material surround scout --seeds 10 11 12
    uv run python examples/run_tournament.py --agents material surround --round-robin --output logs/tournament-latest
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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
)
from generals.analysis import MatchLogger, analyze_map_fairness
from generals.core.game import get_info


def build_builtin_agent(name: str, display_name: str | None = None):
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


def run_match(
    agent_a_name: str,
    agent_b_name: str,
    seed: int,
    output_dir: Path,
    *,
    grid: int,
    truncation: int,
    keyframe_every: int,
    min_fairness_score: float,
    max_map_rerolls: int,
    min_spawn_distance: int,
    min_opening_area: int,
) -> dict[str, Any]:
    agent_a = build_builtin_agent(agent_a_name, agent_a_name.capitalize())
    agent_b = build_builtin_agent(agent_b_name, agent_b_name.capitalize())

    env = GeneralsEnv(
        grid_dims=(grid, grid),
        truncation=truncation,
        min_generals_distance=min_spawn_distance,
    )
    key = jrandom.PRNGKey(seed)
    selected_attempt = 0
    fairness_report = None
    map_accepted = False
    for attempt in range(max_map_rerolls + 1):
        key, reset_key = jrandom.split(key)
        pool, state = env.reset(reset_key)
        fairness_report = analyze_map_fairness(state)
        selected_attempt = attempt
        spawn_distance = fairness_report.get("spawn_distance")
        opening_area = fairness_report.get("opening_area_within_4", [0, 0])
        map_accepted = (
            fairness_report["fairness_score"] >= min_fairness_score
            and not fairness_report["reject_map"]
            and (spawn_distance is None or spawn_distance >= min_spawn_distance)
            and min(opening_area) >= min_opening_area
        )
        if map_accepted:
            break

    logger = MatchLogger(output_dir, keyframe_every=keyframe_every)
    logger.start_game(
        state,
        [agent_a.id, agent_b.id],
        seed=seed,
        env_config={
            "grid": grid,
            "truncation": truncation,
            "min_fairness_score": min_fairness_score,
            "max_map_rerolls": max_map_rerolls,
            "min_spawn_distance": min_spawn_distance,
            "min_opening_area": min_opening_area,
            "generator_min_generals_distance": min_spawn_distance,
            "map_reroll_attempts": selected_attempt,
            "map_accepted": map_accepted,
        },
    )

    agent_a.reset()
    agent_b.reset()
    terminated = truncated = False
    turn = 0

    while not (terminated or truncated):
        state_before = state
        obs_a = get_observation(state_before, 0)
        obs_b = get_observation(state_before, 1)
        key, k1, k2 = jrandom.split(key, 3)
        actions = jnp.stack([agent_a.act(obs_a, k1), agent_b.act(obs_b, k2)])
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
        terminated = bool(timestep.terminated)
        truncated = bool(timestep.truncated)
        turn += 1

    winner = int(timestep.info.winner)
    winner_name = [agent_a.id, agent_b.id][winner] if winner >= 0 else None
    logger.finish_game(winner, winner_name, turn, final_state=state)

    summary = json.loads((output_dir / "summary.json").read_text())
    return {
        "seed": seed,
        "agents": [agent_a_name, agent_b_name],
        "winner": winner,
        "winner_name": winner_name,
        "turns": turn,
        "map_accepted": map_accepted,
        "map_reroll_attempts": selected_attempt,
        "fairness_score": fairness_report["fairness_score"],
        "spawn_distance": fairness_report.get("spawn_distance"),
        "opening_area_within_4": fairness_report.get("opening_area_within_4"),
        "summary": summary,
    }


def pairings_from_args(agents: list[str], round_robin: bool) -> list[tuple[str, str]]:
    if round_robin:
        return list(itertools.combinations(agents, 2))
    if len(agents) != 2:
        raise ValueError("Without --round-robin, exactly two agents are required.")
    return [(agents[0], agents[1])]


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    per_pair: dict[str, dict[str, Any]] = {}
    global_anomalies: Counter[str] = Counter()

    for result in results:
        a, b = result["agents"]
        pair_key = f"{a}_vs_{b}"
        pair = per_pair.setdefault(
            pair_key,
            {
                "agents": [a, b],
                "matches": 0,
                "wins": {a: 0, b: 0},
                "draws": 0,
                "accepted_maps": 0,
                "avg_turns": 0.0,
                "avg_fairness_score": 0.0,
                "avg_rerolls": 0.0,
                "anomaly_counts": Counter(),
            },
        )
        pair["matches"] += 1
        pair["avg_turns"] += result["turns"]
        pair["avg_fairness_score"] += result["fairness_score"]
        pair["avg_rerolls"] += result["map_reroll_attempts"]
        if result["map_accepted"]:
            pair["accepted_maps"] += 1

        if result["winner"] == 0:
            pair["wins"][a] += 1
        elif result["winner"] == 1:
            pair["wins"][b] += 1
        else:
            pair["draws"] += 1

        anomalies = Counter(result["summary"].get("anomaly_counts", {}))
        pair["anomaly_counts"].update(anomalies)
        global_anomalies.update(anomalies)

    for pair in per_pair.values():
        matches = max(pair["matches"], 1)
        pair["avg_turns"] /= matches
        pair["avg_fairness_score"] /= matches
        pair["avg_rerolls"] /= matches
        pair["anomaly_counts"] = dict(pair["anomaly_counts"])

    return {
        "matches": len(results),
        "pairings": per_pair,
        "global_anomaly_counts": dict(global_anomalies),
    }


def main():
    parser = argparse.ArgumentParser(description="Run a tournament across seeds and aggregate results")
    parser.add_argument("--agents", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45])
    parser.add_argument("--round-robin", action="store_true")
    parser.add_argument("--output", type=str, default="logs/tournament-latest")
    parser.add_argument("--grid", type=int, default=15)
    parser.add_argument("--truncation", type=int, default=500)
    parser.add_argument("--keyframe-every", type=int, default=25)
    parser.add_argument("--min-fairness-score", type=float, default=0.72)
    parser.add_argument("--max-map-rerolls", type=int, default=8)
    parser.add_argument("--min-spawn-distance", type=int, default=10)
    parser.add_argument("--min-opening-area", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    matches_dir = output_dir / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)

    pairings = pairings_from_args(args.agents, args.round_robin)
    results = []

    for agent_a, agent_b in pairings:
        for seed in args.seeds:
            match_dir = matches_dir / f"{agent_a}_vs_{agent_b}__seed_{seed}"
            result = run_match(
                agent_a,
                agent_b,
                seed,
                match_dir,
                grid=args.grid,
                truncation=args.truncation,
                keyframe_every=args.keyframe_every,
                min_fairness_score=args.min_fairness_score,
                max_map_rerolls=args.max_map_rerolls,
                min_spawn_distance=args.min_spawn_distance,
                min_opening_area=args.min_opening_area,
            )
            results.append(result)
            print(
                f"{agent_a} vs {agent_b} seed={seed}: "
                f"{result['winner_name'] if result['winner_name'] is not None else 'draw'} "
                f"in {result['turns']} turns; fairness={result['fairness_score']:.3f}; "
                f"accepted={result['map_accepted']}"
            )

    aggregate = aggregate_results(results)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2))
    print(f"Wrote tournament summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
