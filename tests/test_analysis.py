import json

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from generals.agents import MaterialAdvantageAgent, ScoutPressureAgent
from generals.analysis import MatchLogger, analyze_map_fairness, deserialize_game_state, serialize_game_state
from generals.analysis.anomalies import AnomalyEngine
from generals.core import game
from generals.core.env import GeneralsEnv


def create_symmetric_grid(size=5):
    grid = jnp.zeros((size, size), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[size - 1, size - 1].set(2)
    grid = grid.at[1, 1].set(40)
    grid = grid.at[size - 2, size - 2].set(40)
    return grid


def create_close_spawn_grid():
    grid = jnp.zeros((5, 5), dtype=jnp.int32)
    grid = grid.at[2, 2].set(1)
    grid = grid.at[2, 3].set(2)
    grid = grid.at[0, 0].set(40)
    grid = grid.at[4, 4].set(40)
    return grid


def test_keyframe_roundtrip():
    state = game.create_initial_state(create_symmetric_grid())
    payload = serialize_game_state(state)
    reconstructed = deserialize_game_state(payload)

    assert jnp.array_equal(state.armies, reconstructed.armies)
    assert jnp.array_equal(state.ownership, reconstructed.ownership)
    assert int(state.time) == int(reconstructed.time)
    assert int(state.pool_idx) == int(reconstructed.pool_idx)


def test_map_fairness_report_contains_score():
    state = game.create_initial_state(create_symmetric_grid())
    report = analyze_map_fairness(state)

    assert "fairness_score" in report
    assert "reject_map" in report
    assert report["fairness_score"] >= 0.0


def test_map_fairness_rejects_close_spawns():
    state = game.create_initial_state(create_close_spawn_grid())
    report = analyze_map_fairness(state)

    assert report["spawn_distance"] < 8
    assert report["reject_map"] is True
    assert "generals spawn too close for realistic openings" in report["warnings"]


def test_match_logger_writes_artifacts(tmp_path):
    env = GeneralsEnv(grid_dims=(6, 6), truncation=20, pool_size=8)
    key = jrandom.PRNGKey(0)
    pool, state_before = env.reset(key)
    agent_a = MaterialAdvantageAgent("A")
    agent_b = ScoutPressureAgent("B")

    obs_a = game.get_observation(state_before, 0)
    obs_b = game.get_observation(state_before, 1)
    key, k1, k2 = jrandom.split(key, 3)
    actions = jnp.stack([agent_a.act(obs_a, k1), agent_b.act(obs_b, k2)])
    info_before = game.get_info(state_before)
    timestep, state_after = env.step(state_before, actions, pool)

    logger = MatchLogger(tmp_path / "match", enable_keyframes=False)
    logger.start_game(state_before, [agent_a.id, agent_b.id], seed=0, env_config={"grid": 6, "truncation": 20})
    logger.log_turn(0, state_before, info_before, actions, state_after, timestep.info, [agent_a, agent_b])
    logger.finish_game(int(timestep.info.winner), None, 1, final_state=state_after)

    assert (tmp_path / "match" / "metadata.json").exists()
    assert (tmp_path / "match" / "turns.jsonl").exists()
    assert (tmp_path / "match" / "events.jsonl").exists()
    assert (tmp_path / "match" / "summary.json").exists()

    summary = json.loads((tmp_path / "match" / "summary.json").read_text())
    assert "anomaly_scores" in summary
    assert "worst_turns" in summary


def test_match_logger_does_not_keyframe_on_loop_noise(tmp_path):
    logger = MatchLogger(tmp_path / "match")

    loop_only = [
        {
            "player": 0,
            "type": "repeat_path_loop",
            "severity": 4,
            "score": 6.0,
            "details": {"recent_sources": [[1, 1], [1, 2], [1, 1], [1, 2]]},
        },
        {
            "player": 1,
            "type": "repeat_path_loop",
            "severity": 4,
            "score": 6.0,
            "details": {"recent_sources": [[3, 3], [3, 4], [3, 3], [3, 4]]},
        },
    ]

    assert not logger._should_capture_anomaly_keyframe(10, loop_only, [6.0, 6.0])

    high_signal = [
        {
            "player": 0,
            "type": "missed_general_threat",
            "severity": 3,
            "score": 4.0,
            "details": {"chosen_dest": [5, 5]},
        },
        {
            "player": 0,
            "type": "idle_strike_force",
            "severity": 3,
            "score": 7.0,
            "details": {"window": 5},
        },
    ]

    assert logger._should_capture_anomaly_keyframe(10, high_signal, [11.0, 0.0])
    logger._last_anomaly_keyframe_turn = 10
    assert not logger._should_capture_anomaly_keyframe(15, high_signal, [11.0, 0.0])
    assert logger._should_capture_anomaly_keyframe(18, high_signal, [11.0, 0.0])


def test_repeat_path_loop_has_cooldown():
    engine = AnomalyEngine()
    observations = []
    action_kinds = []
    actions = []
    debug_snapshots = [None, None]
    prev_state = {
        "land": np.array([5, 5]),
        "army": np.array([20, 20]),
    }
    next_state = {
        "land": np.array([5, 5]),
        "army": np.array([20, 20]),
    }

    class StubObs:
        def __init__(self):
            self.armies = np.array([[20]])
            self.owned_cells = np.array([[True]])
            self.opponent_cells = np.array([[False]])
            self.neutral_cells = np.array([[False]])
            self.timestep = 0
            self.owned_army_count = 20
            self.opponent_army_count = 20
            self.owned_land_count = 5
            self.opponent_land_count = 5

    observations = [StubObs(), StubObs()]

    for turn in range(4):
        actions = np.array([[0, 0, 0, turn % 2, 0], [0, 0, 0, turn % 2, 0]])
        action_kinds = [
            {"pass": False, "kind": "reinforce"},
            {"pass": False, "kind": "reinforce"},
        ]
        detections = engine.detect(turn, observations, actions, action_kinds, prev_state, next_state, debug_snapshots)
    assert any(d["type"] == "repeat_path_loop" for d in detections)

    detections = engine.detect(5, observations, np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]), action_kinds, prev_state, next_state, debug_snapshots)
    assert not any(d["type"] == "repeat_path_loop" for d in detections)

    detections = engine.detect(14, observations, np.array([[0, 0, 0, 1, 0], [0, 0, 0, 1, 0]]), action_kinds, prev_state, next_state, debug_snapshots)
    assert any(d["type"] == "repeat_path_loop" for d in detections)
