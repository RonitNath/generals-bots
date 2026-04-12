import json

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from generals.agents import MaterialAdvantageAgent, ScoutPressureAgent
from generals.analysis import MatchLogger, analyze_map_fairness, deserialize_game_state, serialize_game_state
from generals.analysis.anomalies import AnomalyEngine
from generals.core import game
from generals.core.env import GeneralsEnv
from generals.core.observation import Observation
from generals.core.grid import generate_grid


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

    assert report["spawn_distance"] < 10
    assert report["reject_map"] is True
    assert "generals spawn too close for realistic openings" in report["warnings"]


def test_generated_map_preserves_realistic_openings():
    key = jrandom.PRNGKey(7)
    grid = generate_grid(key, grid_dims=(15, 15), pad_to=15, min_generals_distance=10)
    state = game.create_initial_state(grid)
    report = analyze_map_fairness(state)

    generals = np.argwhere(np.array(grid) == 1)[0], np.argwhere(np.array(grid) == 2)[0]
    mountain_mask = np.array(grid) == -2

    for general in generals:
        r, c = map(int, general)
        rr, cc = np.indices(grid.shape)
        local_opening = np.abs(rr - r) + np.abs(cc - c) <= 2
        assert not np.any(mountain_mask & local_opening)

    assert report["spawn_distance"] is not None
    assert report["spawn_distance"] >= 10
    assert min(report["opening_area_within_4"]) >= 20
    assert report["nearest_city_gap"] is None or report["nearest_city_gap"] <= 3


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


def _make_observation(
    *,
    armies,
    owned_cells,
    opponent_cells,
    neutral_cells,
    cities,
    generals,
    owned_land_count,
    owned_army_count,
    opponent_land_count,
    opponent_army_count,
    timestep=40,
):
    shape = np.array(armies).shape
    return Observation(
        armies=jnp.array(armies, dtype=jnp.int32),
        generals=jnp.array(generals, dtype=bool),
        cities=jnp.array(cities, dtype=bool),
        mountains=jnp.zeros(shape, dtype=bool),
        neutral_cells=jnp.array(neutral_cells, dtype=bool),
        owned_cells=jnp.array(owned_cells, dtype=bool),
        opponent_cells=jnp.array(opponent_cells, dtype=bool),
        fog_cells=jnp.zeros(shape, dtype=bool),
        structures_in_fog=jnp.zeros(shape, dtype=bool),
        owned_land_count=jnp.array(owned_land_count, dtype=jnp.int32),
        owned_army_count=jnp.array(owned_army_count, dtype=jnp.int32),
        opponent_land_count=jnp.array(opponent_land_count, dtype=jnp.int32),
        opponent_army_count=jnp.array(opponent_army_count, dtype=jnp.int32),
        timestep=jnp.array(timestep, dtype=jnp.int32),
    )


def test_strategic_agent_debug_includes_mode_and_punish_fields():
    agent = MaterialAdvantageAgent("A")
    armies = np.zeros((5, 5), dtype=np.int32)
    armies[2, 2] = 12
    armies[2, 3] = 2
    owned = np.zeros((5, 5), dtype=bool)
    owned[2, 2] = True
    opponent = np.zeros((5, 5), dtype=bool)
    opponent[2, 3] = True
    neutral = ~(owned | opponent)
    cities = np.zeros((5, 5), dtype=bool)
    cities[2, 3] = True
    generals = np.zeros((5, 5), dtype=bool)
    generals[2, 2] = True
    obs = _make_observation(
        armies=armies,
        owned_cells=owned,
        opponent_cells=opponent,
        neutral_cells=neutral,
        cities=cities,
        generals=generals,
        owned_land_count=8,
        owned_army_count=40,
        opponent_land_count=4,
        opponent_army_count=18,
    )

    action = agent.act(obs, jrandom.PRNGKey(0))
    debug = agent.get_debug_snapshot()

    assert int(action[0]) == 0
    assert debug is not None
    assert debug["mode"] == "ahead"
    assert debug["punish_active"] is True
    assert debug["punish_kind"] == "enemy_city"
    assert debug["punish_reason"] == "visible_enemy_city"
    assert tuple(debug["punish_target"]) == (2, 3)
    assert tuple(debug["strategic_target"]) == (2, 3)
    assert debug["chosen"]["dest"] == [2, 3]
    assert "mode_adjustment" in debug["chosen"]
    assert "punish_adjustment" in debug["chosen"]


def test_strategic_agent_behind_mode_does_not_force_city_punish():
    agent = MaterialAdvantageAgent("A")
    armies = np.zeros((5, 5), dtype=np.int32)
    armies[2, 2] = 8
    armies[2, 3] = 12
    owned = np.zeros((5, 5), dtype=bool)
    owned[2, 2] = True
    opponent = np.zeros((5, 5), dtype=bool)
    opponent[2, 3] = True
    neutral = ~(owned | opponent)
    cities = np.zeros((5, 5), dtype=bool)
    cities[2, 3] = True
    generals = np.zeros((5, 5), dtype=bool)
    generals[2, 2] = True
    obs = _make_observation(
        armies=armies,
        owned_cells=owned,
        opponent_cells=opponent,
        neutral_cells=neutral,
        cities=cities,
        generals=generals,
        owned_land_count=4,
        owned_army_count=18,
        opponent_land_count=8,
        opponent_army_count=40,
    )

    agent.act(obs, jrandom.PRNGKey(1))
    debug = agent.get_debug_snapshot()

    assert debug is not None
    assert debug["mode"] == "behind"
    assert debug["punish_active"] is False
    assert debug["punish_kind"] is None
