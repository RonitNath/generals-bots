from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from generals.analysis.anomalies import AnomalyEngine
from generals.analysis.keyframes import render_state_png, write_keyframe_json
from generals.analysis.map_analysis import analyze_map_fairness
from generals.agents.agent import Agent
from generals.core.game import GameInfo, GameState, get_observation


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _classify_action(action: np.ndarray, observation) -> dict[str, Any]:
    pass_turn, row, col, direction, split = [int(x) for x in action.tolist()]
    result = {
        "pass": bool(pass_turn),
        "row": row,
        "col": col,
        "direction": direction,
        "split": bool(split),
        "kind": "pass" if pass_turn else "move",
    }
    if pass_turn:
        return result

    directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)
    dest_row = row + int(directions[direction, 0])
    dest_col = col + int(directions[direction, 1])
    result["dest"] = [dest_row, dest_col]

    if observation.generals[dest_row, dest_col] and observation.opponent_cells[dest_row, dest_col]:
        result["kind"] = "attack_general"
    elif observation.cities[dest_row, dest_col]:
        result["kind"] = "attack_city" if observation.opponent_cells[dest_row, dest_col] or observation.neutral_cells[dest_row, dest_col] else "reinforce_city"
    elif observation.opponent_cells[dest_row, dest_col]:
        result["kind"] = "attack"
    elif observation.neutral_cells[dest_row, dest_col]:
        result["kind"] = "expand"
    elif observation.fog_cells[dest_row, dest_col] or observation.structures_in_fog[dest_row, dest_col]:
        result["kind"] = "scout"
    elif observation.owned_cells[dest_row, dest_col]:
        result["kind"] = "reinforce"
    return result


class MatchLogger:
    """
    Persist map diagnostics, per-turn match logs, event logs, and keyframes.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        keyframe_every: int = 25,
        enable_keyframes: bool = True,
        keyframe_on: set[str] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.turns_path = self.output_dir / "turns.jsonl"
        self.events_path = self.output_dir / "events.jsonl"
        self.metadata_path = self.output_dir / "metadata.json"
        self.summary_path = self.output_dir / "summary.json"
        self.keyframes_dir = self.output_dir / "keyframes"
        self.enable_keyframes = enable_keyframes
        self.keyframe_every = keyframe_every
        self.keyframe_on = keyframe_on or {"anomaly", "city", "general", "periodic", "game_end", "land_swing"}
        self.anomaly_keyframe_threshold = 10.0
        self.anomaly_keyframe_types = {
            "large_army_pass",
            "city_capture_while_behind",
            "idle_strike_force",
            "deep_dead_end_entry",
            "missed_general_threat",
            "low_confidence_move",
            "missed_city_punish",
        }
        self.anomaly_keyframe_cooldown = 8
        if self.enable_keyframes:
            self.keyframes_dir.mkdir(parents=True, exist_ok=True)

        self._metadata_written = False
        self._anomaly_engine = AnomalyEngine()
        self._anomaly_scores = [0.0, 0.0]
        self._anomaly_counts: Counter[str] = Counter()
        self._worst_turns: list[dict[str, Any]] = []
        self._keyframe_count = 0
        self._event_count = 0
        self._general_revealed = [False, False]
        self._last_anomaly_keyframe_turn: int | None = None

    def start_game(self, state: GameState, agent_names: list[str], seed: int | None = None, env_config: dict[str, Any] | None = None):
        metadata = {
            "agents": agent_names,
            "seed": seed,
            "env_config": env_config or {},
            "grid_dims": list(np.array(state.armies).shape),
            "map_fairness": analyze_map_fairness(state),
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2, default=_json_default))
        self.turns_path.write_text("")
        self.events_path.write_text("")
        self._metadata_written = True
        if self.enable_keyframes and "game_start" in self.keyframe_on:
            self._capture_keyframe(0, ["game_start"], state)

    def log_turn(
        self,
        turn: int,
        state_before: GameState,
        info_before: GameInfo,
        actions: np.ndarray,
        state_after: GameState,
        info_after: GameInfo,
        agents: list[Agent],
    ):
        if not self._metadata_written:
            raise RuntimeError("start_game() must be called before log_turn().")

        observations = [get_observation(state_before, 0), get_observation(state_before, 1)]
        actions_np = np.array(actions)
        action_kinds = [_classify_action(action, obs) for action, obs in zip(actions_np, observations, strict=True)]
        debug_snapshots = [agent.get_debug_snapshot() for agent in agents]

        prev_arrays = {
            "land": np.array(info_before.land),
            "army": np.array(info_before.army),
            "cities": np.array(state_before.cities),
            "ownership": np.array(state_before.ownership, dtype=bool),
            "generals": np.array(state_before.generals, dtype=bool),
        }
        next_arrays = {
            "land": np.array(info_after.land),
            "army": np.array(info_after.army),
            "cities": np.array(state_after.cities),
            "ownership": np.array(state_after.ownership, dtype=bool),
            "generals": np.array(state_after.generals, dtype=bool),
        }

        anomalies = self._anomaly_engine.detect(
            turn,
            observations,
            actions_np,
            action_kinds,
            prev_arrays,
            next_arrays,
            debug_snapshots,
        )
        anomaly_score_this_turn = [0.0, 0.0]
        for anomaly in anomalies:
            anomaly_score_this_turn[anomaly["player"]] += anomaly["score"]
            self._anomaly_scores[anomaly["player"]] += anomaly["score"]
            self._anomaly_counts[anomaly["type"]] += 1

        events = self._derive_events(turn, state_before, state_after, info_before, info_after, observations, anomalies)
        for event in events:
            self._write_jsonl(self.events_path, event)
            self._event_count += 1

        reasons: list[str] = []
        if self.keyframe_every > 0 and turn > 0 and turn % self.keyframe_every == 0 and "periodic" in self.keyframe_on:
            reasons.append("periodic")
        if any(event["type"] == "city_capture" for event in events) and "city" in self.keyframe_on:
            reasons.append("city_capture")
        if any(event["type"] == "general_reveal" for event in events) and "general" in self.keyframe_on:
            reasons.append("general_reveal")
        if any(event["type"] == "land_swing" for event in events) and "land_swing" in self.keyframe_on:
            reasons.append("land_swing")
        if self._should_capture_anomaly_keyframe(turn, anomalies, anomaly_score_this_turn) and "anomaly" in self.keyframe_on:
            reasons.append("anomaly")
        if reasons and self.enable_keyframes:
            self._capture_keyframe(turn, reasons, state_after)

        turn_record = {
            "turn": turn,
            "pre_state": {
                "army": np.array(info_before.army),
                "land": np.array(info_before.land),
                "winner": int(info_before.winner),
                "time": int(info_before.time),
            },
            "post_state": {
                "army": np.array(info_after.army),
                "land": np.array(info_after.land),
                "winner": int(info_after.winner),
                "time": int(info_after.time),
            },
            "players": [],
            "anomalies": anomalies,
            "turn_anomaly_score": anomaly_score_this_turn,
        }

        for idx, (agent, obs, action_kind, debug) in enumerate(zip(agents, observations, action_kinds, debug_snapshots, strict=True)):
            turn_record["players"].append(
                {
                    "index": idx,
                    "agent": agent.id,
                    "owned_army_count": int(obs.owned_army_count),
                    "owned_land_count": int(obs.owned_land_count),
                    "opponent_army_count": int(obs.opponent_army_count),
                    "opponent_land_count": int(obs.opponent_land_count),
                    "action": action_kind,
                    "debug": debug,
                }
            )

        self._write_jsonl(self.turns_path, turn_record)
        worst_turn_score = sum(anomaly_score_this_turn)
        if worst_turn_score > 0:
            self._worst_turns.append({"turn": turn, "score": worst_turn_score, "anomalies": anomalies})
            self._worst_turns = sorted(self._worst_turns, key=lambda x: x["score"], reverse=True)[:10]

    def finish_game(self, winner: int, winner_name: str | None, turns: int, final_state: GameState | None = None):
        if final_state is not None and self.enable_keyframes and "game_end" in self.keyframe_on:
            self._capture_keyframe(turns, ["game_end"], final_state)

        summary = {
            "winner": winner,
            "winner_name": winner_name,
            "turns": turns,
            "anomaly_scores": self._anomaly_scores,
            "anomaly_counts": dict(self._anomaly_counts),
            "worst_turns": self._worst_turns,
            "keyframes": self._keyframe_count,
            "events": self._event_count,
        }
        self.summary_path.write_text(json.dumps(summary, indent=2, default=_json_default))

    def _derive_events(
        self,
        turn: int,
        state_before: GameState,
        state_after: GameState,
        info_before: GameInfo,
        info_after: GameInfo,
        observations: list[Any],
        anomalies: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        ownership_before = np.array(state_before.ownership, dtype=bool)
        ownership_after = np.array(state_after.ownership, dtype=bool)
        cities = np.array(state_after.cities, dtype=bool)
        generals = np.array(state_after.generals, dtype=bool)

        for player in range(2):
            captured_cities = np.argwhere(cities & ~ownership_before[player] & ownership_after[player])
            for row, col in captured_cities.tolist():
                events.append({"turn": turn, "type": "city_capture", "player": player, "pos": [row, col]})

            land_delta = int(np.array(info_after.land)[player] - np.array(info_before.land)[player])
            if abs(land_delta) >= 4:
                events.append({"turn": turn, "type": "land_swing", "player": player, "land_delta": land_delta})

        for player, obs in enumerate(observations):
            enemy_general_visible = bool(np.any(np.array(obs.generals) & np.array(obs.opponent_cells)))
            if enemy_general_visible and not self._general_revealed[player]:
                positions = np.argwhere(np.array(obs.generals) & np.array(obs.opponent_cells))
                pos = positions[0].tolist() if len(positions) else None
                events.append({"turn": turn, "type": "general_reveal", "player": player, "pos": pos})
                self._general_revealed[player] = True

        for anomaly in anomalies:
            events.append({"turn": turn, "type": "anomaly", "player": anomaly["player"], "anomaly": anomaly["type"], "score": anomaly["score"]})
        return events

    def _capture_keyframe(self, turn: int, reasons: list[str], state: GameState):
        tag = "__".join(dict.fromkeys(reasons))
        stem = f"{turn:04d}__{tag}"
        write_keyframe_json(self.keyframes_dir / f"{stem}.json", state, reasons)
        render_state_png(self.keyframes_dir / f"{stem}.png", state)
        self._keyframe_count += 1
        if "anomaly" in reasons:
            self._last_anomaly_keyframe_turn = turn
        self._write_jsonl(self.events_path, {"turn": turn, "type": "keyframe", "reasons": reasons, "stem": stem})
        self._event_count += 1

    def _should_capture_anomaly_keyframe(
        self,
        turn: int,
        anomalies: list[dict[str, Any]],
        anomaly_score_this_turn: list[float],
    ) -> bool:
        if not anomalies:
            return False
        if sum(anomaly_score_this_turn) < self.anomaly_keyframe_threshold:
            return False
        if self._last_anomaly_keyframe_turn is not None:
            if turn - self._last_anomaly_keyframe_turn < self.anomaly_keyframe_cooldown:
                return False
        return any(anomaly["type"] in self.anomaly_keyframe_types for anomaly in anomalies)

    def _write_jsonl(self, path: Path, payload: dict[str, Any]):
        with path.open("a") as f:
            f.write(json.dumps(payload, default=_json_default) + "\n")
