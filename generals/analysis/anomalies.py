from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PlayerHistory:
    moves: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=12))
    max_armies: deque[int] = field(default_factory=lambda: deque(maxlen=12))
    last_detection_turn: dict[str, int] = field(default_factory=dict)


@dataclass
class Detection:
    player: int
    type: str
    severity: int
    score: float
    details: dict[str, Any]
    related_turns: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "player": self.player,
            "type": self.type,
            "severity": self.severity,
            "score": self.score,
            "details": self.details,
        }
        if self.related_turns:
            payload["related_turns"] = self.related_turns
        return payload


class AnomalyEngine:
    def __init__(self):
        self._history = [PlayerHistory(), PlayerHistory()]

    def _can_emit(self, history: PlayerHistory, anomaly_type: str, turn: int, cooldown: int = 0) -> bool:
        last_turn = history.last_detection_turn.get(anomaly_type)
        if last_turn is not None and turn - last_turn < cooldown:
            return False
        history.last_detection_turn[anomaly_type] = turn
        return True

    def detect(
        self,
        turn: int,
        observations: list[Any],
        actions: np.ndarray,
        action_kinds: list[dict[str, Any]],
        prev_state: dict[str, np.ndarray],
        next_state: dict[str, np.ndarray],
        debug_snapshots: list[dict[str, Any] | None],
    ) -> list[dict[str, Any]]:
        detections: list[Detection] = []

        for idx, (obs, action, action_kind, debug) in enumerate(zip(observations, actions, action_kinds, debug_snapshots, strict=True)):
            max_army = int(np.max(np.array(obs.armies) * np.array(obs.owned_cells)))
            history = self._history[idx]
            history.max_armies.append(max_army)

            if bool(action_kind["pass"]) and max_army >= 20 and self._can_emit(history, "large_army_pass", turn, cooldown=6):
                detections.append(Detection(idx, "large_army_pass", 4, 8.0, {"max_owned_army": max_army}))

            if (
                action_kind["kind"] == "attack_city"
                and int(obs.owned_army_count) < int(obs.opponent_army_count)
                and self._can_emit(history, "city_capture_while_behind", turn, cooldown=8)
            ):
                detections.append(
                    Detection(
                        idx,
                        "city_capture_while_behind",
                        3,
                        4.0,
                        {
                            "owned_army_count": int(obs.owned_army_count),
                            "opponent_army_count": int(obs.opponent_army_count),
                        },
                    )
                )

            if (
                (int(obs.timestep) + 1) % 50 == 0
                and action_kind["kind"] in {"pass", "reinforce"}
                and max_army >= 8
                and self._can_emit(history, "income_timing_miss", turn, cooldown=25)
            ):
                detections.append(Detection(idx, "income_timing_miss", 2, 2.0, {"max_owned_army": max_army}))

            if (
                action_kind["kind"] == "reinforce"
                and int(obs.opponent_army_count) > int(obs.owned_army_count) * 1.2
                and self._can_emit(history, "overextension", turn, cooldown=5)
            ):
                detections.append(Detection(idx, "overextension", 2, 2.0, {"owned_land": int(obs.owned_land_count), "opponent_land": int(obs.opponent_land_count)}))

            if (
                action_kind["kind"] == "pass"
                and len(history.max_armies) >= 3
                and min(history.max_armies) >= 15
                and self._can_emit(history, "idle_strike_force", turn, cooldown=10)
            ):
                detections.append(Detection(idx, "idle_strike_force", 3, 5.0, {"window": len(history.max_armies)}))

            if (
                action_kind["kind"] == "reinforce"
                and max_army >= 25
                and int(obs.opponent_land_count) >= int(obs.owned_land_count)
                and self._can_emit(history, "frontline_retreat_without_threat", turn, cooldown=8)
            ):
                detections.append(Detection(idx, "frontline_retreat_without_threat", 2, 2.5, {"max_owned_army": max_army}))

            if (
                action_kind["kind"] == "scout"
                and max_army >= 18
                and debug
                and debug.get("own_general") is not None
            ):
                home = debug["own_general"]
                dest = action_kind.get("dest")
                if dest is not None:
                    dist_from_home = abs(dest[0] - home[0]) + abs(dest[1] - home[1])
                    if dist_from_home >= 8 and self._can_emit(history, "deep_dead_end_entry", turn, cooldown=12):
                        detections.append(Detection(idx, "deep_dead_end_entry", 2, 2.5, {"distance_from_home": dist_from_home}))

            if (
                debug
                and debug.get("enemy_general_estimate") is not None
                and int(obs.opponent_army_count) > int(obs.owned_army_count) * 1.4
            ):
                chosen = debug.get("chosen")
                own_general = debug.get("own_general")
                if chosen and own_general:
                    if (
                        abs(chosen["dest"][0] - own_general[0]) + abs(chosen["dest"][1] - own_general[1]) > 5
                        and self._can_emit(history, "missed_general_threat", turn, cooldown=8)
                    ):
                        detections.append(Detection(idx, "missed_general_threat", 3, 4.0, {"chosen_dest": chosen["dest"]}))

            if (
                debug
                and debug.get("enemy_general_estimate") is not None
                and max_army >= 18
            ):
                chosen = debug.get("chosen")
                if chosen:
                    dest = chosen["dest"]
                    estimate = debug["enemy_general_estimate"]
                    dist = abs(dest[0] - estimate[0]) + abs(dest[1] - estimate[1])
                    candidate_scores = [c["score"] for c in debug.get("top_candidates", [])]
                    if (
                        candidate_scores
                        and chosen["score"] < max(candidate_scores) - 25
                        and self._can_emit(history, "low_confidence_move", turn, cooldown=8)
                    ):
                        detections.append(Detection(idx, "low_confidence_move", 2, 1.5, {"distance_to_estimate": dist}))

            current_move = {
                "turn": turn,
                "source": [int(action[1]), int(action[2])],
                "direction": int(action[3]),
                "kind": action_kind["kind"],
            }
            history.moves.append(current_move)

            if len(history.moves) >= 2:
                prev = history.moves[-2]
                curr = history.moves[-1]
                if prev["source"] == curr["source"] and prev["direction"] != curr["direction"] and curr["kind"] != "pass":
                    if self._can_emit(history, "direction_flip", turn, cooldown=4):
                        detections.append(
                            Detection(
                                idx,
                                "direction_flip",
                                2,
                                2.0,
                                {
                                    "previous_direction": prev["direction"],
                                    "current_direction": curr["direction"],
                                },
                                related_turns=[prev["turn"], curr["turn"]],
                            )
                        )

            if len(history.moves) >= 4:
                recent_sources = [tuple(move["source"]) for move in list(history.moves)[-4:]]
                if len(set(recent_sources)) <= 2 and all(move["kind"] != "pass" for move in list(history.moves)[-4:]):
                    if self._can_emit(history, "repeat_path_loop", turn, cooldown=10):
                        detections.append(
                            Detection(
                                idx,
                                "repeat_path_loop",
                                4,
                                4.0,
                                {"recent_sources": [list(s) for s in recent_sources]},
                                related_turns=[move["turn"] for move in list(history.moves)[-4:]],
                            )
                        )

            if len(history.moves) >= 4:
                recent_kinds = [move["kind"] for move in list(history.moves)[-4:]]
                if (
                    recent_kinds.count("attack") + recent_kinds.count("attack_city") == 0
                    and max_army >= 18
                    and self._can_emit(history, "surround_failure", turn, cooldown=10)
                ):
                    detections.append(Detection(idx, "surround_failure", 2, 1.5, {"recent_kinds": recent_kinds}))

        land_delta = next_state["land"] - prev_state["land"]
        for idx in range(2):
            if abs(int(land_delta[idx])) >= 4:
                detections.append(Detection(idx, "land_swing", 2, 1.0, {"land_delta": int(land_delta[idx])}))
            history = self._history[idx]
            if (
                int(next_state["army"][1 - idx]) - int(prev_state["army"][1 - idx]) >= 8
                and action_kinds[idx]["kind"] == "reinforce"
                and self._can_emit(history, "missed_city_punish", turn, cooldown=10)
            ):
                detections.append(Detection(idx, "missed_city_punish", 2, 2.0, {"opponent_army_gain": int(next_state["army"][1 - idx] - prev_state["army"][1 - idx])}))

        return [d.to_dict() for d in detections]
