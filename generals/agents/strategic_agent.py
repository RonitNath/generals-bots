from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from generals.core.action import compute_valid_move_mask
from generals.core.observation import Observation

from .agent import Agent


@dataclass(frozen=True)
class MoveFeatures:
    row: int
    col: int
    direction: int
    dest_row: int
    dest_col: int
    source_army: int
    dest_army: int
    moving_army: int
    can_capture: bool
    dest_is_owned: bool
    dest_is_opponent: bool
    dest_is_neutral: bool
    dest_is_city: bool
    dest_is_general: bool
    dest_is_fog: bool
    dest_is_structure_fog: bool


class StrategicAgent(Agent):
    """
    Base class for stateful heuristic agents.

    These agents keep lightweight memory across turns because the game is partially
    observable and the paper emphasizes the value of memory and inference.
    """

    def __init__(self, id: str = "Strategic"):
        super().__init__(id)
        self.reset()

    def reset(self):
        self._seen = None
        self._last_seen_opponent = None
        self._last_seen_generals = None
        self._last_seen_cities = None
        self._own_general = None
        self._enemy_general_estimate = None
        self._current_target = None
        self._target_kind = None
        self._target_commitment = 0
        self._phase = "opening"
        self._mode = "even"
        self._mode_metrics = {}
        self._punish_target = None
        self._punish_kind = None
        self._punish_reason = None
        self._punish_strength = 0.0
        self._punish_window = 0
        self._last_debug = None
        self._recent_choices = deque(maxlen=8)
        self._recent_action_kinds = deque(maxlen=8)

    def act(self, observation: Observation, key: jnp.ndarray) -> jnp.ndarray:
        self._update_memory(observation)
        self._phase = self._infer_phase(observation)
        self._mode = self._infer_mode(observation)
        self._update_punish_window(observation)
        self._update_strategy_target(observation)
        moves = self._extract_moves(observation)
        if not moves:
            self._last_debug = {
                "decision": "pass",
                "reason": "no_valid_moves",
                "phase": self._phase,
                "mode": self._mode,
                "mode_metrics": self._mode_metrics,
                "enemy_general_estimate": self._enemy_general_estimate,
                "strategic_target": self._current_target,
                "target_kind": self._target_kind,
                "punish_active": bool(self._punish_window > 0 and self._punish_target is not None),
                "punish_target": self._punish_target,
                "punish_kind": self._punish_kind,
                "punish_reason": self._punish_reason,
                "punish_window": int(self._punish_window),
            }
            return jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)

        base_scores = np.array([self.score_move(observation, move) for move in moves], dtype=np.float32)
        phase_scores = np.array([self._phase_adjustment(observation, move) for move in moves], dtype=np.float32)
        mode_scores = np.array([self._mode_adjustment(observation, move) for move in moves], dtype=np.float32)
        punish_scores = np.array([self._punish_adjustment(observation, move) for move in moves], dtype=np.float32)
        continuation_scores = np.array([self._continuation_value(observation, move) for move in moves], dtype=np.float32)
        adjustments = np.array([self._score_adjustment(observation, move) for move in moves], dtype=np.float32)
        noise = np.array(jrandom.uniform(key, shape=(len(moves),), minval=-0.15, maxval=0.15), dtype=np.float32)
        scores = base_scores + phase_scores + mode_scores + punish_scores + continuation_scores + adjustments + noise
        best_idx = int(np.argmax(scores))
        best = moves[best_idx]
        split = self.choose_split(observation, best)
        chosen_kind = self._classify_move(best)
        self._remember_choice(best, chosen_kind)
        top_order = np.argsort(scores)[::-1][: min(3, len(scores))]
        self._last_debug = {
            "decision": "move",
            "phase": self._phase,
            "mode": self._mode,
            "mode_metrics": self._mode_metrics,
            "enemy_general_estimate": self._enemy_general_estimate,
            "own_general": self._own_general,
            "strategic_target": self._current_target,
            "target_kind": self._target_kind,
            "target_commitment": int(self._target_commitment),
            "punish_active": bool(self._punish_window > 0 and self._punish_target is not None),
            "punish_target": self._punish_target,
            "punish_kind": self._punish_kind,
            "punish_reason": self._punish_reason,
            "punish_window": int(self._punish_window),
            "top_candidates": [
                {
                    "source": [moves[i].row, moves[i].col],
                    "direction": moves[i].direction,
                    "dest": [moves[i].dest_row, moves[i].dest_col],
                    "base_score": float(base_scores[i]),
                    "phase_adjustment": float(phase_scores[i]),
                    "mode_adjustment": float(mode_scores[i]),
                    "punish_adjustment": float(punish_scores[i]),
                    "continuation": float(continuation_scores[i]),
                    "adjustment": float(adjustments[i]),
                    "score": float(scores[i]),
                    "capture": bool(moves[i].can_capture),
                    "city": bool(moves[i].dest_is_city),
                    "opponent": bool(moves[i].dest_is_opponent),
                    "fog": bool(moves[i].dest_is_fog),
                }
                for i in top_order.tolist()
            ],
            "chosen": {
                "source": [best.row, best.col],
                "direction": best.direction,
                "dest": [best.dest_row, best.dest_col],
                "kind": chosen_kind,
                "split": int(split),
                "base_score": float(base_scores[best_idx]),
                "phase_adjustment": float(phase_scores[best_idx]),
                "mode_adjustment": float(mode_scores[best_idx]),
                "punish_adjustment": float(punish_scores[best_idx]),
                "continuation": float(continuation_scores[best_idx]),
                "adjustment": float(adjustments[best_idx]),
                "score": float(scores[best_idx]),
            },
        }
        return jnp.array([0, best.row, best.col, best.direction, split], dtype=jnp.int32)

    def score_move(self, observation: Observation, move: MoveFeatures) -> float:
        raise NotImplementedError

    def choose_split(self, observation: Observation, move: MoveFeatures) -> int:
        del observation
        return int(move.dest_is_owned and move.source_army >= 8)

    def _update_memory(self, observation: Observation) -> None:
        h, w = observation.armies.shape
        visible = ~(np.asarray(observation.fog_cells) | np.asarray(observation.structures_in_fog))

        if self._seen is None:
            self._seen = np.zeros((h, w), dtype=bool)
            self._last_seen_opponent = np.zeros((h, w), dtype=bool)
            self._last_seen_generals = np.zeros((h, w), dtype=bool)
            self._last_seen_cities = np.zeros((h, w), dtype=bool)

        self._seen |= visible
        self._last_seen_opponent = np.where(
            np.asarray(observation.opponent_cells),
            True,
            self._last_seen_opponent,
        )
        visible_enemy_generals = np.asarray(observation.generals) & np.asarray(observation.opponent_cells)
        self._last_seen_generals = visible_enemy_generals
        self._last_seen_cities = np.where(
            np.asarray(observation.cities),
            True,
            self._last_seen_cities,
        )

        own_generals = np.argwhere(np.asarray(observation.generals) & np.asarray(observation.owned_cells))
        if len(own_generals):
            self._own_general = tuple(map(int, own_generals[0]))

        enemy_generals = np.argwhere(np.asarray(observation.generals) & np.asarray(observation.opponent_cells))
        if len(enemy_generals):
            self._enemy_general_estimate = tuple(map(int, enemy_generals[0]))
            return

        last_seen_enemy = np.argwhere(self._last_seen_opponent)
        if len(last_seen_enemy):
            avg = np.mean(last_seen_enemy, axis=0)
            self._enemy_general_estimate = (int(avg[0]), int(avg[1]))
            return

        fog_structures = np.argwhere(np.asarray(observation.structures_in_fog))
        if len(fog_structures):
            if self._own_general is not None:
                distances = np.sum(np.abs(fog_structures - np.asarray(self._own_general)), axis=1)
                self._enemy_general_estimate = tuple(map(int, fog_structures[int(np.argmax(distances))]))
            else:
                self._enemy_general_estimate = tuple(map(int, fog_structures[0]))
            return

        unseen = np.argwhere(~self._seen)
        if len(unseen):
            if self._own_general is not None:
                distances = np.sum(np.abs(unseen - np.asarray(self._own_general)), axis=1)
                self._enemy_general_estimate = tuple(map(int, unseen[int(np.argmax(distances))]))
            else:
                self._enemy_general_estimate = tuple(map(int, unseen[0]))

    def _extract_moves(self, observation: Observation) -> list[MoveFeatures]:
        valid_mask = np.asarray(
            compute_valid_move_mask(
                observation.armies,
                observation.owned_cells,
                observation.mountains,
            )
        )
        moves_idx = np.argwhere(valid_mask)
        if len(moves_idx) == 0:
            return []

        armies = np.asarray(observation.armies)
        owned = np.asarray(observation.owned_cells)
        opp = np.asarray(observation.opponent_cells)
        neutral = np.asarray(observation.neutral_cells)
        cities = np.asarray(observation.cities)
        generals = np.asarray(observation.generals)
        fog = np.asarray(observation.fog_cells)
        structures_fog = np.asarray(observation.structures_in_fog)
        directions = np.asarray([[-1, 0], [1, 0], [0, -1], [0, 1]])

        moves: list[MoveFeatures] = []
        for row, col, direction in moves_idx.tolist():
            dr, dc = directions[direction]
            dest_row, dest_col = row + int(dr), col + int(dc)
            source_army = int(armies[row, col])
            moving_army = max(source_army - 1, 0)
            dest_army = int(armies[dest_row, dest_col])
            moves.append(
                MoveFeatures(
                    row=row,
                    col=col,
                    direction=direction,
                    dest_row=dest_row,
                    dest_col=dest_col,
                    source_army=source_army,
                    dest_army=dest_army,
                    moving_army=moving_army,
                    can_capture=moving_army > dest_army,
                    dest_is_owned=bool(owned[dest_row, dest_col]),
                    dest_is_opponent=bool(opp[dest_row, dest_col]),
                    dest_is_neutral=bool(neutral[dest_row, dest_col]),
                    dest_is_city=bool(cities[dest_row, dest_col]),
                    dest_is_general=bool(generals[dest_row, dest_col]),
                    dest_is_fog=bool(fog[dest_row, dest_col]),
                    dest_is_structure_fog=bool(structures_fog[dest_row, dest_col]),
                )
            )
        return moves

    def _distance_to_enemy_estimate(self, move: MoveFeatures) -> int:
        if self._enemy_general_estimate is None:
            return 0
        return abs(move.dest_row - self._enemy_general_estimate[0]) + abs(move.dest_col - self._enemy_general_estimate[1])

    def _distance_from_own_general(self, move: MoveFeatures) -> int:
        if self._own_general is None:
            return 0
        return abs(move.dest_row - self._own_general[0]) + abs(move.dest_col - self._own_general[1])

    def _is_income_timing_turn(self, observation: Observation) -> bool:
        next_turn = int(observation.timestep) + 1
        return next_turn % 50 == 0

    def _material_ratio(self, observation: Observation) -> float:
        mine = max(float(observation.owned_army_count), 1.0)
        opp = max(float(observation.opponent_army_count), 1.0)
        return mine / opp

    def _infer_phase(self, observation: Observation) -> str:
        timestep = int(observation.timestep)
        ratio = self._material_ratio(observation)
        enemy_visible = bool(np.any(np.asarray(observation.opponent_cells)))
        if ratio < 0.8:
            return "defense"
        if self._enemy_general_estimate is not None and ratio > 1.2 and enemy_visible:
            return "kill"
        if timestep < 60 and int(observation.owned_land_count) < 18:
            return "opening"
        if ratio > 1.1:
            return "consolidation"
        if enemy_visible or timestep >= 60:
            return "pressure"
        return "expansion"

    def _infer_mode(self, observation: Observation) -> str:
        army_ratio = self._material_ratio(observation)
        land_ratio = max(float(observation.owned_land_count), 1.0) / max(float(observation.opponent_land_count), 1.0)
        visible_enemy_cells = int(np.count_nonzero(np.asarray(observation.opponent_cells)))
        visible_enemy_cities = int(np.count_nonzero(np.asarray(observation.opponent_cells) & np.asarray(observation.cities)))
        own_land = int(observation.owned_land_count)
        opp_land = int(observation.opponent_land_count)
        own_army = int(observation.owned_army_count)
        opp_army = int(observation.opponent_army_count)

        if army_ratio >= 1.3 or (army_ratio >= 1.15 and land_ratio >= 1.1):
            mode = "ahead"
        elif army_ratio <= 0.78 or (army_ratio <= 0.92 and land_ratio <= 0.9):
            mode = "behind"
        else:
            mode = "even"

        self._mode_metrics = {
            "army_ratio": round(army_ratio, 3),
            "land_ratio": round(land_ratio, 3),
            "owned_army": own_army,
            "opponent_army": opp_army,
            "owned_land": own_land,
            "opponent_land": opp_land,
            "visible_enemy_cells": visible_enemy_cells,
            "visible_enemy_cities": visible_enemy_cities,
        }
        return mode

    def _distance_to_target(self, row: int, col: int) -> int:
        if self._current_target is None:
            return 0
        return abs(row - self._current_target[0]) + abs(col - self._current_target[1])

    def get_debug_snapshot(self) -> dict | None:
        return self._last_debug

    def _update_strategy_target(self, observation: Observation) -> None:
        if self._target_commitment > 0:
            self._target_commitment -= 1

        if self._punish_window > 0 and self._punish_target is not None:
            self._current_target = self._punish_target
            self._target_kind = self._punish_kind
            self._target_commitment = max(self._target_commitment, self._punish_window)
            return

        if self._current_target is not None and self._target_kind == "enemy_general" and self._enemy_general_estimate is not None:
            self._current_target = self._enemy_general_estimate

        if self._should_retarget(observation):
            target = self._select_target(observation)
            if target is None:
                self._current_target = None
                self._target_kind = None
                self._target_commitment = 0
            else:
                self._current_target, self._target_kind = target
                self._target_commitment = self._target_commitment_for(self._target_kind)

    def _target_commitment_for(self, kind: str | None) -> int:
        base = {
            "enemy_general": 10,
            "enemy_city": 8,
            "enemy_frontier": 7,
            "frontier_break": 6,
            "structure_fog": 6,
            "neutral_city": 6,
            "unseen": 5,
        }.get(kind or "", 6)
        if self._mode == "ahead":
            base += 2
        elif self._mode == "behind":
            base = max(base - 2, 3)
        return base

    def _should_retarget(self, observation: Observation) -> bool:
        if self._current_target is None or self._target_kind is None:
            return True
        if self._target_commitment <= 0:
            return True
        if self._punish_window > 0 and self._punish_target is not None:
            return tuple(self._current_target) != tuple(self._punish_target)

        row, col = self._current_target
        owned = np.asarray(observation.owned_cells)
        opponent = np.asarray(observation.opponent_cells)
        cities = np.asarray(observation.cities)
        visible = ~(np.asarray(observation.fog_cells) | np.asarray(observation.structures_in_fog))

        if self._target_kind == "enemy_general":
            return False
        if visible[row, col]:
            if self._target_kind == "enemy_city":
                return bool(owned[row, col] or not cities[row, col])
            if self._target_kind == "enemy_frontier":
                return not bool(opponent[row, col])
            if self._target_kind == "neutral_city":
                return bool(owned[row, col] or opponent[row, col] or not cities[row, col])
            if self._target_kind in {"structure_fog", "unseen"}:
                return True
        return False

    def _select_target(self, observation: Observation) -> tuple[tuple[int, int], str] | None:
        if self._punish_window > 0 and self._punish_target is not None and self._punish_kind is not None:
            return self._punish_target, self._punish_kind

        if self._enemy_general_estimate is not None:
            return self._enemy_general_estimate, "enemy_general"

        opponent = np.asarray(observation.opponent_cells)
        cities = np.asarray(observation.cities)
        structures_fog = np.asarray(observation.structures_in_fog)
        fog = np.asarray(observation.fog_cells)
        owned = np.asarray(observation.owned_cells)

        enemy_city_positions = np.argwhere(opponent & cities)
        if len(enemy_city_positions):
            return tuple(map(int, enemy_city_positions[0])), "enemy_city"

        enemy_positions = np.argwhere(opponent)
        if len(enemy_positions):
            best_enemy = max(
                enemy_positions.tolist(),
                key=lambda pos: self._target_priority(observation, tuple(pos), "enemy_frontier"),
            )
            return tuple(map(int, best_enemy)), "enemy_frontier"

        structure_positions = np.argwhere(structures_fog)
        if len(structure_positions):
            best_structure = max(
                structure_positions.tolist(),
                key=lambda pos: self._target_priority(observation, tuple(pos), "structure_fog"),
            )
            return tuple(map(int, best_structure)), "structure_fog"

        neutral_city_positions = np.argwhere(cities & ~owned & ~opponent)
        if len(neutral_city_positions):
            best_city = max(
                neutral_city_positions.tolist(),
                key=lambda pos: self._target_priority(observation, tuple(pos), "neutral_city"),
            )
            return tuple(map(int, best_city)), "neutral_city"

        unseen_positions = np.argwhere(fog)
        if len(unseen_positions):
            best_unseen = max(
                unseen_positions.tolist(),
                key=lambda pos: self._target_priority(observation, tuple(pos), "unseen"),
            )
            return tuple(map(int, best_unseen)), "unseen"
        return None

    def _update_punish_window(self, observation: Observation) -> None:
        if self._punish_window > 0:
            self._punish_window -= 1
            if self._punish_window <= 0:
                self._clear_punish_window()

        detection = self._detect_punish_opportunity(observation)
        if detection is None:
            return

        if detection["strength"] >= self._punish_strength or self._punish_window <= 0:
            self._punish_target = detection["target"]
            self._punish_kind = detection["kind"]
            self._punish_reason = detection["reason"]
            self._punish_strength = detection["strength"]
            self._punish_window = detection["window"]

    def _clear_punish_window(self) -> None:
        self._punish_target = None
        self._punish_kind = None
        self._punish_reason = None
        self._punish_strength = 0.0
        self._punish_window = 0

    def _detect_punish_opportunity(self, observation: Observation) -> dict | None:
        owned = np.asarray(observation.owned_cells)
        opponent = np.asarray(observation.opponent_cells)
        cities = np.asarray(observation.cities)
        generals = np.asarray(observation.generals)
        armies = np.asarray(observation.armies)

        enemy_generals = np.argwhere(generals & opponent)
        if len(enemy_generals):
            target = tuple(map(int, enemy_generals[0]))
            return {
                "target": target,
                "kind": "enemy_general",
                "reason": "visible_enemy_general",
                "strength": 100.0,
                "window": 10,
            }

        enemy_cities = np.argwhere(opponent & cities)
        best_city: tuple[tuple[int, int], float] | None = None
        for row, col in enemy_cities.tolist():
            local_margin = self._local_owned_margin(owned, armies, row, col) - int(armies[row, col])
            if self._mode != "behind" and local_margin >= 2:
                strength = 28.0 + 2.5 * local_margin
                if self._mode == "ahead":
                    strength += 8.0
                if best_city is None or strength > best_city[1]:
                    best_city = ((row, col), strength)
        if best_city is not None:
            return {
                "target": best_city[0],
                "kind": "enemy_city",
                "reason": "visible_enemy_city",
                "strength": best_city[1],
                "window": 6 if self._mode == "ahead" else 4,
            }

        enemy_frontier = np.argwhere(opponent)
        best_frontier: tuple[tuple[int, int], float] | None = None
        for row, col in enemy_frontier.tolist():
            local_margin = self._local_owned_margin(owned, armies, row, col) - int(armies[row, col])
            if local_margin < 4 or self._mode == "behind":
                continue
            dist_term = 0.0
            if self._enemy_general_estimate is not None:
                dist_term = max(0.0, 10.0 - 0.6 * (abs(row - self._enemy_general_estimate[0]) + abs(col - self._enemy_general_estimate[1])))
            strength = 18.0 + 1.8 * local_margin + dist_term
            if best_frontier is None or strength > best_frontier[1]:
                best_frontier = ((row, col), strength)
        if best_frontier is not None:
            return {
                "target": best_frontier[0],
                "kind": "frontier_break",
                "reason": "exposed_enemy_frontier",
                "strength": best_frontier[1],
                "window": 4,
            }

        return None

    def _local_owned_margin(self, owned: np.ndarray, armies: np.ndarray, row: int, col: int) -> int:
        h, w = armies.shape
        margin = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = row + dr
            nc = col + dc
            if 0 <= nr < h and 0 <= nc < w and owned[nr, nc]:
                margin = max(margin, int(armies[nr, nc]) - 1)
        return margin

    def _target_priority(self, observation: Observation, pos: tuple[int, int], kind: str) -> float:
        base_scores = {
            "enemy_general": 300.0,
            "enemy_city": 220.0,
            "enemy_frontier": 180.0,
            "frontier_break": 190.0,
            "structure_fog": 150.0,
            "neutral_city": 120.0,
            "unseen": 90.0,
        }
        score = base_scores[kind]
        if self._own_general is not None:
            home_dist = abs(pos[0] - self._own_general[0]) + abs(pos[1] - self._own_general[1])
            score -= 0.8 * home_dist
        if self._enemy_general_estimate is not None:
            enemy_dist = abs(pos[0] - self._enemy_general_estimate[0]) + abs(pos[1] - self._enemy_general_estimate[1])
            score -= 0.3 * enemy_dist
        if kind in {"enemy_frontier", "enemy_city"}:
            score += 10.0 * self._material_ratio(observation)
        if kind == "neutral_city" and self._material_ratio(observation) < 0.9:
            score -= 20.0
        if self._mode == "ahead" and kind in {"enemy_general", "enemy_city", "enemy_frontier", "frontier_break"}:
            score += 18.0
        if self._mode == "behind" and kind == "neutral_city":
            score += 12.0
        if self._mode == "behind" and kind in {"enemy_frontier", "frontier_break"}:
            score -= 20.0
        if self._punish_window > 0 and self._punish_target is not None and tuple(pos) == tuple(self._punish_target):
            score += 40.0
        return score

    def _phase_adjustment(self, observation: Observation, move: MoveFeatures) -> float:
        kind = self._classify_move(move)
        score = 0.0
        ratio = self._material_ratio(observation)
        if self._phase == "opening":
            if kind == "expand" and move.can_capture:
                score += 10.0
            if kind == "scout":
                score += 8.0
            if kind in {"reinforce", "reinforce_city"}:
                score -= 6.0
        elif self._phase == "expansion":
            if kind in {"expand", "attack_city"} and move.can_capture:
                score += 8.0
            if kind == "reinforce":
                score -= 4.0
        elif self._phase == "pressure":
            if kind in {"attack", "attack_city", "scout"}:
                score += 8.0
            if kind in {"reinforce", "reinforce_city"} and move.source_army >= 8:
                score -= 6.0
        elif self._phase == "consolidation":
            if ratio > 1.15 and kind in {"attack", "attack_city"} and move.can_capture:
                score += 12.0
            if kind == "scout":
                score += 5.0
            if kind == "reinforce" and move.source_army >= 12:
                score -= 8.0
        elif self._phase == "kill":
            if kind in {"attack", "attack_city"} and move.can_capture:
                score += 16.0
            if self._enemy_general_estimate is not None:
                source_enemy_dist = abs(move.row - self._enemy_general_estimate[0]) + abs(move.col - self._enemy_general_estimate[1])
                dest_enemy_dist = self._distance_to_enemy_estimate(move)
                score += 5.0 * (source_enemy_dist - dest_enemy_dist)
            if kind in {"reinforce", "reinforce_city"}:
                score -= 12.0
        elif self._phase == "defense":
            if self._own_general is not None:
                source_home_dist = abs(move.row - self._own_general[0]) + abs(move.col - self._own_general[1])
                dest_home_dist = self._distance_from_own_general(move)
                if dest_home_dist < source_home_dist and kind in {"reinforce", "attack"}:
                    score += 10.0
            if kind == "attack_city" and ratio < 0.9:
                score -= 12.0
        return score

    def _continuation_value(self, observation: Observation, move: MoveFeatures) -> float:
        armies = np.asarray(observation.armies)
        owned = np.asarray(observation.owned_cells)
        opponent = np.asarray(observation.opponent_cells)
        neutral = np.asarray(observation.neutral_cells)
        cities = np.asarray(observation.cities)
        fog = np.asarray(observation.fog_cells)
        structures_fog = np.asarray(observation.structures_in_fog)
        mountains = np.asarray(observation.mountains)
        h, w = armies.shape
        score = 0.0

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = move.dest_row + dr
            nc = move.dest_col + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w or mountains[nr, nc]:
                continue
            if opponent[nr, nc]:
                score += 4.5
                if move.moving_army > int(armies[nr, nc]):
                    score += 2.0
            elif neutral[nr, nc]:
                score += 2.5
                if move.moving_army > int(armies[nr, nc]):
                    score += 1.0
            elif cities[nr, nc] and not owned[nr, nc]:
                score += 5.0
            elif fog[nr, nc] or structures_fog[nr, nc]:
                score += 2.0
            elif owned[nr, nc]:
                score += 0.5

        if self._current_target is not None:
            score += max(0.0, 6.0 - self._distance_to_target(move.dest_row, move.dest_col) * 0.5)
        if move.dest_is_opponent and move.can_capture:
            score += 3.0
        if move.dest_is_city and move.can_capture:
            score += 3.0
        return score

    def _mode_adjustment(self, observation: Observation, move: MoveFeatures) -> float:
        kind = self._classify_move(move)
        score = 0.0
        if self._mode == "ahead":
            if kind in {"attack", "attack_city", "attack_general"} and move.can_capture:
                score += 10.0
            if kind == "scout" and move.source_army >= 6:
                score += 4.0
            if kind in {"reinforce", "reinforce_city"} and move.source_army >= 10:
                score -= 8.0
        elif self._mode == "behind":
            if self._own_general is not None:
                source_home_dist = abs(move.row - self._own_general[0]) + abs(move.col - self._own_general[1])
                dest_home_dist = self._distance_from_own_general(move)
                if dest_home_dist < source_home_dist and kind in {"reinforce", "attack"}:
                    score += 8.0
            if kind in {"attack_city", "scout"} and move.source_army >= 10:
                score -= 10.0
            if kind == "reinforce" and move.source_army >= 8:
                score += 4.0
        else:
            if kind in {"attack", "attack_city"} and move.can_capture:
                score += 3.0
            if kind == "reinforce" and move.source_army >= 14:
                score -= 3.0
        if self._is_income_timing_turn(observation) and self._mode == "ahead" and kind in {"expand", "attack"} and move.can_capture:
            score += 4.0
        return score

    def _punish_adjustment(self, observation: Observation, move: MoveFeatures) -> float:
        del observation
        if self._punish_window <= 0 or self._punish_target is None:
            return 0.0

        kind = self._classify_move(move)
        source_dist = abs(move.row - self._punish_target[0]) + abs(move.col - self._punish_target[1])
        dest_dist = abs(move.dest_row - self._punish_target[0]) + abs(move.dest_col - self._punish_target[1])
        progress = source_dist - dest_dist
        score = 6.0 * progress
        if progress > 0 and kind in {"attack", "attack_city", "attack_general", "scout", "expand"}:
            score += 8.0
        if progress < 0 and kind in {"reinforce", "reinforce_city"} and move.source_army >= 8:
            score += 6.0 * progress

        if self._punish_kind == "enemy_city":
            if move.dest_is_city and move.can_capture:
                score += 22.0
            elif kind in {"reinforce", "reinforce_city"}:
                score -= 6.0
        elif self._punish_kind == "frontier_break":
            if move.dest_is_opponent and move.can_capture:
                score += 16.0
            if kind == "scout" and move.source_army < 6:
                score -= 4.0
        elif self._punish_kind == "enemy_general":
            if kind == "attack_general" and move.can_capture:
                score += 1000.0
            elif kind in {"reinforce", "reinforce_city"}:
                score -= 10.0

        return score

    def _classify_move(self, move: MoveFeatures) -> str:
        if move.dest_is_general and move.can_capture:
            return "attack_general"
        if move.dest_is_city:
            return "attack_city" if (move.dest_is_opponent or move.dest_is_neutral) else "reinforce_city"
        if move.dest_is_opponent:
            return "attack"
        if move.dest_is_neutral:
            return "expand"
        if move.dest_is_fog or move.dest_is_structure_fog:
            return "scout"
        if move.dest_is_owned:
            return "reinforce"
        return "move"

    def _remember_choice(self, move: MoveFeatures, kind: str) -> None:
        self._recent_choices.append(
            {
                "source": (move.row, move.col),
                "dest": (move.dest_row, move.dest_col),
                "kind": kind,
            }
        )
        self._recent_action_kinds.append(kind)

    def _score_adjustment(self, observation: Observation, move: MoveFeatures) -> float:
        score = 0.0
        dest = (move.dest_row, move.dest_col)
        source = (move.row, move.col)
        kind = self._classify_move(move)
        recent_choices = list(self._recent_choices)
        recent_kinds = list(self._recent_action_kinds)

        if recent_choices:
            last = recent_choices[-1]
            if source == last["dest"] and dest == last["source"]:
                score -= 28.0
            if dest == last["source"]:
                score -= 10.0
            if dest == last["dest"] and kind in {"reinforce", "reinforce_city"}:
                score -= 8.0

        if len(recent_choices) >= 4:
            recent_sources = [choice["source"] for choice in recent_choices[-4:]]
            recent_dests = [choice["dest"] for choice in recent_choices[-4:]]
            if dest in recent_sources[-3:] or dest in recent_dests[-3:]:
                score -= 6.0

        stagnating = len(recent_kinds) >= 4 and all(kind_name in {"reinforce", "reinforce_city"} for kind_name in recent_kinds[-4:])
        if stagnating:
            if kind in {"attack", "attack_city", "scout", "expand"} and move.source_army >= 6:
                score += 16.0
            if kind in {"reinforce", "reinforce_city"} and move.source_army >= 10:
                score -= 12.0

        if move.dest_is_owned and move.source_army >= 18:
            score -= 6.0
        if move.dest_is_opponent and move.can_capture:
            score += 8.0
        if move.dest_is_city and move.can_capture:
            score += 6.0
        if move.dest_is_fog and move.source_army >= 6:
            score += 4.0
        if self._enemy_general_estimate is not None:
            source_enemy_dist = abs(move.row - self._enemy_general_estimate[0]) + abs(move.col - self._enemy_general_estimate[1])
            dest_enemy_dist = self._distance_to_enemy_estimate(move)
            if dest_enemy_dist < source_enemy_dist and not move.dest_is_owned:
                score += 3.0
        if self._current_target is not None:
            source_target_dist = self._distance_to_target(move.row, move.col)
            dest_target_dist = self._distance_to_target(move.dest_row, move.dest_col)
            progress = source_target_dist - dest_target_dist
            score += 4.5 * progress
            if progress < 0 and kind in {"reinforce", "reinforce_city"} and move.source_army >= 8:
                score += 5.0 * progress
            if progress > 0 and kind in {"attack", "attack_city", "scout", "expand"}:
                score += 6.0
            if self._target_kind == "enemy_city" and move.dest_is_city and move.can_capture:
                score += 18.0
            if self._target_kind == "enemy_frontier" and move.dest_is_opponent and move.can_capture:
                score += 14.0
            if self._target_kind in {"structure_fog", "unseen"} and kind == "scout":
                score += 10.0

        if self._own_general is not None and move.dest_is_owned:
            source_home_dist = abs(move.row - self._own_general[0]) + abs(move.col - self._own_general[1])
            dest_home_dist = self._distance_from_own_general(move)
            if source_home_dist <= 2 and dest_home_dist <= 2 and move.source_army >= 12:
                score -= 8.0

        if kind == "expand" and move.source_army < 3:
            score -= 6.0

        return score


class MaterialAdvantageAgent(StrategicAgent):
    """Focuses on robust expansion, city capture, and income-timed land grabs."""

    def __init__(self, id: str = "MaterialAdvantage"):
        super().__init__(id)

    def score_move(self, observation: Observation, move: MoveFeatures) -> float:
        score = float(move.source_army) * 0.15

        if move.dest_is_general and move.can_capture:
            return 1e6

        if move.dest_is_opponent and move.can_capture:
            score += 140.0 + 1.5 * move.moving_army

        if move.dest_is_city and move.can_capture:
            score += 110.0
            if self._material_ratio(observation) > 1.05:
                score += 35.0

        if move.dest_is_neutral and move.can_capture:
            score += 30.0

        if self._is_income_timing_turn(observation) and (move.dest_is_neutral or move.dest_is_opponent) and move.can_capture:
            score += 80.0

        if move.dest_is_owned:
            score += 10.0 - 0.5 * move.dest_army

        if move.dest_is_fog:
            score += 12.0

        score -= 0.8 * self._distance_to_enemy_estimate(move)
        return score


class ScoutPressureAgent(StrategicAgent):
    """Scouts aggressively early, then converts information into pressure."""

    def __init__(self, id: str = "ScoutPressure"):
        super().__init__(id)

    def score_move(self, observation: Observation, move: MoveFeatures) -> float:
        score = 0.0
        early_game = int(observation.timestep) < 80

        if move.dest_is_general and move.can_capture:
            return 1e6

        if move.dest_is_fog:
            score += 70.0 if early_game else 20.0

        if move.dest_is_structure_fog:
            score += 55.0

        if move.dest_is_opponent and move.can_capture:
            score += 95.0

        if move.dest_is_city and move.can_capture:
            score += 75.0

        if move.dest_is_neutral and move.can_capture:
            score += 18.0

        if self._enemy_general_estimate is not None:
            score += max(0.0, 40.0 - self._distance_to_enemy_estimate(move))

        if move.dest_is_owned:
            score -= 10.0

        if move.source_army >= 12:
            score += 6.0

        return score


class BackdoorAgent(StrategicAgent):
    """Prefers deep incursions and island creation inside enemy territory."""

    def __init__(self, id: str = "Backdoor"):
        super().__init__(id)

    def score_move(self, observation: Observation, move: MoveFeatures) -> float:
        score = float(move.source_army) * 0.1

        if move.dest_is_general and move.can_capture:
            return 1e6

        toward_enemy = -self._distance_to_enemy_estimate(move)
        away_from_home = self._distance_from_own_general(move)

        if move.dest_is_opponent and move.can_capture:
            score += 100.0 + 0.8 * away_from_home

        if move.dest_is_fog:
            score += 25.0 + 0.5 * away_from_home

        if move.dest_is_structure_fog:
            score += 35.0

        if move.dest_is_neutral and move.can_capture:
            score += 12.0

        if move.dest_is_city and move.can_capture:
            score += 50.0

        if move.dest_is_owned:
            score -= 4.0

        score += 0.8 * toward_enemy
        return score


class DefenseCounterAgent(StrategicAgent):
    """Keeps a stronger defensive shell and counterattacks once pressure is absorbed."""

    def __init__(self, id: str = "DefenseCounter"):
        super().__init__(id)

    def score_move(self, observation: Observation, move: MoveFeatures) -> float:
        score = 0.0
        home_dist = self._distance_from_own_general(move)
        enemy_dist = self._distance_to_enemy_estimate(move)
        behind = int(observation.owned_army_count) < int(observation.opponent_army_count)

        if move.dest_is_general and move.can_capture:
            return 1e6

        if move.dest_is_owned:
            score += 18.0 - 0.7 * move.dest_army
            score += max(0.0, 16.0 - 1.2 * home_dist)

        if move.dest_is_city and move.can_capture:
            score += 50.0 if not behind else 20.0

        if move.dest_is_opponent and move.can_capture:
            score += 40.0 if behind else 70.0

        if move.dest_is_neutral and move.can_capture:
            score += 10.0 if behind else 18.0

        if move.dest_is_fog:
            score += 6.0 if behind else 12.0

        score += max(0.0, 10.0 - 0.8 * home_dist)
        score -= 0.25 * enemy_dist
        return score


class SurroundPressureAgent(StrategicAgent):
    """Prefers flank growth and multi-angle pressure over direct linear pushes."""

    def __init__(self, id: str = "SurroundPressure"):
        super().__init__(id)

    def score_move(self, observation: Observation, move: MoveFeatures) -> float:
        score = 0.0

        if move.dest_is_general and move.can_capture:
            return 1e6

        enemy_dist = self._distance_to_enemy_estimate(move)
        home_dist = self._distance_from_own_general(move)

        if move.dest_is_opponent and move.can_capture:
            score += 90.0 + 0.3 * home_dist

        if move.dest_is_city and move.can_capture:
            score += 65.0

        if move.dest_is_neutral and move.can_capture:
            score += 20.0

        if move.dest_is_fog or move.dest_is_structure_fog:
            score += 22.0

        if move.dest_is_owned:
            score -= 8.0

        # Prefer medium-distance approach cells rather than only shortest-path tunneling.
        score += max(0.0, 30.0 - abs(enemy_dist - 6) * 3.0)
        score += min(home_dist, 8) * 0.6
        return score


class TurtleAgent(StrategicAgent):
    """City-first defensive bot that prefers a thick home shell over map tempo."""

    def __init__(self, id: str = "Turtle"):
        super().__init__(id)

    def score_move(self, observation: Observation, move: MoveFeatures) -> float:
        home_dist = self._distance_from_own_general(move)
        score = 0.0
        if move.dest_is_general and move.can_capture:
            return 1e6
        if move.dest_is_owned:
            score += 22.0 - 0.8 * move.dest_army
            score += max(0.0, 18.0 - 1.4 * home_dist)
        if move.dest_is_city and move.can_capture:
            score += 80.0
        if move.dest_is_opponent and move.can_capture:
            score += 28.0 if home_dist <= 6 else 12.0
        if move.dest_is_neutral and move.can_capture:
            score += 8.0 if home_dist <= 5 else 2.0
        if move.dest_is_fog and home_dist <= 6:
            score += 5.0
        if home_dist > 7:
            score -= 12.0
        return score


class PunishAgent(StrategicAgent):
    """Aggressively attacks exposed enemy cities and weak frontier tiles."""

    def __init__(self, id: str = "Punish"):
        super().__init__(id)

    def score_move(self, observation: Observation, move: MoveFeatures) -> float:
        score = float(move.source_army) * 0.12
        if move.dest_is_general and move.can_capture:
            return 1e6
        if move.dest_is_city and move.can_capture:
            score += 140.0 + 1.2 * move.moving_army
        if move.dest_is_opponent and move.can_capture:
            score += 110.0 + 0.8 * move.moving_army
        if move.dest_is_neutral and move.can_capture:
            score += 12.0
        if move.dest_is_fog:
            score += 6.0
        if move.dest_is_owned:
            score -= 10.0
        if self._punish_window > 0:
            score += 18.0
        score -= 0.5 * self._distance_to_enemy_estimate(move)
        return score


class SwarmAgent(StrategicAgent):
    """Spreads quickly across land and favors multi-lane map control over concentration."""

    def __init__(self, id: str = "Swarm"):
        super().__init__(id)

    def choose_split(self, observation: Observation, move: MoveFeatures) -> int:
        del observation
        return int(move.source_army >= 6)

    def score_move(self, observation: Observation, move: MoveFeatures) -> float:
        score = 0.0
        if move.dest_is_general and move.can_capture:
            return 1e6
        if move.dest_is_neutral and move.can_capture:
            score += 42.0
        if move.dest_is_opponent and move.can_capture:
            score += 65.0
        if move.dest_is_city and move.can_capture:
            score += 48.0
        if move.dest_is_fog or move.dest_is_structure_fog:
            score += 24.0
        if move.dest_is_owned:
            score -= 14.0
        score += min(self._distance_from_own_general(move), 10) * 1.2
        score += max(0.0, 20.0 - self._distance_to_enemy_estimate(move))
        return score


class SniperAgent(StrategicAgent):
    """Concentrates armies and biases hard toward direct enemy-general progress."""

    def __init__(self, id: str = "Sniper"):
        super().__init__(id)

    def choose_split(self, observation: Observation, move: MoveFeatures) -> int:
        del observation
        return int(move.dest_is_owned and move.source_army >= 18)

    def score_move(self, observation: Observation, move: MoveFeatures) -> float:
        score = float(move.source_army) * 0.3
        if move.dest_is_general and move.can_capture:
            return 1e6
        if move.dest_is_opponent and move.can_capture:
            score += 88.0 + 1.8 * move.moving_army
        if move.dest_is_city and move.can_capture:
            score += 26.0
        if move.dest_is_neutral and move.can_capture:
            score += 6.0
        if move.dest_is_owned:
            score += 4.0 - 0.2 * move.dest_army
        if self._enemy_general_estimate is not None:
            score += max(0.0, 55.0 - 4.0 * self._distance_to_enemy_estimate(move))
        if move.dest_is_fog:
            score += 8.0
        return score


class GreedyCityAgent(StrategicAgent):
    """Overvalues city capture and economy, useful as a punishable sparring partner."""

    def __init__(self, id: str = "GreedyCity"):
        super().__init__(id)

    def score_move(self, observation: Observation, move: MoveFeatures) -> float:
        score = 0.0
        if move.dest_is_general and move.can_capture:
            return 1e6
        if move.dest_is_city and move.can_capture:
            score += 180.0
        if move.dest_is_neutral and move.can_capture:
            score += 14.0
        if move.dest_is_opponent and move.can_capture:
            score += 36.0
        if move.dest_is_fog:
            score += 10.0
        if move.dest_is_owned:
            score -= 6.0
        return score


class ChaosAgent(StrategicAgent):
    """Fog-chasing opportunist with unstable priorities to diversify sparring behavior."""

    def __init__(self, id: str = "Chaos"):
        super().__init__(id)

    def score_move(self, observation: Observation, move: MoveFeatures) -> float:
        score = 0.0
        if move.dest_is_general and move.can_capture:
            return 1e6
        if move.dest_is_fog:
            score += 60.0
        if move.dest_is_structure_fog:
            score += 40.0
        if move.dest_is_opponent and move.can_capture:
            score += 54.0 + 0.5 * move.moving_army
        if move.dest_is_city and move.can_capture:
            score += 52.0
        if move.dest_is_neutral and move.can_capture:
            score += 18.0
        if move.dest_is_owned:
            score -= 12.0
        if self._enemy_general_estimate is not None:
            score += max(0.0, 25.0 - 1.5 * self._distance_to_enemy_estimate(move))
        return score
