from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
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
        self._last_debug = None

    def act(self, observation: Observation, key: jnp.ndarray) -> jnp.ndarray:
        del key
        self._update_memory(observation)
        moves = self._extract_moves(observation)
        if not moves:
            self._last_debug = {
                "decision": "pass",
                "reason": "no_valid_moves",
                "enemy_general_estimate": self._enemy_general_estimate,
            }
            return jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)

        scores = np.array([self.score_move(observation, move) for move in moves], dtype=np.float32)
        best_idx = int(np.argmax(scores))
        best = moves[best_idx]
        split = self.choose_split(observation, best)
        top_order = np.argsort(scores)[::-1][: min(3, len(scores))]
        self._last_debug = {
            "decision": "move",
            "enemy_general_estimate": self._enemy_general_estimate,
            "own_general": self._own_general,
            "top_candidates": [
                {
                    "source": [moves[i].row, moves[i].col],
                    "direction": moves[i].direction,
                    "dest": [moves[i].dest_row, moves[i].dest_col],
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
                "split": int(split),
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

    def get_debug_snapshot(self) -> dict | None:
        return self._last_debug


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
