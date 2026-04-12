from __future__ import annotations

import heapq
from collections import deque

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from generals.core.action import compute_valid_move_mask
from generals.core.observation import Observation

from .strategic_agent import MoveFeatures, StrategicAgent

# Direction offsets: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
_DR = np.array([-1, 1, 0, 0], dtype=np.int32)
_DC = np.array([0, 0, -1, 1], dtype=np.int32)
# Reverse direction: moving FROM neighbor TO current cell
_REVERSE_DIR = np.array([1, 0, 3, 2], dtype=np.int32)

_INF = 10**9


class GraphSearchAgent(StrategicAgent):
    """
    Agent that uses BFS and Dijkstra pathfinding on the grid graph to plan
    multi-step army movements, replacing greedy Manhattan-distance heuristics
    with true shortest-path and cost-aware routing.
    """

    def __init__(self, id: str = "GraphSearch"):
        super().__init__(id)

    def reset(self):
        super().reset()
        self._distance_map: np.ndarray | None = None
        self._cost_map: np.ndarray | None = None
        self._flow_direction: np.ndarray | None = None
        self._gather_map: np.ndarray | None = None
        self._gather_direction: np.ndarray | None = None
        self._plan_target: tuple[int, int] | None = None
        self._plan_gather_point: tuple[int, int] | None = None
        self._plan_turn: int = -_INF
        self._plan_mode: str = "expand"
        self._recompute_interval: int = 3

    # ------------------------------------------------------------------
    # Pathfinding algorithms
    # ------------------------------------------------------------------

    @staticmethod
    def _bfs_from(
        target: tuple[int, int], passable: np.ndarray, H: int, W: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """BFS from *target* over passable cells.

        Returns:
            distance_map (H, W): shortest-path distance from each cell to target.
            flow_direction (H, W): direction index (0-3) to move one step closer
                to target along the BFS tree.  -1 for unreachable / target itself.
        """
        dist = np.full((H, W), _INF, dtype=np.int32)
        flow = np.full((H, W), -1, dtype=np.int32)
        tr, tc = target
        if not (0 <= tr < H and 0 <= tc < W and passable[tr, tc]):
            return dist, flow
        dist[tr, tc] = 0
        queue: deque[tuple[int, int]] = deque()
        queue.append((tr, tc))
        while queue:
            r, c = queue.popleft()
            d = dist[r, c] + 1
            for direction in range(4):
                nr = r + _DR[direction]
                nc = c + _DC[direction]
                if 0 <= nr < H and 0 <= nc < W and passable[nr, nc] and dist[nr, nc] > d:
                    dist[nr, nc] = d
                    # To go from (nr, nc) toward target, move in the reverse direction
                    flow[nr, nc] = _REVERSE_DIR[direction]
                    queue.append((nr, nc))
        return dist, flow

    @staticmethod
    def _dijkstra_cost_from(
        target: tuple[int, int],
        passable: np.ndarray,
        armies: np.ndarray,
        owned: np.ndarray,
        opponent: np.ndarray,
        fog: np.ndarray,
        structures_fog: np.ndarray,
        timestep: int,
        H: int,
        W: int,
    ) -> np.ndarray:
        """Dijkstra from *target* backward — computes minimum army cost to
        fight from each cell to target."""

        def cell_cost(r: int, c: int) -> int:
            if owned[r, c]:
                return 0
            a = int(armies[r, c])
            if opponent[r, c]:
                return a + 1
            if structures_fog[r, c]:
                return max(20, 10 + timestep // 25)
            if fog[r, c]:
                return max(1, timestep // 50)
            # Neutral visible cell
            return a + 1 if a > 0 else 1

        cost = np.full((H, W), _INF, dtype=np.int64)
        tr, tc = target
        if not (0 <= tr < H and 0 <= tc < W and passable[tr, tc]):
            return cost
        cost[tr, tc] = cell_cost(tr, tc)
        heap: list[tuple[int, int, int]] = [(int(cost[tr, tc]), tr, tc)]
        while heap:
            c_val, r, c = heapq.heappop(heap)
            if c_val > cost[r, c]:
                continue
            for direction in range(4):
                nr = r + _DR[direction]
                nc = c + _DC[direction]
                if 0 <= nr < H and 0 <= nc < W and passable[nr, nc]:
                    new_cost = c_val + cell_cost(nr, nc)
                    if new_cost < cost[nr, nc]:
                        cost[nr, nc] = new_cost
                        heapq.heappush(heap, (int(new_cost), nr, nc))
        return cost

    @staticmethod
    def _bfs_from_owned(
        gather_point: tuple[int, int],
        owned: np.ndarray,
        passable: np.ndarray,
        H: int,
        W: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """BFS restricted to owned & passable cells — for army gathering."""
        dist = np.full((H, W), _INF, dtype=np.int32)
        flow = np.full((H, W), -1, dtype=np.int32)
        gr, gc = gather_point
        if not (0 <= gr < H and 0 <= gc < W and owned[gr, gc] and passable[gr, gc]):
            return dist, flow
        dist[gr, gc] = 0
        queue: deque[tuple[int, int]] = deque()
        queue.append((gr, gc))
        while queue:
            r, c = queue.popleft()
            d = dist[r, c] + 1
            for direction in range(4):
                nr = r + _DR[direction]
                nc = c + _DC[direction]
                if (
                    0 <= nr < H
                    and 0 <= nc < W
                    and passable[nr, nc]
                    and owned[nr, nc]
                    and dist[nr, nc] > d
                ):
                    dist[nr, nc] = d
                    flow[nr, nc] = _REVERSE_DIR[direction]
                    queue.append((nr, nc))
        return dist, flow

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------

    def _select_gather_point(self, observation: Observation) -> tuple[int, int] | None:
        """Pick the owned cell closest to the target with the best local army."""
        if self._distance_map is None or self._current_target is None:
            return self._own_general

        owned = np.asarray(observation.owned_cells)
        armies = np.asarray(observation.armies)
        H, W = armies.shape
        owned_positions = np.argwhere(owned)
        if len(owned_positions) == 0:
            return self._own_general

        best_score = -_INF
        best_pos: tuple[int, int] | None = None
        for pos in owned_positions:
            r, c = int(pos[0]), int(pos[1])
            dist = self._distance_map[r, c]
            if dist >= _INF:
                continue
            # Sum armies in 3x3 owned neighborhood
            r_lo, r_hi = max(0, r - 1), min(H, r + 2)
            c_lo, c_hi = max(0, c - 1), min(W, c + 2)
            local_army = int(np.sum(armies[r_lo:r_hi, c_lo:c_hi] * owned[r_lo:r_hi, c_lo:c_hi]))
            score = local_army * 0.5 - dist * 2.0
            if score > best_score:
                best_score = score
                best_pos = (r, c)
        return best_pos if best_pos is not None else self._own_general

    def _select_plan_mode(self, observation: Observation) -> str:
        if self._phase == "defense":
            return "defend"

        # Check if we can attack the enemy general directly
        if (
            self._enemy_general_estimate is not None
            and self._cost_map is not None
        ):
            owned = np.asarray(observation.owned_cells)
            armies = np.asarray(observation.armies)
            owned_armies = armies * owned
            if owned_armies.size > 0:
                best_idx = np.argmax(owned_armies)
                best_r, best_c = divmod(int(best_idx), armies.shape[1])
                best_army = int(armies[best_r, best_c])
                cost = int(self._cost_map[best_r, best_c])
                if cost < _INF and best_army > cost * 1.3:
                    return "attack"

        # Gather when ahead but no single cell can attack
        if self._mode == "ahead" and int(observation.owned_army_count) > 50:
            return "gather"

        return "expand"

    def _recompute_maps(self, observation: Observation) -> None:
        mountains = np.asarray(observation.mountains)
        passable = ~mountains
        armies = np.asarray(observation.armies)
        owned = np.asarray(observation.owned_cells)
        opponent = np.asarray(observation.opponent_cells)
        fog = np.asarray(observation.fog_cells)
        structures_fog = np.asarray(observation.structures_in_fog)
        timestep = int(observation.timestep)
        H, W = armies.shape

        target = self._current_target
        self._plan_target = target
        self._plan_turn = timestep

        if target is None:
            self._distance_map = None
            self._cost_map = None
            self._flow_direction = None
            self._gather_map = None
            self._gather_direction = None
            return

        # BFS distance + flow direction
        self._distance_map, self._flow_direction = self._bfs_from(target, passable, H, W)

        # Dijkstra army cost
        self._cost_map = self._dijkstra_cost_from(
            target, passable, armies, owned, opponent, fog, structures_fog, timestep, H, W
        )

        # Gather map (over owned territory)
        gather_point = self._select_gather_point(observation)
        if gather_point is not None:
            self._plan_gather_point = gather_point
            self._gather_map, self._gather_direction = self._bfs_from_owned(
                gather_point, owned, passable, H, W
            )
        else:
            self._gather_map = None
            self._gather_direction = None

    # ------------------------------------------------------------------
    # Core overrides
    # ------------------------------------------------------------------

    def act(self, observation: Observation, key: jnp.ndarray) -> jnp.ndarray:
        # Parent state updates
        self._update_memory(observation)
        self._phase = self._infer_phase(observation)
        self._mode = self._infer_mode(observation)
        self._update_punish_window(observation)
        self._update_strategy_target(observation)

        # Recompute pathfinding maps when needed
        timestep = int(observation.timestep)
        target_changed = self._current_target != self._plan_target
        should_recompute = (
            target_changed
            or self._distance_map is None
            or (timestep - self._plan_turn) >= self._recompute_interval
        )
        if should_recompute:
            self._recompute_maps(observation)

        self._plan_mode = self._select_plan_mode(observation)

        # Move extraction and scoring (mirrors parent act() pipeline)
        moves = self._extract_moves(observation)
        if not moves:
            self._last_debug = {
                "decision": "pass",
                "reason": "no_valid_moves",
                "phase": self._phase,
                "mode": self._mode,
                "plan_mode": self._plan_mode,
                "plan_target": self._plan_target,
                "plan_gather_point": self._plan_gather_point,
            }
            return jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)

        base_scores = np.array([self.score_move(observation, m) for m in moves], dtype=np.float32)
        phase_scores = np.array([self._phase_adjustment(observation, m) for m in moves], dtype=np.float32)
        mode_scores = np.array([self._mode_adjustment(observation, m) for m in moves], dtype=np.float32)
        punish_scores = np.array([self._punish_adjustment(observation, m) for m in moves], dtype=np.float32)
        continuation_scores = np.array([self._continuation_value(observation, m) for m in moves], dtype=np.float32)
        adjustments = np.array([self._score_adjustment(observation, m) for m in moves], dtype=np.float32)
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
            "plan_mode": self._plan_mode,
            "plan_target": self._plan_target,
            "plan_gather_point": self._plan_gather_point,
            "recompute_turn": self._plan_turn,
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
        score = float(move.source_army) * 0.1

        # Instant win
        if move.dest_is_general and move.can_capture:
            return 1e6

        # --- Path alignment ---
        if self._distance_map is not None:
            source_dist = int(self._distance_map[move.row, move.col])
            dest_dist = int(self._distance_map[move.dest_row, move.dest_col])
            if source_dist < _INF and dest_dist < _INF:
                path_progress = source_dist - dest_dist  # +1 if closer, -1 if farther
                score += 15.0 * path_progress
                # Bonus for following the exact BFS-optimal direction
                if self._flow_direction is not None:
                    optimal_dir = int(self._flow_direction[move.row, move.col])
                    if optimal_dir >= 0 and move.direction == optimal_dir:
                        score += 10.0

        # --- Cost feasibility (attack/gather modes) ---
        if self._cost_map is not None and self._plan_mode in ("attack", "gather"):
            cost_here = int(self._cost_map[move.row, move.col])
            if cost_here < _INF:
                if move.moving_army > cost_here * 0.5:
                    score += 20.0
                elif move.moving_army < cost_here * 0.2 and not move.dest_is_owned:
                    score -= 10.0

        # --- Gathering (gather mode, moving through own territory) ---
        if self._plan_mode == "gather" and self._gather_map is not None and move.dest_is_owned:
            src_gather = int(self._gather_map[move.row, move.col])
            dst_gather = int(self._gather_map[move.dest_row, move.dest_col])
            if src_gather < _INF and dst_gather < _INF:
                gather_progress = src_gather - dst_gather
                score += 12.0 * gather_progress + 8.0
                score += move.source_army * 0.3

        # --- Tactical bonuses ---
        if move.dest_is_opponent and move.can_capture:
            score += 60.0
        if move.dest_is_city and move.can_capture:
            score += 50.0
        if move.dest_is_neutral and move.can_capture:
            score += 15.0
        if move.dest_is_fog:
            score += 8.0
        if move.dest_is_owned and self._plan_mode != "gather":
            score -= 5.0

        # --- Defense mode ---
        if self._plan_mode == "defend" and self._own_general is not None:
            home_dist = self._distance_from_own_general(move)
            if home_dist <= 4:
                score += 15.0 - home_dist * 2.0

        return score

    def choose_split(self, observation: Observation, move: MoveFeatures) -> int:
        # When gathering, split to leave some defense behind
        if self._plan_mode == "gather" and move.dest_is_owned and move.source_army >= 6:
            return 1
        # Split when reinforcing through own territory
        if move.dest_is_owned and move.source_army >= 10:
            return 1
        # Full force when attacking
        return 0

    def get_debug_snapshot(self) -> dict | None:
        return self._last_debug
