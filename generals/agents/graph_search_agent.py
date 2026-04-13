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
        self._city_dist_map: np.ndarray | None = None
        self._city_flow: np.ndarray | None = None
        self._city_rush_target: tuple[int, int] | None = None
        self._city_rush_garrison: int = 0
        self._frontier_dist_map: np.ndarray | None = None
        self._own_territory_dist: np.ndarray | None = None
        self._scout_target: tuple[int, int] | None = None
        self._scout_dist_map: np.ndarray | None = None
        self._scout_flow: np.ndarray | None = None
        self._plan_target: tuple[int, int] | None = None
        self._plan_gather_point: tuple[int, int] | None = None
        self._plan_turn: int = -_INF
        self._plan_mode: str = "expand"
        self._recompute_interval: int = 3
        self._stagnation_counter: int = 0
        self._last_land_count: int = 0

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
        seen: np.ndarray | None = None,
        last_seen_opponent: np.ndarray | None = None,
    ) -> np.ndarray:
        """Dijkstra from *target* backward — computes minimum army cost to
        fight from each cell to target.  Uses persistent memory (seen,
        last_seen_opponent) to give better estimates for fogged cells."""

        def cell_cost(r: int, c: int) -> int:
            if owned[r, c]:
                return 0
            a = int(armies[r, c])
            if opponent[r, c]:
                return a + 1
            if structures_fog[r, c]:
                # Known structure in fog — could be city with garrison
                return max(20, 10 + timestep // 25)
            if fog[r, c]:
                # Use memory: if we've seen this cell before and it wasn't
                # enemy territory, it's likely still cheap to traverse
                if seen is not None and seen[r, c]:
                    if last_seen_opponent is not None and last_seen_opponent[r, c]:
                        # Was enemy territory — assume they still hold it
                        return max(3, timestep // 40)
                    # Was passable empty/neutral — cheap
                    return 1
                # Never seen — unknown, moderate cost
                return max(2, timestep // 50)
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

    @staticmethod
    def _multi_source_bfs(
        sources: list[tuple[int, int]], passable: np.ndarray, H: int, W: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Multi-source BFS — returns distance and flow direction to nearest source."""
        dist = np.full((H, W), _INF, dtype=np.int32)
        flow = np.full((H, W), -1, dtype=np.int32)
        queue: deque[tuple[int, int]] = deque()
        for r, c in sources:
            if 0 <= r < H and 0 <= c < W and passable[r, c]:
                dist[r, c] = 0
                queue.append((r, c))
        while queue:
            r, c = queue.popleft()
            d = dist[r, c] + 1
            for direction in range(4):
                nr = r + _DR[direction]
                nc = c + _DC[direction]
                if 0 <= nr < H and 0 <= nc < W and passable[nr, nc] and dist[nr, nc] > d:
                    dist[nr, nc] = d
                    flow[nr, nc] = _REVERSE_DIR[direction]
                    queue.append((nr, nc))
        return dist, flow

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------

    def _select_gather_point(self, observation: Observation) -> tuple[int, int] | None:
        """Pick a frontier owned cell closest to the target — gather ON the
        front line so armies push outward, not pile up in the interior.
        In city_rush mode, gather toward the city rush target instead."""
        # In city rush mode, gather toward the city
        if self._city_rush_target is not None:
            owned = np.asarray(observation.owned_cells)
            cr, cc = self._city_rush_target
            # Find the owned cell closest to the city
            owned_positions = np.argwhere(owned)
            if len(owned_positions) == 0:
                return self._own_general
            dists = np.abs(owned_positions[:, 0] - cr) + np.abs(owned_positions[:, 1] - cc)
            best_idx = int(np.argmin(dists))
            return (int(owned_positions[best_idx, 0]), int(owned_positions[best_idx, 1]))

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
            dist = int(self._distance_map[r, c])
            if dist >= _INF:
                continue
            is_frontier = self._is_frontier(r, c, owned, H, W)
            frontier_bonus = 20.0 if is_frontier else 0.0
            local_army = int(armies[r, c])
            score = frontier_bonus + local_army * 0.3 - dist * 4.0
            if score > best_score:
                best_score = score
                best_pos = (r, c)
        return best_pos if best_pos is not None else self._own_general

    def _find_city_rush_target(self, observation: Observation) -> tuple[tuple[int, int], int] | None:
        """Find the best city to rush — closest capturable city where we have
        enough concentrated army within BFS distance 4 to capture it."""
        cities = np.asarray(observation.cities)
        owned = np.asarray(observation.owned_cells)
        armies = np.asarray(observation.armies)
        mountains = np.asarray(observation.mountains)
        H, W = armies.shape

        # Only visible cities (not fogged guesses — too unreliable for rush)
        candidates: list[tuple[int, int, int]] = []
        for r, c in np.argwhere(cities & ~owned).tolist():
            garrison = int(armies[r, c])
            if garrison <= 0:
                garrison = 1
            candidates.append((int(r), int(c), garrison))

        if not candidates:
            return None

        passable = ~mountains
        best: tuple[tuple[int, int], int, float] | None = None
        for cr, cc, garrison in candidates:
            # BFS from city, tight radius of 4 — army must be CLOSE
            dist = np.full((H, W), _INF, dtype=np.int32)
            dist[cr, cc] = 0
            queue: deque[tuple[int, int]] = deque()
            queue.append((cr, cc))
            while queue:
                r, c = queue.popleft()
                d = dist[r, c] + 1
                if d > 4:
                    break
                for direction in range(4):
                    nr = r + _DR[direction]
                    nc = c + _DC[direction]
                    if 0 <= nr < H and 0 <= nc < W and passable[nr, nc] and dist[nr, nc] > d:
                        dist[nr, nc] = d
                        queue.append((nr, nc))

            # Sum movable army (army - 1) of owned cells within range
            in_range = owned & (dist <= 4)
            nearby_movable = int(np.sum((armies - 1) * in_range))
            # Need a strong surplus — movable army must clearly exceed garrison
            # to avoid wasting turns gathering when we should be expanding
            if nearby_movable > garrison * 1.8:
                # Score: prefer cities we can capture with less distance
                avg_dist = float(np.mean(dist[in_range])) if np.any(in_range) else 4.0
                score = (nearby_movable - garrison) / max(avg_dist, 1.0)
                if best is None or score > best[2]:
                    best = ((cr, cc), garrison, score)

        if best is not None:
            return best[0], best[1]
        return None

    def _select_plan_mode(self, observation: Observation) -> str:
        if self._phase == "defense":
            return "defend"

        owned = np.asarray(observation.owned_cells)
        armies = np.asarray(observation.armies)
        owned_armies = armies * owned
        total_army = int(observation.owned_army_count)
        opp_army = int(observation.opponent_army_count)
        land = int(observation.owned_land_count)

        # Track stagnation
        if land <= self._last_land_count:
            self._stagnation_counter += 1
        else:
            self._stagnation_counter = 0
        self._last_land_count = land

        # Anti-stagnation: force expansion when stuck (highest priority)
        if self._stagnation_counter >= 15:
            self._city_rush_target = None
            self._city_rush_garrison = 0
            return "desperate_expand"

        # City rush: check if we can capture a city (only if not stagnating)
        city_rush = self._find_city_rush_target(observation)
        if city_rush is not None:
            self._city_rush_target, self._city_rush_garrison = city_rush
            return "city_rush"
        else:
            self._city_rush_target = None
            self._city_rush_garrison = 0

        # Scout when we haven't found enemy general yet
        if self._enemy_general_estimate is None:
            return "scout"

        # Find our strongest stack
        best_army = 0
        best_r, best_c = 0, 0
        if owned_armies.size > 0:
            best_idx = int(np.argmax(owned_armies))
            best_r, best_c = divmod(best_idx, armies.shape[1])
            best_army = int(armies[best_r, best_c])

        # Attack if our strongest stack can fight through to the target
        if self._cost_map is not None and best_army > 0:
            cost = int(self._cost_map[best_r, best_c])
            if cost < _INF and best_army > cost * 1.1:
                return "attack"

        # Force attack if we have overwhelming army advantage
        if total_army > opp_army * 1.5 and best_army >= 30:
            return "attack"

        if best_army >= 80:
            return "attack"

        # Gather only when we have decent army but need concentration
        if total_army > 30 and best_army < 30:
            return "gather"

        if self._phase in ("opening", "expansion"):
            return "scout"

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

        # BFS distance + flow direction to strategic target
        self._distance_map, self._flow_direction = self._bfs_from(target, passable, H, W)

        # Dijkstra army cost (with persistent memory for better fog estimates)
        self._cost_map = self._dijkstra_cost_from(
            target, passable, armies, owned, opponent, fog, structures_fog, timestep, H, W,
            seen=self._seen, last_seen_opponent=self._last_seen_opponent,
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

        # City distance map — multi-source BFS from all visible capturable cities
        cities = np.asarray(observation.cities)
        city_sources: list[tuple[int, int]] = []
        for r, c in np.argwhere(cities & ~owned).tolist():
            city_sources.append((int(r), int(c)))
        if self._last_seen_cities is not None:
            for r, c in np.argwhere(self._last_seen_cities & ~cities & ~owned).tolist():
                city_sources.append((int(r), int(c)))
        if city_sources:
            self._city_dist_map, self._city_flow = self._multi_source_bfs(
                city_sources, passable, H, W
            )
        else:
            self._city_dist_map = None
            self._city_flow = None

        # Frontier distance map — BFS from all frontier cells (owned cells
        # adjacent to non-owned).  Used by the pipeline bonus to identify
        # cells along the general→frontier corridor.
        frontier_sources: list[tuple[int, int]] = []
        owned_positions = np.argwhere(owned)
        for pos in owned_positions:
            r, c = int(pos[0]), int(pos[1])
            if self._is_frontier(r, c, owned, H, W):
                frontier_sources.append((r, c))
        if frontier_sources:
            self._frontier_dist_map, _ = self._multi_source_bfs(
                frontier_sources, passable, H, W
            )
        else:
            self._frontier_dist_map = None

        # Own-territory distance map — BFS from all owned cells.  Used by
        # the coverage bonus to reward expanding into areas far from existing
        # territory (wider expansion).
        owned_sources = [(int(r), int(c)) for r, c in owned_positions.tolist()]
        if owned_sources:
            self._own_territory_dist, _ = self._multi_source_bfs(
                owned_sources, passable, H, W
            )
        else:
            self._own_territory_dist = None

        # Scout target — when enemy general unknown, BFS toward the best
        # enemy-side structure in fog for active scouting
        self._scout_target = None
        self._scout_dist_map = None
        self._scout_flow = None
        if self._enemy_general_estimate is None or (
            self._last_seen_generals is not None
            and not np.any(self._last_seen_generals)
        ):
            # Haven't directly seen enemy general — set up scout target
            scout_candidates: list[tuple[int, int, float]] = []
            home = self._own_general or (H // 2, W // 2)

            # Structures in fog on the enemy side
            for r, c in np.argwhere(structures_fog).tolist():
                home_dist = abs(r - home[0]) + abs(c - home[1])
                if home_dist > 5:
                    scout_candidates.append((int(r), int(c), home_dist))

            # Also unseen fog areas far from home
            if not scout_candidates:
                for r, c in np.argwhere(fog & ~structures_fog).tolist():
                    home_dist = abs(r - home[0]) + abs(c - home[1])
                    if home_dist > max(H, W) // 2:
                        scout_candidates.append((int(r), int(c), home_dist * 0.5))

            if scout_candidates:
                scout_candidates.sort(key=lambda x: -x[2])
                sr, sc, _ = scout_candidates[0]
                self._scout_target = (sr, sc)
                self._scout_dist_map, self._scout_flow = self._bfs_from(
                    (sr, sc), passable, H, W
                )

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

    def _is_frontier(self, r: int, c: int, owned: np.ndarray, H: int, W: int) -> bool:
        """Check if (r, c) is a frontier cell — owned with a non-owned neighbor."""
        for d in range(4):
            nr, nc = r + _DR[d], c + _DC[d]
            if 0 <= nr < H and 0 <= nc < W and not owned[nr, nc]:
                return True
        return False

    def _count_fog_neighbors(self, r: int, c: int, fog: np.ndarray, H: int, W: int) -> int:
        count = 0
        for d in range(4):
            nr, nc = r + _DR[d], c + _DC[d]
            if 0 <= nr < H and 0 <= nc < W and fog[nr, nc]:
                count += 1
        return count

    def score_move(self, observation: Observation, move: MoveFeatures) -> float:
        score = 0.0
        army = move.source_army
        moving = move.moving_army
        owned = np.asarray(observation.owned_cells)
        fog = np.asarray(observation.fog_cells)
        H, W = owned.shape
        land = int(observation.owned_land_count)
        timestep = int(observation.timestep)
        is_opening = timestep < 60 and land < 25

        # Instant win — but only if it's the ENEMY general, not ours
        if move.dest_is_general and move.can_capture and not move.dest_is_owned:
            return 1e6

        # --- Army weight: big armies follow the path, small armies explore ---
        army_weight = 1.0 if army < 10 else 1.0 + (army - 10) * 0.15

        # === 1. PATH ALIGNMENT (scaled by army weight) ===
        # During opening, reduce path alignment so it doesn't dominate expansion
        path_scale = 8.0 if is_opening else 15.0
        if self._distance_map is not None:
            source_dist = int(self._distance_map[move.row, move.col])
            dest_dist = int(self._distance_map[move.dest_row, move.dest_col])
            if source_dist < _INF and dest_dist < _INF:
                path_progress = source_dist - dest_dist
                score += path_scale * path_progress * army_weight
                if self._flow_direction is not None:
                    optimal_dir = int(self._flow_direction[move.row, move.col])
                    if optimal_dir >= 0 and move.direction == optimal_dir:
                        score += (5.0 if is_opening else 10.0) * army_weight

        # === 2. COST FEASIBILITY ===
        if self._cost_map is not None:
            cost_here = int(self._cost_map[move.row, move.col])
            if cost_here < _INF:
                if moving > cost_here * 0.5:
                    score += 25.0 + moving * 0.5
                elif moving > cost_here * 0.2:
                    score += 10.0
                elif not move.dest_is_owned:
                    score -= 8.0

        # === 3. GENERAL EVACUATION — don't hoard on the general ===
        if self._own_general is not None:
            gr, gc = self._own_general
            if move.row == gr and move.col == gc and army >= 3:
                # Strong evacuation — scale with army size
                score += 35.0 + army * 0.8
                # Bonus for moving toward the frontier (pipeline direction)
                if self._frontier_dist_map is not None:
                    src_fd = int(self._frontier_dist_map[move.row, move.col])
                    dst_fd = int(self._frontier_dist_map[move.dest_row, move.dest_col])
                    if src_fd < _INF and dst_fd < _INF and dst_fd < src_fd:
                        score += 15.0  # moving toward frontier

        # === 4. ARMY PIPELINE — feed armies from interior to frontier ===
        # Cells between general and frontier get a bonus for forwarding army
        # in the direction of the frontier.  This creates a consistent flow.
        if (
            move.dest_is_owned
            and self._frontier_dist_map is not None
            and self._own_general is not None
            and self._plan_mode not in ("gather", "defend", "city_rush")
        ):
            src_fd = int(self._frontier_dist_map[move.row, move.col])
            dst_fd = int(self._frontier_dist_map[move.dest_row, move.dest_col])
            if src_fd < _INF and dst_fd < _INF and dst_fd < src_fd:
                # Moving toward frontier through owned territory = pipeline
                pipeline_bonus = 12.0 + army * 0.3
                # Stronger when the source is close to general (interior)
                gen_dist = abs(move.row - self._own_general[0]) + abs(move.col - self._own_general[1])
                if gen_dist <= 4:
                    pipeline_bonus += 8.0
                score += pipeline_bonus

        # === 5. TERRITORIAL EXPANSION — branch outward ===
        if not move.dest_is_owned and move.can_capture:
            if move.dest_is_neutral or move.dest_is_fog:
                score += 40.0  # strong base incentive
                # Branching bonus: more unexplored neighbors = better expansion point
                dest_fog_neighbors = self._count_fog_neighbors(
                    move.dest_row, move.dest_col, fog, H, W
                )
                score += dest_fog_neighbors * 5.0
                # Coverage bonus: reward expanding into areas far from existing territory
                if self._own_territory_dist is not None:
                    territory_dist = int(self._own_territory_dist[move.dest_row, move.dest_col])
                    if territory_dist >= 2:
                        score += min(territory_dist * 4.0, 20.0)  # up to +20 for distant cells
                # Early game urgency
                if is_opening:
                    score += 20.0

        # === 6. CITY INVESTMENT & CITY RUSH ===
        if move.dest_is_city and move.can_capture:
            score += 100.0  # city capture is extremely high value
        elif move.dest_is_city and not move.can_capture and moving > move.dest_army * 0.5:
            score += 50.0  # close to capturing — keep pushing
        elif self._city_dist_map is not None:
            src_city_dist = int(self._city_dist_map[move.row, move.col])
            dst_city_dist = int(self._city_dist_map[move.dest_row, move.dest_col])
            if src_city_dist < _INF and dst_city_dist < _INF:
                city_progress = src_city_dist - dst_city_dist
                if city_progress > 0 and army >= 10:
                    score += 20.0 * city_progress

        # City rush mode — massive bonus for gathering toward the target city
        if self._plan_mode == "city_rush" and self._city_rush_target is not None:
            cr, cc = self._city_rush_target
            src_dist = abs(move.row - cr) + abs(move.col - cc)
            dst_dist = abs(move.dest_row - cr) + abs(move.dest_col - cc)
            progress = src_dist - dst_dist
            if progress > 0:
                score += 60.0 + army * 0.5  # strong pull toward city
            # Gathering through owned territory toward city
            if move.dest_is_owned and progress > 0:
                score += 30.0 + army * 0.3
            # At the city — attack it
            if move.dest_row == cr and move.dest_col == cc:
                if move.can_capture:
                    score += 120.0
                else:
                    score += 40.0 + moving * 0.5  # weaken it

        # === 7. COMBAT — capturing opponent cells ===
        if move.dest_is_opponent and move.can_capture:
            score += 60.0 + moving * 0.3
            dest_fog_n = self._count_fog_neighbors(
                move.dest_row, move.dest_col, fog, H, W
            )
            if dest_fog_n > 0:
                score += 20.0  # breaking through to scout behind enemy lines

        # === 8. SCOUTING & EXPLORATION ===
        if move.dest_is_fog:
            fog_bonus = 20.0 if army < 5 else 12.0
            score += fog_bonus
            if self._enemy_general_estimate is not None:
                er, ec = self._enemy_general_estimate
                curr_dist = abs(move.row - er) + abs(move.col - ec)
                dest_dist_to_enemy = abs(move.dest_row - er) + abs(move.dest_col - ec)
                if dest_dist_to_enemy < curr_dist:
                    score += 15.0
        if move.dest_is_structure_fog:
            score += 18.0
            if self._own_general is not None:
                home_dist = abs(move.dest_row - self._own_general[0]) + abs(move.dest_col - self._own_general[1])
                if home_dist > 6:
                    score += 25.0

        # Active scouting mission — dedicated BFS toward enemy-side fog structures
        if self._scout_dist_map is not None and army >= 5:
            src_sd = int(self._scout_dist_map[move.row, move.col])
            dst_sd = int(self._scout_dist_map[move.dest_row, move.dest_col])
            if src_sd < _INF and dst_sd < _INF and dst_sd < src_sd:
                # Moving toward scout target
                scout_bonus = 18.0
                # Stronger for medium armies (good scouts, not wasted big stacks)
                if 5 <= army <= 20:
                    scout_bonus += 10.0
                # Bonus for pushing through enemy territory
                if move.dest_is_opponent and move.can_capture:
                    scout_bonus += 15.0
                score += scout_bonus

        # === 9. ENEMY GENERAL HUNT ===
        if move.dest_is_opponent and not move.can_capture:
            if self._enemy_general_estimate is not None:
                er, ec = self._enemy_general_estimate
                dest_dist_to_enemy = abs(move.dest_row - er) + abs(move.dest_col - ec)
                if dest_dist_to_enemy <= 5:
                    score += 8.0

        # === 10. REINFORCEMENT — penalize piling up, reward feeding the front ===
        if move.dest_is_owned and self._plan_mode not in ("gather", "defend", "city_rush"):
            dest_is_frontier = self._is_frontier(
                move.dest_row, move.dest_col, owned, H, W
            )
            if dest_is_frontier:
                score += 15.0 + army * 0.2
            else:
                # Penalize interior reinforcement unless it's pipeline movement
                # (pipeline bonus in section 4 counteracts this for good moves)
                score -= 10.0
                if self._distance_map is not None:
                    src_d = int(self._distance_map[move.row, move.col])
                    dst_d = int(self._distance_map[move.dest_row, move.dest_col])
                    if src_d < _INF and dst_d < _INF and dst_d < src_d:
                        score += 4.0

        # === 11. GATHERING ===
        if self._plan_mode == "gather" and self._gather_map is not None and move.dest_is_owned:
            src_gather = int(self._gather_map[move.row, move.col])
            dst_gather = int(self._gather_map[move.dest_row, move.dest_col])
            if src_gather < _INF and dst_gather < _INF:
                gather_progress = src_gather - dst_gather
                score += 12.0 * gather_progress + 8.0
                score += army * 0.4

        # === 12. DEFENSE ===
        if self._plan_mode == "defend" and self._own_general is not None:
            home_dist = self._distance_from_own_general(move)
            if home_dist <= 4:
                score += 15.0 - home_dist * 2.0

        # === 13. DESPERATE EXPAND — anti-stagnation escalation ===
        if self._plan_mode == "desperate_expand":
            if not move.dest_is_owned and move.can_capture:
                score += 50.0  # massive expansion bonus
                if move.dest_is_fog:
                    score += 20.0
            elif move.dest_is_owned:
                # Only allow reinforcement toward frontier
                dest_is_frontier = self._is_frontier(
                    move.dest_row, move.dest_col, owned, H, W
                )
                if not dest_is_frontier:
                    score -= 25.0  # strongly suppress interior shuffling

        return score

    def choose_split(self, observation: Observation, move: MoveFeatures) -> int:
        # Never split when moving off the general — evacuate fully
        if self._own_general is not None:
            if move.row == self._own_general[0] and move.col == self._own_general[1]:
                return 0
        # Never split when attacking — send full force
        if move.dest_is_opponent or move.dest_is_fog or move.dest_is_structure_fog:
            return 0
        # Never split when heading toward a city we want to capture
        if move.dest_is_city:
            return 0
        # Never split in city rush mode — concentrate everything
        if self._plan_mode == "city_rush":
            return 0
        # Never split in desperate expand — need all force at frontier
        if self._plan_mode == "desperate_expand":
            return 0
        # Don't split pipeline moves (moving toward frontier with army >= 8)
        if move.dest_is_owned and self._frontier_dist_map is not None:
            src_fd = int(self._frontier_dist_map[move.row, move.col])
            dst_fd = int(self._frontier_dist_map[move.dest_row, move.dest_col])
            if src_fd < _INF and dst_fd < _INF and dst_fd < src_fd:
                # Moving toward frontier — don't split big stacks
                if move.source_army >= 8:
                    return 0
        # Don't split big stacks moving toward the target
        if move.dest_is_owned and move.source_army >= 15:
            if self._distance_map is not None:
                src_d = int(self._distance_map[move.row, move.col])
                dst_d = int(self._distance_map[move.dest_row, move.dest_col])
                if src_d < _INF and dst_d < _INF and dst_d < src_d:
                    return 0
        # Split small reinforcements moving laterally through territory
        if move.dest_is_owned and move.source_army >= 10:
            return 1
        return 0

    def get_debug_snapshot(self) -> dict | None:
        return self._last_debug
