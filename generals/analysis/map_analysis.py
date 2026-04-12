from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from generals.core.game import GameState


def _bfs_distances(passable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = passable.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    if not passable[start]:
        return dist

    queue: deque[tuple[int, int]] = deque([start])
    dist[start] = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        r, c = queue.popleft()
        base = dist[r, c]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            if not passable[nr, nc] or dist[nr, nc] >= 0:
                continue
            dist[nr, nc] = base + 1
            queue.append((nr, nc))
    return dist


def _mean_valid(values: np.ndarray) -> float | None:
    valid = values[values >= 0]
    if len(valid) == 0:
        return None
    return float(np.mean(valid))


def _city_metrics(cities: np.ndarray, dist: np.ndarray) -> dict[str, Any]:
    city_positions = np.argwhere(cities)
    if len(city_positions) == 0:
        return {
            "count": 0,
            "nearest_distance": None,
            "mean_distance": None,
            "cities_within_6": 0,
            "cities_within_10": 0,
        }

    dists = np.array([dist[r, c] for r, c in city_positions], dtype=np.int32)
    valid = dists[dists >= 0]
    return {
        "count": int(len(city_positions)),
        "nearest_distance": int(np.min(valid)) if len(valid) else None,
        "mean_distance": float(np.mean(valid)) if len(valid) else None,
        "cities_within_6": int(np.sum((dists >= 0) & (dists <= 6))),
        "cities_within_10": int(np.sum((dists >= 0) & (dists <= 10))),
    }


def _reachable_within(dist: np.ndarray, steps: int) -> int:
    return int(np.sum((dist >= 0) & (dist <= steps)))


def analyze_map_fairness(state: GameState) -> dict[str, Any]:
    """
    Compute a lightweight fairness report for an initial map.

    This is not a formal balance proof. It is a practical diagnostic to flag
    maps where one side appears to have much easier access to cities, territory,
    or the center of the board.
    """
    passable = np.array(state.passable, dtype=bool)
    cities = np.array(state.cities, dtype=bool)
    generals = [tuple(map(int, pos)) for pos in np.array(state.general_positions)]

    dist0 = _bfs_distances(passable, generals[0])
    dist1 = _bfs_distances(passable, generals[1])

    reachable0 = dist0 >= 0
    reachable1 = dist1 >= 0
    jointly_reachable = reachable0 & reachable1

    closer0 = jointly_reachable & (dist0 < dist1)
    closer1 = jointly_reachable & (dist1 < dist0)
    contested = jointly_reachable & (np.abs(dist0 - dist1) <= 2)

    city0 = _city_metrics(cities, dist0)
    city1 = _city_metrics(cities, dist1)
    city0["cities_within_15"] = int(np.sum((dist0 >= 0) & cities & (dist0 <= 15)))
    city1["cities_within_15"] = int(np.sum((dist1 >= 0) & cities & (dist1 <= 15)))
    opening_area_4 = [_reachable_within(dist0, 4), _reachable_within(dist1, 4)]
    opening_area_6 = [_reachable_within(dist0, 6), _reachable_within(dist1, 6)]
    spawn_distance = int(dist0[generals[1]]) if dist0[generals[1]] >= 0 else None

    center = (passable.shape[0] // 2, passable.shape[1] // 2)
    center_dist0 = int(dist0[center]) if dist0[center] >= 0 else None
    center_dist1 = int(dist1[center]) if dist1[center] >= 0 else None

    territory_balance = abs(int(np.sum(closer0)) - int(np.sum(closer1)))
    territory_balance_ratio = territory_balance / max(int(np.sum(jointly_reachable)), 1)
    frontier0 = int(np.sum((dist0 >= 0) & (dist0 <= 5)))
    frontier1 = int(np.sum((dist1 >= 0) & (dist1 <= 5)))
    chokepoint_ratio = abs(frontier0 - frontier1) / max(frontier0 + frontier1, 1)

    nearest_city_gap = None
    if city0["nearest_distance"] is not None and city1["nearest_distance"] is not None:
        nearest_city_gap = abs(city0["nearest_distance"] - city1["nearest_distance"])

    center_gap = None
    if center_dist0 is not None and center_dist1 is not None:
        center_gap = abs(center_dist0 - center_dist1)

    warnings: list[str] = []
    if territory_balance_ratio > 0.12:
        warnings.append("territory reachability is noticeably asymmetric")
    if nearest_city_gap is not None and nearest_city_gap >= 3:
        warnings.append("nearest city access differs materially")
    if center_gap is not None and center_gap >= 3:
        warnings.append("center access differs materially")
    if abs(city0["cities_within_10"] - city1["cities_within_10"]) >= 2:
        warnings.append("mid-range city density differs materially")
    if chokepoint_ratio > 0.18:
        warnings.append("early frontier width differs materially")
    if spawn_distance is not None and spawn_distance < 8:
        warnings.append("generals spawn too close for realistic openings")
    if min(opening_area_4) < 18:
        warnings.append("opening space is cramped near at least one spawn")
    if abs(opening_area_4[0] - opening_area_4[1]) >= 5:
        warnings.append("opening space differs materially between spawns")

    fairness_score = 1.0
    fairness_score -= min(territory_balance_ratio * 1.5, 0.35)
    fairness_score -= min((nearest_city_gap or 0) * 0.06, 0.18)
    fairness_score -= min((center_gap or 0) * 0.04, 0.12)
    fairness_score -= min(chokepoint_ratio * 0.8, 0.15)
    if spawn_distance is not None and spawn_distance < 8:
        fairness_score -= min((8 - spawn_distance) * 0.08, 0.32)
    if min(opening_area_4) < 18:
        fairness_score -= min((18 - min(opening_area_4)) * 0.03, 0.24)
    fairness_score -= min(abs(opening_area_4[0] - opening_area_4[1]) * 0.02, 0.12)
    fairness_score = max(0.0, fairness_score)
    reject_map = fairness_score < 0.65
    if spawn_distance is not None and spawn_distance < 8:
        reject_map = True
    if min(opening_area_4) < 18:
        reject_map = True

    return {
        "general_positions": [list(generals[0]), list(generals[1])],
        "spawn_distance": spawn_distance,
        "passable_cells": int(np.sum(passable)),
        "jointly_reachable_cells": int(np.sum(jointly_reachable)),
        "closer_to_p0": int(np.sum(closer0)),
        "closer_to_p1": int(np.sum(closer1)),
        "contested_cells": int(np.sum(contested)),
        "territory_balance_ratio": territory_balance_ratio,
        "center_distance": [center_dist0, center_dist1],
        "opening_area_within_4": opening_area_4,
        "opening_area_within_6": opening_area_6,
        "frontier_cells_within_5": [frontier0, frontier1],
        "frontier_asymmetry_ratio": chokepoint_ratio,
        "city_access": [city0, city1],
        "fairness_score": fairness_score,
        "reject_map": reject_map,
        "warnings": warnings,
        "recommendations": [
            "reroll the seed for competitive evaluation" if warnings else "map is acceptable for casual testing"
        ],
    }
