from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=['grid_dims', 'pad_to', 'mountain_density_range', 'num_cities_range',
                                    'min_generals_distance', 'max_generals_distance', 'castle_val_range',
                                    'spawn_candidate_count', 'terrain_candidate_count'])
def generate_grid(
    key: jax.random.PRNGKey,
    grid_dims: tuple[int, int] = (23, 23),
    pad_to: int | None = None,
    mountain_density_range: tuple[float, float] = (0.18, 0.26),
    num_cities_range: tuple[int, int] = (9, 11),
    min_generals_distance: int = 17,
    max_generals_distance: int | None = None,
    castle_val_range: tuple[int, int] = (40, 51),
    spawn_candidate_count: int = 3,
    terrain_candidate_count: int = 4,
) -> jnp.ndarray:
    """
    Generate a valid grid with stronger fairness and opening-space constraints.

    The generator still preserves asymmetry and randomness, but it now biases
    maps toward:
    1. interior, well-separated spawns
    2. clear local openings around each general
    3. terrain candidates with similar center/frontier access
    4. city placement that avoids large early-access disparities
    5. connectivity without relying solely on late-stage rerolls
    
    Args:
        key: JAX random key
        grid_dims: Grid dimensions (height, width) - supports non-square grids
        pad_to: Pad grid to this size for batching (None = max(h, w) + 1)
        mountain_density_range: (min, max) fraction of tiles that are mountains
        num_cities_range: (min, max) number of cities to place
        min_generals_distance: Minimum BFS (shortest path) distance between generals
        max_generals_distance: Maximum BFS (shortest path) distance between generals (None = no limit)
        castle_val_range: (min, max) army value for cities
        
    Returns:
        Grid is always valid (validity=True always)
    """
    keys = jax.random.split(key, 120)

    h, w = grid_dims
    num_tiles = h * w

    # Random number of cities in range
    num_cities = jax.random.randint(keys[0], (), num_cities_range[0], num_cities_range[1] + 1)

    # Number of mountains: sample uniformly from density range
    min_mountains = int(mountain_density_range[0] * num_tiles)
    max_mountains = int(mountain_density_range[1] * num_tiles)
    num_mountains = jax.random.randint(keys[1], (), min_mountains, max_mountains + 1)

    base_grid = jnp.full(grid_dims, 0, dtype=jnp.int32)

    # Step 1: sample several spawn pairs and several terrain layouts per pair.
    margin = jnp.where(jnp.minimum(h, w) >= 12, 2, 1)
    row_idx = jnp.arange(h)[:, None]
    col_idx = jnp.arange(w)[None, :]
    interior = (
        (row_idx >= margin)
        & (row_idx < h - margin)
        & (col_idx >= margin)
        & (col_idx < w - margin)
    )
    center_dist_candidate = jnp.abs(row_idx - (h // 2)) + jnp.abs(col_idx - (w // 2))
    overall_candidate_grids = []
    overall_candidate_scores = []
    for spawn_idx in range(spawn_candidate_count):
        spawn_key_offset = 2 + spawn_idx * 4
        horizontal_split = jax.random.bernoulli(keys[spawn_key_offset])
        left_band = interior & (col_idx <= (w // 2 - 1))
        right_band = interior & (col_idx >= (w // 2))
        top_band = interior & (row_idx <= (h // 2 - 1))
        bottom_band = interior & (row_idx >= (h // 2))

        spawn_a_band = jnp.where(horizontal_split, left_band, top_band)
        spawn_b_band = jnp.where(horizontal_split, right_band, bottom_band)
        spawn_a_band = jnp.where(jnp.any(spawn_a_band), spawn_a_band, interior)
        spawn_b_band = jnp.where(jnp.any(spawn_b_band), spawn_b_band, interior)

        spawn_a_pref = (
            -0.8 * jnp.where(horizontal_split, jnp.abs(row_idx - (h // 2)), jnp.abs(col_idx - (w // 2))).astype(jnp.float32)
            -0.2 * center_dist_candidate.astype(jnp.float32)
        )
        pos_first = sample_weighted_from_mask(spawn_a_band, spawn_a_pref, keys[spawn_key_offset + 1])
        dist_from_first = manhattan_distance_from(pos_first, grid_dims)
        separation_bias = jnp.where(horizontal_split, jnp.abs(row_idx - pos_first[0]), jnp.abs(col_idx - pos_first[1]))
        second_valid = spawn_b_band & (dist_from_first >= min_generals_distance)
        second_valid = second_valid & (separation_bias >= jnp.maximum(1, jnp.minimum(h, w) // 5))
        if max_generals_distance is not None:
            second_valid = second_valid & (dist_from_first <= max_generals_distance)
        second_fallback = interior & (dist_from_first >= min_generals_distance)
        if max_generals_distance is not None:
            second_fallback = second_fallback & (dist_from_first <= max_generals_distance)
        second_valid = jnp.where(jnp.any(second_valid), second_valid, second_fallback)
        mirrored_row = (h - 1) - pos_first[0]
        mirrored_col = (w - 1) - pos_first[1]
        center_dist_first = jnp.abs(pos_first[0] - (h // 2)) + jnp.abs(pos_first[1] - (w // 2))
        mirror_pref = jnp.where(
            horizontal_split,
            -jnp.abs(row_idx - mirrored_row).astype(jnp.float32),
            -jnp.abs(col_idx - mirrored_col).astype(jnp.float32),
        )
        second_pref = (
            0.25 * dist_from_first.astype(jnp.float32)
            + 0.75 * mirror_pref
            - 0.35 * jnp.abs(center_dist_candidate - center_dist_first).astype(jnp.float32)
        )
        pos_second = sample_weighted_from_mask(second_valid, second_pref, keys[spawn_key_offset + 2])

        swap = jax.random.bernoulli(keys[spawn_key_offset + 3])
        pos_a = jax.tree.map(lambda a, b: jnp.where(swap, b, a), pos_first, pos_second)
        pos_b = jax.tree.map(lambda a, b: jnp.where(swap, a, b), pos_first, pos_second)

        grid = base_grid.at[pos_a].set(1)
        grid = grid.at[pos_b].set(2)
        manhattan_a = manhattan_distance_from(pos_a, grid_dims)
        manhattan_b = manhattan_distance_from(pos_b, grid_dims)
        opening_buffer = (manhattan_a <= 2) | (manhattan_b <= 2)

        for terrain_idx in range(terrain_candidate_count):
            candidate_grid = grid
            key_offset = 14 + (spawn_idx * terrain_candidate_count + terrain_idx) * 8

            mountain_available = (candidate_grid == 0) & (~opening_buffer)
            centrality = -(jnp.abs(row_idx - (h // 2)) + jnp.abs(col_idx - (w // 2))).astype(jnp.float32)
            symmetry_bias = -jnp.abs(manhattan_a - manhattan_b).astype(jnp.float32)
            spawn_distance_bias = jnp.minimum(manhattan_a, manhattan_b).astype(jnp.float32)
            mountain_preference = 0.20 * centrality + 0.55 * symmetry_bias + 0.15 * spawn_distance_bias
            mountain_indices = weighted_top_k_from_mask(
                mountain_available,
                mountain_preference,
                num_tiles // 4,
                keys[key_offset],
            )
            flat_grid = candidate_grid.reshape(-1)
            flat_grid = place_values_at_indices(
                flat_grid,
                mountain_indices,
                jnp.full((num_tiles // 4,), -2, dtype=jnp.int32),
                num_mountains,
            )
            candidate_grid = flat_grid.reshape(grid_dims)

            # Keep the direct opening neighborhood clear even after mountain placement.
            candidate_grid = jnp.where(opening_buffer & (candidate_grid == -2), 0, candidate_grid)

            connected = flood_fill_connected(candidate_grid, pos_a, pos_b)
            candidate_grid = jax.lax.cond(
                connected,
                lambda g: g,
                lambda g: carve_l_path(g, pos_a, pos_b),
                candidate_grid,
            )
            if max_generals_distance is not None:
                dist = bfs_distance(candidate_grid, pos_a, pos_b)
                candidate_grid = jax.lax.cond(
                    dist > max_generals_distance,
                    lambda g: carve_l_path(g, pos_a, pos_b),
                    lambda g: g,
                    candidate_grid,
                )

            terrain_passable = candidate_grid != -2
            dist_a = bfs_distance_map(terrain_passable, pos_a)
            dist_b = bfs_distance_map(terrain_passable, pos_b)

            castle_val_a = jax.random.randint(keys[key_offset + 1], (), castle_val_range[0], castle_val_range[1])
            castle_val_b = jax.random.randint(keys[key_offset + 2], (), castle_val_range[0], castle_val_range[1])
            near_a_primary = (
                (candidate_grid == 0)
                & (~opening_buffer)
                & (dist_a >= 4)
                & (dist_a <= 6)
                & ((dist_b < 0) | (dist_a + 1 < dist_b))
            )
            near_a_secondary = (
                (candidate_grid == 0)
                & (~opening_buffer)
                & (dist_a >= 3)
                & (dist_a <= 8)
                & ((dist_b < 0) | (dist_a < dist_b))
            )
            near_a_fallback = (candidate_grid == 0) & (~opening_buffer) & (dist_a >= 3)
            castle_a_mask = first_nonempty_mask(near_a_primary, near_a_secondary, near_a_fallback)
            pos_castle_a = sample_from_mask(castle_a_mask, keys[key_offset + 3])
            candidate_grid = candidate_grid.at[pos_castle_a].set(castle_val_a)

            near_b_primary = (
                (candidate_grid == 0)
                & (~opening_buffer)
                & (dist_b >= 4)
                & (dist_b <= 6)
                & ((dist_a < 0) | (dist_b + 1 < dist_a))
            )
            near_b_secondary = (
                (candidate_grid == 0)
                & (~opening_buffer)
                & (dist_b >= 3)
                & (dist_b <= 8)
                & ((dist_a < 0) | (dist_b < dist_a))
            )
            near_b_fallback = (candidate_grid == 0) & (~opening_buffer) & (dist_b >= 3)
            castle_b_mask = first_nonempty_mask(near_b_primary, near_b_secondary, near_b_fallback)
            pos_castle_b = sample_from_mask(castle_b_mask, keys[key_offset + 4])
            candidate_grid = candidate_grid.at[pos_castle_b].set(castle_val_b)

            remaining_cities = jnp.maximum(0, num_cities - 2)
            city_slots = min(12, num_tiles)
            city_dist_target = 8.0

            side_a_mask = (
                (candidate_grid == 0)
                & (~opening_buffer)
                & (dist_a >= 5)
                & (dist_a <= 11)
                & ((dist_b < 0) | (dist_a + 1 < dist_b))
            )
            side_b_mask = (
                (candidate_grid == 0)
                & (~opening_buffer)
                & (dist_b >= 5)
                & (dist_b <= 11)
                & ((dist_a < 0) | (dist_b + 1 < dist_a))
            )
            contested_mask = (
                (candidate_grid == 0)
                & (~opening_buffer)
                & (jnp.minimum(dist_a, dist_b) >= 6)
                & (jnp.minimum(dist_a, dist_b) <= 12)
                & (jnp.abs(dist_a - dist_b) <= 2)
            )

            side_a_pref = -jnp.abs(dist_a.astype(jnp.float32) - city_dist_target) - 0.4 * jnp.abs((dist_a - dist_b).astype(jnp.float32))
            side_b_pref = -jnp.abs(dist_b.astype(jnp.float32) - city_dist_target) - 0.4 * jnp.abs((dist_a - dist_b).astype(jnp.float32))
            contested_pref = -jnp.abs(jnp.minimum(dist_a, dist_b).astype(jnp.float32) - (city_dist_target + 1.0)) - 0.6 * jnp.abs((dist_a - dist_b).astype(jnp.float32))

            contested_target = jnp.minimum(2, remaining_cities // 3)
            side_target = remaining_cities - contested_target
            side_a_target = side_target // 2
            side_b_target = side_target - side_a_target

            side_a_indices = weighted_top_k_from_mask(side_a_mask, side_a_pref, city_slots, keys[key_offset + 5])
            city_values_a = jax.random.randint(keys[key_offset + 6], (city_slots,), castle_val_range[0], castle_val_range[1])
            flat_grid = candidate_grid.reshape(-1)
            flat_grid = place_values_at_indices(flat_grid, side_a_indices, city_values_a, jnp.minimum(side_a_target, jnp.sum(side_a_mask)))
            candidate_grid = flat_grid.reshape(grid_dims)

            side_b_mask = side_b_mask & (candidate_grid == 0)
            contested_mask = contested_mask & (candidate_grid == 0)
            side_b_indices = weighted_top_k_from_mask(side_b_mask, side_b_pref, city_slots, keys[key_offset + 5] ^ jnp.uint32(13))
            city_values_b = jax.random.randint(keys[key_offset + 6] ^ jnp.uint32(29), (city_slots,), castle_val_range[0], castle_val_range[1])
            flat_grid = candidate_grid.reshape(-1)
            flat_grid = place_values_at_indices(flat_grid, side_b_indices, city_values_b, jnp.minimum(side_b_target, jnp.sum(side_b_mask)))
            candidate_grid = flat_grid.reshape(grid_dims)

            contested_mask = contested_mask & (candidate_grid == 0)
            contested_indices = weighted_top_k_from_mask(contested_mask, contested_pref, city_slots, keys[key_offset + 7])
            city_values_c = jax.random.randint(keys[key_offset + 6] ^ jnp.uint32(43), (city_slots,), castle_val_range[0], castle_val_range[1])
            flat_grid = candidate_grid.reshape(-1)
            flat_grid = place_values_at_indices(flat_grid, contested_indices, city_values_c, jnp.minimum(contested_target, jnp.sum(contested_mask)))
            candidate_grid = flat_grid.reshape(grid_dims)

            connected = flood_fill_connected(candidate_grid, pos_a, pos_b)
            candidate_grid = jax.lax.cond(
                connected,
                lambda g: g,
                lambda g: carve_l_path(g, pos_a, pos_b),
                candidate_grid,
            )

            candidate_score = score_layout(candidate_grid, pos_a, pos_b)
            overall_candidate_grids.append(candidate_grid)
            overall_candidate_scores.append(candidate_score)

    score_stack = jnp.stack(overall_candidate_scores)
    grid = jnp.stack(overall_candidate_grids)[jnp.argmax(score_stack)]

    # Step 3: Dynamic padding.
    # Default padding: max dimension + 1 (for batching)
    if pad_to is None:
        target_size = max(h, w) + 1
    else:
        target_size = pad_to
    
    # Pad both dimensions to target_size
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)
    
    if pad_h > 0 or pad_w > 0:
        grid = jnp.pad(
            grid,
            ((0, pad_h), (0, pad_w)),
            mode='constant',
            constant_values=-2,  # Mountains
        )
    
    # Grid is always valid by construction
    return grid


def sample_from_mask(mask: jax.Array, key: jax.random.PRNGKey) -> tuple[int, int]:
    """
    Sample one index from a boolean mask using Gumbel-max trick.
    XLA-efficient alternative to jax.random.choice.
    
    Args:
        mask: 2D boolean array where True indicates valid positions
        key: JAX random key
        
    Returns:
        (i, j) tuple of the sampled position
    """
    flat_mask = mask.reshape(-1).astype(jnp.float32)
    logits = jnp.where(flat_mask > 0, 0.0, -jnp.inf)
    gumbel_noise = jax.random.gumbel(key, shape=logits.shape)
    idx = jnp.argmax(logits + gumbel_noise)
    return jnp.unravel_index(idx, mask.shape)


def sample_weighted_from_mask(mask: jax.Array, preference: jax.Array, key: jax.random.PRNGKey) -> tuple[int, int]:
    """Sample one index from a mask using weighted Gumbel-max."""
    flat_mask = mask.reshape(-1)
    flat_pref = preference.reshape(-1)
    logits = jnp.where(flat_mask, flat_pref, -jnp.inf)
    gumbel_noise = jax.random.gumbel(key, shape=logits.shape)
    idx = jnp.argmax(logits + gumbel_noise)
    return jnp.unravel_index(idx, mask.shape)


def sample_k_from_mask(mask: jax.Array, k: int, key: jax.random.PRNGKey) -> jax.Array:
    """
    Sample k indices from a boolean mask using Gumbel-max trick + top_k.
    Maintains static shapes for XLA compatibility.
    
    Args:
        mask: 2D boolean array where True indicates valid positions
        k: Number of positions to sample (must be static)
        key: JAX random key
        
    Returns:
        Array of shape (k,) containing flat indices of sampled positions
    """
    flat_mask = mask.reshape(-1).astype(jnp.float32)
    logits = jnp.where(flat_mask > 0, 0.0, -jnp.inf)
    gumbel_noise = jax.random.gumbel(key, shape=logits.shape)
    scores = logits + gumbel_noise
    _, top_indices = jax.lax.top_k(scores, k)
    return top_indices


def weighted_top_k_from_mask(mask: jax.Array, preference: jax.Array, k: int, key: jax.random.PRNGKey) -> jax.Array:
    """Sample top-k positions from a mask using weighted Gumbel-max."""
    flat_mask = mask.reshape(-1)
    flat_pref = preference.reshape(-1)
    logits = jnp.where(flat_mask, flat_pref, -jnp.inf)
    gumbel_noise = jax.random.gumbel(key, shape=logits.shape)
    _, top_indices = jax.lax.top_k(logits + gumbel_noise, k)
    return top_indices


def place_values_at_indices(flat_grid: jax.Array, indices: jax.Array, values: jax.Array, count: jax.Array) -> jax.Array:
    """Place the first `count` values at the provided flat indices."""
    limit = jnp.minimum(count, indices.shape[0])

    def place_one(carry, args):
        idx, value, i = args
        carry = jax.lax.cond(
            i < limit,
            lambda arr: arr.at[idx].set(value),
            lambda arr: arr,
            carry,
        )
        return carry, None

    flat_grid, _ = jax.lax.scan(place_one, flat_grid, (indices, values, jnp.arange(indices.shape[0])))
    return flat_grid


def first_nonempty_mask(*masks: jax.Array) -> jax.Array:
    """Return the first mask with at least one true cell, else the last one."""
    selected = masks[-1]
    for mask in reversed(masks[:-1]):
        selected = jnp.where(jnp.any(mask), mask, selected)
    return selected


def _expand_frontier(mask: jax.Array) -> jax.Array:
    up = jnp.roll(mask, -1, axis=0).at[-1, :].set(False)
    down = jnp.roll(mask, 1, axis=0).at[0, :].set(False)
    left = jnp.roll(mask, -1, axis=1).at[:, -1].set(False)
    right = jnp.roll(mask, 1, axis=1).at[:, 0].set(False)
    return up | down | left | right


def bfs_distance_map(passable: jax.Array, start_pos: tuple[int, int]) -> jax.Array:
    """Return a full BFS distance map over passable cells."""
    h, w = passable.shape
    unreached = jnp.int32(h * w + 1)
    frontier = jnp.zeros((h, w), dtype=jnp.bool_).at[start_pos].set(True)
    seen = frontier
    dist = jnp.full((h, w), unreached, dtype=jnp.int32).at[start_pos].set(0)

    def body(step, state):
        frontier, seen, dist = state
        next_frontier = _expand_frontier(frontier) & passable & (~seen)
        dist = jnp.where(next_frontier, step + 1, dist)
        seen = seen | next_frontier
        return next_frontier, seen, dist

    frontier, seen, dist = jax.lax.fori_loop(0, h * w - 1, body, (frontier, seen, dist))
    del frontier, seen
    return jnp.where(dist == unreached, -1, dist)


def score_layout(grid: jax.Array, pos_a: tuple[int, int], pos_b: tuple[int, int]) -> jax.Array:
    """
    Score a layout for fairness/opening realism.

    This mirrors the diagnostic model in map_analysis closely enough that
    generation and evaluation stay on the same policy.
    """
    passable = grid != -2
    dist_a = bfs_distance_map(passable, pos_a)
    dist_b = bfs_distance_map(passable, pos_b)

    reachable_a = dist_a >= 0
    reachable_b = dist_b >= 0
    jointly_reachable = reachable_a & reachable_b
    closer_a = jointly_reachable & (dist_a < dist_b)
    closer_b = jointly_reachable & (dist_b < dist_a)
    territory_balance_ratio = jnp.abs(jnp.sum(closer_a) - jnp.sum(closer_b)) / jnp.maximum(jnp.sum(jointly_reachable), 1)

    center = (grid.shape[0] // 2, grid.shape[1] // 2)
    center_gap = jnp.abs(dist_a[center] - dist_b[center])

    opening_a = jnp.sum((dist_a >= 0) & (dist_a <= 4))
    opening_b = jnp.sum((dist_b >= 0) & (dist_b <= 4))
    frontier_a = jnp.sum((dist_a >= 0) & (dist_a <= 6))
    frontier_b = jnp.sum((dist_b >= 0) & (dist_b <= 6))
    frontier_ratio = jnp.abs(frontier_a - frontier_b) / jnp.maximum(frontier_a + frontier_b, 1)

    cities = grid > 2
    city_dists_a = jnp.where(cities, dist_a, jnp.int32(grid.shape[0] * grid.shape[1] + 1))
    city_dists_b = jnp.where(cities, dist_b, jnp.int32(grid.shape[0] * grid.shape[1] + 1))
    nearest_city_a = jnp.min(city_dists_a)
    nearest_city_b = jnp.min(city_dists_b)
    city_gap = jnp.abs(nearest_city_a - nearest_city_b)

    spawn_distance = dist_a[pos_b]
    score = jnp.float32(1.0)
    score -= jnp.minimum(territory_balance_ratio * 1.35, 0.32)
    score -= jnp.minimum(frontier_ratio * 0.85, 0.16)
    score -= jnp.minimum(jnp.abs(opening_a - opening_b) * 0.018, 0.12)
    score -= jnp.minimum(jnp.maximum(0, 20 - jnp.minimum(opening_a, opening_b)) * 0.028, 0.24)
    score -= jnp.minimum(jnp.maximum(0, 10 - spawn_distance) * 0.08, 0.30)
    score -= jnp.minimum(city_gap * 0.055, 0.16)
    score -= jnp.minimum(center_gap * 0.035, 0.12)
    return jnp.maximum(score, 0.0)


def manhattan_distance_from(pos: tuple[int, int], grid_shape: tuple[int, int]) -> jax.Array:
    """
    Compute Manhattan distance from a position to all cells in grid.
    
    Args:
        pos: (i, j) position
        grid_shape: (height, width) of grid
        
    Returns:
        2D array of Manhattan distances
    """
    h, w = grid_shape
    i_idx = jnp.arange(h)[:, None]
    j_idx = jnp.arange(w)[None, :]
    return jnp.abs(i_idx - pos[0]) + jnp.abs(j_idx - pos[1])


def valid_base_a_mask(grid_shape: tuple[int, int], min_distance: int, max_distance: int | None = None) -> jax.Array:
    """
    Create mask of valid positions for Base A.
    A position is valid if there exists at least one cell >= min_distance away
    (and optionally <= max_distance away).
    
    For a cell (i,j), the max Manhattan distance to any corner is:
    max(i+j, i+(w-1-j), (h-1-i)+j, (h-1-i)+(w-1-j))
    
    Args:
        grid_shape: (height, width) of grid
        min_distance: Minimum required distance to Base B
        max_distance: Maximum allowed distance to Base B (None = no limit)
        
    Returns:
        2D boolean mask
    """
    h, w = grid_shape
    i_idx = jnp.arange(h)[:, None]
    j_idx = jnp.arange(w)[None, :]
    
    # Distance to each corner
    dist_top_left = i_idx + j_idx
    dist_top_right = i_idx + (w - 1 - j_idx)
    dist_bottom_left = (h - 1 - i_idx) + j_idx
    dist_bottom_right = (h - 1 - i_idx) + (w - 1 - j_idx)
    
    max_dist = jnp.maximum(
        jnp.maximum(dist_top_left, dist_top_right),
        jnp.maximum(dist_bottom_left, dist_bottom_right)
    )
    
    # Min distance constraint
    valid = max_dist >= min_distance
    
    # Max distance constraint (if specified)
    # For max constraint, we need the min distance to any corner to be within max_distance
    if max_distance is not None:
        min_dist = jnp.minimum(
            jnp.minimum(dist_top_left, dist_top_right),
            jnp.minimum(dist_bottom_left, dist_bottom_right)
        )
        # At least one position should be reachable within max_distance
        # This means the grid diagonal should allow it
        grid_diagonal = h + w - 2
        # If we can't satisfy max_distance, just use min_distance
        valid = jnp.where(
            grid_diagonal >= max_distance,
            valid & (min_dist <= max_distance),
            valid
        )
    
    return valid


def bfs_reachable_within_k(grid: jax.Array, start_pos: tuple[int, int], k: int) -> jax.Array:
    """
    BFS flood fill from start_pos for exactly k steps.
    Returns boolean mask of all cells reachable within k BFS steps.
    The start cell itself is excluded from the result.

    Passable cells: not mountains (-2). Generals and empty cells are passable.

    Args:
        grid: 2D grid array
        start_pos: Starting position (i, j)
        k: Number of BFS steps

    Returns:
        2D boolean mask of reachable cells (excluding start_pos)
    """
    h, w = grid.shape
    passable = grid != -2

    reachable = jnp.zeros((h, w), dtype=jnp.bool_)
    reachable = reachable.at[start_pos].set(True)

    def dilate(reachable):
        up = jnp.roll(reachable, -1, axis=0).at[-1, :].set(False)
        down = jnp.roll(reachable, 1, axis=0).at[0, :].set(False)
        left = jnp.roll(reachable, -1, axis=1).at[:, -1].set(False)
        right = jnp.roll(reachable, 1, axis=1).at[:, 0].set(False)
        return (reachable | up | down | left | right) & passable

    def body_fn(_, reachable):
        return dilate(reachable)

    reachable = jax.lax.fori_loop(0, k, body_fn, reachable)

    # Exclude the start position itself
    reachable = reachable.at[start_pos].set(False)
    return reachable


def flood_fill_connected(grid: jax.Array, start_pos: tuple[int, int], end_pos: tuple[int, int]) -> bool:
    """
    Check if start_pos can reach end_pos using parallel flood fill with early termination.
    Uses jax.lax.while_loop for efficient early exit when target is reached.
    
    Args:
        grid: 2D grid array (-2=mountain, 0=passable, 1/2=generals, 40-50=cities)
        start_pos: Starting position (i, j)
        end_pos: Target position (i, j)
        
    Returns:
        Boolean indicating if end_pos is reachable from start_pos
    """
    h, w = grid.shape
    
    # Only empty tiles and generals are passable (not cities/castles)
    passable = (grid >= 0) & (grid <= 2)
    
    # Initialize: only start position is reachable
    reachable = jnp.zeros((h, w), dtype=jnp.bool_)
    reachable = reachable.at[start_pos].set(True)
    
    def dilate(reachable):
        """Single dilation step - expand reachable cells to neighbors."""
        # 4-neighbor dilation using roll + boundary fix
        up = jnp.roll(reachable, -1, axis=0).at[-1, :].set(False)
        down = jnp.roll(reachable, 1, axis=0).at[0, :].set(False)
        left = jnp.roll(reachable, -1, axis=1).at[:, -1].set(False)
        right = jnp.roll(reachable, 1, axis=1).at[:, 0].set(False)
        return (reachable | up | down | left | right) & passable
    
    def cond_fn(state):
        reachable, prev_reachable, _ = state
        # Continue if: target not reached AND frontier is still expanding
        target_reached = reachable[end_pos]
        still_expanding = jnp.any(reachable != prev_reachable)
        return ~target_reached & still_expanding
    
    def body_fn(state):
        reachable, _, step = state
        new_reachable = dilate(reachable)
        return (new_reachable, reachable, step + 1)
    
    # Initialize with one dilation already done
    initial_reachable = dilate(reachable)
    init_state = (initial_reachable, reachable, jnp.int32(1))
    
    final_reachable, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
    
    return final_reachable[end_pos]


def bfs_distance(grid: jax.Array, start_pos: tuple[int, int], end_pos: tuple[int, int]) -> jax.Array:
    """
    Compute shortest path (BFS) distance between two positions.
    Only mountains (-2) are impassable.

    Args:
        grid: 2D grid array
        start_pos: Starting position (i, j)
        end_pos: Target position (i, j)

    Returns:
        Scalar integer: BFS distance, or h*w if unreachable.
    """
    h, w = grid.shape
    # Only empty tiles and generals are passable (not cities/castles)
    passable = (grid >= 0) & (grid <= 2)

    reached = jnp.zeros((h, w), dtype=jnp.bool_)
    reached = reached.at[start_pos].set(True)

    def dilate(r):
        up = jnp.roll(r, -1, axis=0).at[-1, :].set(False)
        down = jnp.roll(r, 1, axis=0).at[0, :].set(False)
        left = jnp.roll(r, -1, axis=1).at[:, -1].set(False)
        right = jnp.roll(r, 1, axis=1).at[:, 0].set(False)
        return (r | up | down | left | right) & passable

    def cond_fn(state):
        r, prev_r, _ = state
        return ~r[end_pos] & jnp.any(r != prev_r)

    def body_fn(state):
        r, _, step = state
        return (dilate(r), r, step + 1)

    first = dilate(reached)
    final_r, _, final_step = jax.lax.while_loop(
        cond_fn, body_fn, (first, reached, jnp.int32(1))
    )

    return jnp.where(final_r[end_pos], final_step, h * w)


def carve_l_path(grid: jax.Array, pos_a: tuple[int, int], pos_b: tuple[int, int]) -> jax.Array:
    """
    Carve an L-shaped path between two positions using jnp.where (no branching).
    Clears mountains and cities on the path, preserves generals.
    
    The path goes: horizontal from pos_a to (pos_a[0], pos_b[1]), 
                   then vertical to pos_b.
    
    Args:
        grid: 2D grid array
        pos_a: Start position (i, j)
        pos_b: End position (i, j)
        
    Returns:
        Grid with L-shaped path carved (obstacles removed)
    """
    h, w = grid.shape
    i1, j1 = pos_a
    j2 = pos_b[1]
    i2 = pos_b[0]
    
    # Create coordinate grids
    i_idx = jnp.arange(h)[:, None]
    j_idx = jnp.arange(w)[None, :]
    
    # Horizontal segment: row i1, columns from min(j1, j2) to max(j1, j2)
    h_mask = (i_idx == i1) & \
             (j_idx >= jnp.minimum(j1, j2)) & \
             (j_idx <= jnp.maximum(j1, j2))
    
    # Vertical segment: column j2, rows from min(i1, i2) to max(i1, i2)
    v_mask = (j_idx == j2) & \
             (i_idx >= jnp.minimum(i1, i2)) & \
             (i_idx <= jnp.maximum(i1, i2))
    
    path_mask = h_mask | v_mask
    
    # Clear obstacles on path, but preserve generals (values 1, 2)
    is_obstacle = (grid == -2) | (grid > 2)  # Mountain or city
    grid = jnp.where(path_mask & is_obstacle, 0, grid)
    
    return grid
