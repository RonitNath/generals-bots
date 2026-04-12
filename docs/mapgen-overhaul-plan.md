# FFA-First Map Generation Rearchitecture

## Summary

Replace the current single-function JAX heuristic in `generals/core/grid.py` with a new `generals/mapgen/` pipeline that plans macro-structure first, realizes it into tiles second, and evaluates it with multi-objective critics instead of one coarse fairness score.

The redesign will optimize for balanced variety across a wide rectangular size range and make FFA the primary mode, with 1v1 handled as the smallest specialization of the same system. Variety will come from named archetypes, not random noise. V1 will also expand map semantics with watchtowers/control points and destructible blockers, but will not add new economy systems beyond cities.

## Research Basis

- [Generals.io 1v1 guide](https://wiki.generals.io/1v1guide.html): spawn shape and opening topology materially change strategy.
- [StarCraft II melee map design](https://code.tutsplus.com/starcraft-ii-level-design-introduction-and-melee-maps--gamedev-3304t): equal starts, contested center objectives, expansions, and chokepoints drive interesting RTS flow.
- [Dormans 2010](https://pcgworkshop.com/archive/dormans2010adventures.pdf): generate large-scale structure separately from space realization.
- [Liapis et al. 2015](https://www.antoniosliapis.com/papers/searching_for_good_and_diverse_game_levels.pdf): optimize both strategic quality and diversity; large maps need normalized objectives.
- [Togelius et al. 2010](https://www.um.edu.mt/library/oar/bitstream/123456789/29280/1/Multiobjective_exploration_of_the_starcraft_map_space_2010.pdf): use multi-objective evaluation instead of collapsing everything into one weighted scalar.
- [Kowalski and Szykuła 2018](https://jakubkowalski.tech/Publications/Kowalski2018StrategicFeatures.pdf): keep the generator modular around zones, obstacles, and strategic feature placement.

## Key Changes

- Create a staged pipeline:
  1. `ScalePolicy` chooses player-count bands, density budgets, and target metrics from map area/aspect ratio.
  2. `ArchetypePlanner` builds a sector graph with spawn sectors, safe hinterlands, contested hubs, and lane connections.
  3. `LayoutSynthesizer` rasterizes that graph into passable terrain, obstacle fields, and macro routes.
  4. `FeaturePasses` place spawns, cities, watchtowers, and destructible walls from role-based rules.
  5. `MapCritic` scores hard constraints and soft objectives, then triggers local repair or bounded rerolls.
  6. `Compiler` emits the runtime map artifact consumed by env/game code.

- Ship named archetypes in v1:
  - `open_center`
  - `split_lanes`
  - `ring_spokes`
  - `pocket_basins`
  - `braided_frontiers`

- Use area-based player bands as the default policy:
  - `<=256` cells: 2 players
  - `257-576`: 2-4 players
  - `577-1024`: 4-6 players
  - `1025-2304`: 6-8 players
  - `>2304`: 8-12 players
  - Reject a player count if the resulting spawn sectors cannot satisfy minimum sector diameter, opening room, and pairwise path-distance constraints.

- Add new map mechanics in v1:
  - `watchtower`: capturable control tile that grants fixed-radius vision.
  - `destructible_wall`: impassable tile with HP; attacks reduce HP until it becomes passable ground.
  - No roads, swamps, portals, or extra resource nodes in v1.

- Replace scalar fairness-only thinking with normalized objectives:
  - opening room per player
  - nearest city access
  - nearest watchtower access
  - first-contact distance
  - route multiplicity between sectors
  - contested-hub parity
  - sector territory balance
  - blocker leverage without permanent isolation

## Public Interfaces and Type Changes

- Replace `generate_grid(key, ...)` with `build_map(spec, key) -> MapArtifact`.
- Add `MapSpec` with:
  - `dims`
  - `format` (`duel` or `ffa`)
  - `player_count` (`int | auto`)
  - `archetype_weights`
  - `fairness_tier`
  - `enabled_elements`
  - `seed`
- Add `MapArtifact` with:
  - terrain layers
  - cities
  - watchtowers
  - destructible wall state
  - spawn positions
  - `player_count`
  - `archetype_id`
  - normalized metrics and diagnostics
- Generalize runtime state for FFA and new elements:
  - `GameState.ownership -> (P, H, W)`
  - `general_positions -> (P, 2)`
  - new layers for watchtowers and wall HP
  - `Observation` exposes per-player visibility/stat arrays plus landmark layers
- Update `GeneralsEnv` to build/reset from `MapSpec` or a `MapPoolBuilder`, not raw grid-only generation.

## Test Plan

- Property tests across `12x12` through `64x64` rectangular maps and all supported player bands:
  - connectivity
  - spawn clearance
  - no trapped starts
  - no isolated sectors after wall destruction
  - archetype-specific route-count guarantees

- Fairness tests per archetype and size band:
  - bounded normalized gaps for city access, tower access, opening room, and first-contact distance
  - no player with systematically easier safe expansion than others in the same band

- Variety tests:
  - generated batches must populate all enabled archetypes
  - archetype fingerprints must remain separable
  - repeated seeds within one archetype must vary at lane/resource/layout level

- Agent simulation tests in duel and FFA:
  - first contact timing
  - tower contest frequency
  - wall breach timing
  - city timing spread
  - elimination skew and spawn disadvantage outliers

- Performance tests:
  - benchmark pool generation by size band
  - keep map consumption fast in env runtime
  - allow slower hybrid planning during pool build, but not per-turn gameplay

## Assumptions and Defaults

- Clean break is accepted; no compatibility wrapper is planned.
- FFA is the primary target; 1v1 remains supported through the same architecture.
- "Arbitrary size" is treated as a wide practical envelope, not unbounded size; v1 explicitly targets `12x12` to `64x64`.
- Player-count policy lives in external config/data, not hardcoded inside generator logic.
- V1 scope is the new pipeline plus the minimum engine/observation changes required for FFA, watchtowers, and destructible walls; additional terrain/resource systems stay out of scope.
