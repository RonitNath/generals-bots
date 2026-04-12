# Map Diagnostics

Map generation currently guarantees validity, not strong competitive balance.

The fairness report is intended to tell you when a match is structurally suspect.

## Current Metrics

- `fairness_score`: coarse aggregate balance score
- `reject_map`: whether the map is likely too biased for evaluation
- `closer_to_p0` / `closer_to_p1`: territory reachability split
- `center_distance`: how quickly each side reaches the center
- `frontier_cells_within_5`: early expansion width near spawn
- `city_access`: nearest / mean / local city access

## Warning Meanings

### Territory Reachability Asymmetric

One side simply reaches more passable territory earlier than the other.

### Nearest City Access Differs Materially

One side can reach a city much faster, which can distort midgame economy.

### Center Access Differs Materially

One side gets to central influence sooner, usually increasing scouting and pressure options.

### Early Frontier Width Differs Materially

One side has wider safe opening expansion or fewer early chokepoints.

## Recommended Interpretation

- Casual testing: warnings are fine if you just want to watch strategic behavior.
- Strategy tuning: prefer maps without major warnings.
- Agent evaluation: reject maps flagged with `reject_map: true`.

## Current Limitation

This pass adds diagnostics only. It does not yet redesign the generator to produce mirrored or fairness-constrained maps.
