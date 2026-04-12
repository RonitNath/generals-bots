# Strategy Notes

This file distills the practical strategy ideas used for the heuristic agent pack.

Primary source inputs:

- the RL paper associated with this repo
- `docs/perpleixty_guide.md`

## Core Strategic Ideas

### Material Advantage

Land, army, and city control matter more than isolated local wins.

Useful rules:

- prefer enemy land over neutral land once contact is made
- city captures matter when they are defensible
- do not throw away concentrated army for low-value tiles

### Timing Around Income Pulses

The external strategy guide frames this around turn-25 pulses. In this repo’s current implementation, the exact growth timing differs, but the broad lesson still matters:

- swings near economy ticks are disproportionately valuable
- capturing land just before growth is better than just after

### Scouting Under Fog

Fog creates information asymmetry. Good heuristic play should:

- widen vision early
- preserve memory of last-seen enemy regions
- infer likely enemy-general regions from unexplored space and structure fog

### Backdoors and Flanks

Strong agents should not only march straight toward the nearest frontier.

Important behaviors:

- preserve alternate lanes
- threaten enemy rear territory
- keep pressure on the possibility of a general-line attack

### Defense and Counterattack

A good bot should not overextend just because it can expand.

Useful habits:

- maintain a home-defense buffer
- avoid leaving the general exposed while scouting deep
- convert defense into concentrated counterpressure

## Mapping To Built-In Agents

- `MaterialAdvantageAgent`: city timing, favorable trades, economy-aware expansion
- `ScoutPressureAgent`: scouting and enemy-location inference
- `BackdoorAgent`: deep route preference and indirect pressure
- `DefenseCounterAgent`: conservative defense-first posture
- `SurroundPressureAgent`: flank growth and multi-angle attack shaping
