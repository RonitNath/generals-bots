# Generals.io Strategy Guide

## Overview

Generals.io is a fast-paced, browser-based multiplayer real-time strategy game where players compete on a tile grid to capture enemy generals while protecting their own. The game rewards efficient troop management, map awareness, and smart timing over raw aggression. Despite its simple look, it features deep strategic complexity — AI researchers have used it as a benchmark for reinforcement learning agents.[1][2]

***

## Core Mechanics (Know These First)

Understanding the underlying math of the game is essential before applying any strategy.

- **Turns & moves**: Each turn lasts 1 second, and you can make **2 moves per turn**. Using both moves every turn is one of the biggest skill separators.[3]
- **Rounds**: Every **25 turns**, all your occupied tiles gain +1 troop. This "pulse" is the single most important timing rhythm in the game.[3]
- **General**: Your crown tile generates 1 troop *per turn* (not just per round). If the enemy captures it, you lose.[4]
- **Cities**: Gray squares that cost ~40–50 troops to capture, but then generate 1 troop per turn just like your general — making them extremely valuable in the mid and late game.[5][3]
- **Fog of war**: You can only see tiles adjacent to tiles you own. Enemy movements are hidden until they touch your border — a critical asymmetry to exploit.[6]
- **Movement cost**: Moving troops off a tile always leaves 1 troop behind, so each move costs at least 1 army. Chaining moves without wasting them is the foundation of high-level play.[4]

***

## Early Game (Turns 0–25: The First Round)

### The 25-Land Opening

The first round is the most formulaic part of the game and where skilled players build decisive advantages. The goal is to **claim exactly 25 tiles by turn 25** — the moment the first round pulse fires. Reaching 25 land means the pulse gives you 25 troops, while someone at 20 land only gets 20.[7]

Optimal openings are named after how much army you generate before your first move. A "13-start" means waiting until your general has 13 army, then expanding out. Memorizing one or two standard openings is strongly recommended. Multi-directional openings (expanding in 3–4 directions) are better than going in a straight line — they give you more map vision to locate the enemy.[7]

### Scouting and Spawn Prediction

In 1v1, generals always spawn at least 15 tiles apart (Manhattan distance), so you can immediately rule out a large area around you as a possible enemy location. Expanding toward the center of the map is usually the right call because it shortens the distance to find the enemy's border. In FFA (8 players), generals spawn at least 9 tiles apart on a larger 30–40 tile map.[7]

### Early City Decisions

**Do not capture a neutral city in round 1.** A city costs ~50 troops — troops you don't have yet — and provides no net advantage over claiming blank tiles during the first round, since 25 blank tiles also yields 25 troops per round. Cities only become net-positive later when your land acquisition slows down. An early city grab signals to an opponent that your troop count just dropped, inviting a counterattack.[8][5]

***

## Mid Game: Expansion and Positioning

### Attack Enemy Land, Not Neutral Tiles

When you encounter an enemy's border, prioritize taking their tiles over neutral ones. Attacking enemy land swings the land advantage by **2 per half-turn** (you gain 1, they lose 1), whereas taking neutral land only swings it by 1. Additionally, captured enemy land denies them income. Make this trade as often as possible.[7]

### Timing: Strike Before the Pulse

A key mid-game technique is to **attack right before a round pulse (turn 25, 50, 75…)**. Tiles you capture just before the pulse immediately benefit from the +1 bonus, while the enemy loses that bonus on the same tiles. Conversely, use the beginning of a round to consolidate and collect troops, then expand aggressively toward the end of the round.[9][7]

### Entropy and Troop Collection

Spreading your army thin across many tiles ("high entropy") makes it hard to form an attack force. Advanced players **collect troops onto a single tile** before attacking, creating a concentrated strike force. The free army metric — `army - land` — tells you how much concentrated striking power you have. Keeping free army low (near 0 at round end) means you're using your resources efficiently.[10][7]

### The Blob Shape

A "blob" expansion pattern — spreading in a roughly circular/hexagonal shape around your general — is a strong default in FFA. It:[10]
- Keeps your general far from the fog-of-war edge, making surprise attacks harder
- Creates an even front to defend on all sides
- Avoids exposing long, thin corridors that are easy to cut off

### Surrounding the Enemy (1v1)

In 1v1, one of the most powerful positional moves is to **surround the enemy general from multiple angles**. When you have two separate attack paths to their general, they must defend both — halving their effective striking force. A single attack route is predictable and easy to wall off.[7]

***

## City Strategy

Cities are the main economic lever in the mid-to-late game. Each city you hold generates 1 extra troop per turn — equivalent to owning 25 more land tiles in terms of per-round income. The strategic calculus:[3]

| Situation | Recommendation |
|-----------|---------------|
| Early game (Round 1–2) | Skip cities; focus on land |
| Seen a neutral city near the frontline | Capture it if you can defend it; deny the enemy[11] |
| Unclaimed city visible from edge of fog | Pad it with 100+ troops to make capturing it cost-prohibitive[11] |
| Enemy just captured a city at low troops | Attack it immediately before it replenishes[11] |
| Falling behind in FFA | Cities can compensate for smaller land — buy one as a "farming" investment[10] |

Avoid capturing cities you cannot defend — a city at 1 troop is a gift to the enemy.[3]

***

## Key Tactics and Techniques

### Split Attacking (Z Key)

Pressing **Z** splits your current army stack in half, letting you send troops in two directions simultaneously. This is invaluable for:[12]
- Pressuring two flanks at once
- Keeping 50% of your troops back for defense while sending the other half to attack
- Misdirecting the opponent during an opening assault

### Backdooring

Backdooring means sneaking a small army **behind enemy lines** through an unexpected route to attack their general directly while their main army is elsewhere. This is one of the most game-winning advanced moves. Conditions for a successful backdoor:[1][8]
- Your troops are hidden in fog — the enemy doesn't see you coming
- Their main army is deployed far forward
- The path to their general is short or under-defended

Even the threat of a backdoor forces opponents to keep troops near their general at all times, limiting their offensive options.[5]

### Misdirection and Feinting

Moving a large stack toward the enemy from one direction while secretly routing your real attack from another is a high-level tactic. Misdirection forces the enemy to commit defensive troops to the wrong area. This is most effective when the opponent has limited fog visibility and cannot quickly reposition.[1][3]

### Defending Your General

The further your general is from the fog-of-war boundary, the harder it is to backdoor. Key defensive habits:[10]
- Keep 50–100+ troops on or adjacent to your general at all times
- Use the **Q key** (cancel all queued moves) to quickly reset and redirect to defense
- Count the tiles between your general and the enemy border — every tile of distance is "defense time"

***

## FFA-Specific Strategy

FFA (8 players) rewards patience and opportunism more than 1v1.[8]

### Early Game: Stay Passive
Avoid fighting before turn 25–45. Pre-turn 45 rushes usually stall into a costly war of attrition while other players grow undisturbed. Instead, expand quietly, get at least one city, and read the leaderboard.[8]

### Use the Leaderboard as Intel
The sidebar shows army and land counts for every player. If an opponent's land count suddenly stops growing, they're likely fighting someone — prime time to strike. If their army is much lower than yours, they're vulnerable.[10][7]

### Mexican Standoff (Final 3 Players)
When 3 players remain, aggression often benefits the third player watching the fight. The optimal play is to **minimize conflict** and let the other two weaken each other, then attack decisively once one is eliminated. Relinquishing a city you can't defend can even be used to persuade another player to target someone else.[10]

### The Blitzkrieg Finish
The FFA meta rewards building up quietly and then launching a decisive blitzkrieg — overwhelming a target before they can react. Collect a large concentrated army, locate your target's general, then strike from multiple angles simultaneously. Speed matters: the longer a fight drags on, the more other players benefit from your mutual weakening.[8]

***

## Controls Quick Reference

| Key | Action |
|-----|--------|
| WASD or Arrow Keys | Move selected army |
| Z | Split army in half |
| Q | Cancel all queued moves |
| Space | Deselect current tile |
| Click + WASD | Queue moves rapidly |

Using keyboard shortcuts is essential at higher levels — clicking with the mouse for every move is too slow to execute two moves per turn reliably.[12]

***

## Skill Progression Checklist

| Skill | Focus |
|-------|-------|
| Beginner | Move stacks twice per turn consistently; don't leave large armies idle |
| Intermediate | Execute 25-land opening; attack enemy land over neutral; time attacks near round pulses |
| Advanced | Split attacks; find and exploit backdoor routes; surround enemies; use leaderboard for intel |
| Expert | Optimize free army to near-zero each round; misdirection feints; predict enemy spawn from opening |
