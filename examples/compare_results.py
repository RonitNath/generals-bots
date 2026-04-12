"""
Compare two match or tournament summaries.

Examples:
    uv run python examples/compare_results.py --left logs/tournament-a/summary.json --right logs/tournament-b/summary.json
    uv run python examples/compare_results.py --left logs/match-a/summary.json --right logs/match-b/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: str):
    return json.loads(Path(path).read_text())


def is_tournament_summary(payload: dict) -> bool:
    return "pairings" in payload and "matches" in payload


def compare_counts(left: dict, right: dict) -> list[str]:
    keys = sorted(set(left) | set(right))
    lines = []
    for key in keys:
        left_val = left.get(key, 0)
        right_val = right.get(key, 0)
        delta = right_val - left_val
        sign = "+" if delta > 0 else ""
        lines.append(f"- {key}: {left_val} -> {right_val} ({sign}{delta})")
    return lines


def compare_match_summaries(left: dict, right: dict) -> str:
    lines = ["**Match Comparison**"]
    for field in ["winner_name", "turns", "keyframes", "events"]:
        lines.append(f"- {field}: {left.get(field)} -> {right.get(field)}")
    lines.append("- anomaly_scores:")
    left_scores = left.get("anomaly_scores", [])
    right_scores = right.get("anomaly_scores", [])
    for idx in range(max(len(left_scores), len(right_scores))):
        lv = left_scores[idx] if idx < len(left_scores) else 0
        rv = right_scores[idx] if idx < len(right_scores) else 0
        lines.append(f"  player {idx}: {lv} -> {rv} ({rv - lv:+.1f})")
    lines.append("- anomaly_counts:")
    lines.extend(compare_counts(left.get("anomaly_counts", {}), right.get("anomaly_counts", {})))
    return "\n".join(lines)


def compare_tournament_summaries(left: dict, right: dict) -> str:
    lines = ["**Tournament Comparison**"]
    lines.append(f"- matches: {left.get('matches')} -> {right.get('matches')}")
    lines.append("- global_anomaly_counts:")
    lines.extend(compare_counts(left.get("global_anomaly_counts", {}), right.get("global_anomaly_counts", {})))

    pair_keys = sorted(set(left.get("pairings", {})) | set(right.get("pairings", {})))
    for pair_key in pair_keys:
        left_pair = left.get("pairings", {}).get(pair_key, {})
        right_pair = right.get("pairings", {}).get(pair_key, {})
        lines.append(f"- pairing {pair_key}:")
        for field in ["matches", "draws", "accepted_maps"]:
            lines.append(f"  {field}: {left_pair.get(field)} -> {right_pair.get(field)}")
        for field in ["avg_turns", "avg_fairness_score", "avg_rerolls"]:
            lv = left_pair.get(field, 0.0)
            rv = right_pair.get(field, 0.0)
            lines.append(f"  {field}: {lv:.3f} -> {rv:.3f} ({rv - lv:+.3f})")
        lines.append("  anomaly_counts:")
        for line in compare_counts(left_pair.get("anomaly_counts", {}), right_pair.get("anomaly_counts", {})):
            lines.append(f"  {line[2:]}")
        left_wins = left_pair.get("wins", {})
        right_wins = right_pair.get("wins", {})
        lines.append("  wins:")
        for line in compare_counts(left_wins, right_wins):
            lines.append(f"  {line[2:]}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare two match or tournament summaries")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    args = parser.parse_args()

    left = load_json(args.left)
    right = load_json(args.right)

    if is_tournament_summary(left) and is_tournament_summary(right):
        print(compare_tournament_summaries(left, right))
    else:
        print(compare_match_summaries(left, right))


if __name__ == "__main__":
    main()
