"""
View or re-render a captured keyframe.
"""

import argparse
from pathlib import Path
import json

from generals.analysis.keyframes import deserialize_game_state, render_state_png
from generals.gui import ReplayGUI

parser = argparse.ArgumentParser(description="View a captured keyframe")
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--png-out", type=str, default=None)
parser.add_argument("--show-tile-types", action="store_true")
args = parser.parse_args()

payload = json.loads(Path(args.input).read_text())
state = deserialize_game_state(payload["state"])

if args.png_out:
    render_state_png(args.png_out, state)
    print(f"Wrote {args.png_out}")

gui = ReplayGUI(state, agent_ids=["Player 0", "Player 1"], show_tile_types=args.show_tile_types)
gui.update(state)
while True:
    gui.tick(fps=4)
