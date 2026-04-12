from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pygame

from generals.core.config import Dimension, Path as AssetPath
from generals.core.game import GameState

PLAYER_COLORS = ((220, 56, 56), (56, 120, 220))
EMPTY = (230, 230, 230)
FOG = (70, 73, 76)
MOUNTAIN = (187, 187, 187)
CITY_NEUTRAL = (128, 128, 128)
BLACK = (0, 0, 0)
WHITE = (230, 230, 230)


def serialize_game_state(state: GameState) -> dict[str, Any]:
    return {
        "armies": np.array(state.armies).tolist(),
        "ownership": np.array(state.ownership).astype(int).tolist(),
        "ownership_neutral": np.array(state.ownership_neutral).astype(int).tolist(),
        "generals": np.array(state.generals).astype(int).tolist(),
        "cities": np.array(state.cities).astype(int).tolist(),
        "mountains": np.array(state.mountains).astype(int).tolist(),
        "passable": np.array(state.passable).astype(int).tolist(),
        "general_positions": np.array(state.general_positions).tolist(),
        "time": int(state.time),
        "winner": int(state.winner),
        "pool_idx": int(state.pool_idx),
    }


def deserialize_game_state(payload: dict[str, Any]) -> GameState:
    return GameState(
        armies=jnp.array(payload["armies"], dtype=jnp.int32),
        ownership=jnp.array(payload["ownership"], dtype=bool),
        ownership_neutral=jnp.array(payload["ownership_neutral"], dtype=bool),
        generals=jnp.array(payload["generals"], dtype=bool),
        cities=jnp.array(payload["cities"], dtype=bool),
        mountains=jnp.array(payload["mountains"], dtype=bool),
        passable=jnp.array(payload["passable"], dtype=bool),
        general_positions=jnp.array(payload["general_positions"], dtype=jnp.int32),
        time=jnp.int32(payload["time"]),
        winner=jnp.int32(payload["winner"]),
        pool_idx=jnp.int32(payload.get("pool_idx", 0)),
    )


def write_keyframe_json(path: str | Path, state: GameState, reasons: list[str]):
    out = Path(path)
    out.write_text(
        json.dumps(
            {
                "reasons": reasons,
                "state": serialize_game_state(state),
            },
            indent=2,
        )
    )


def render_state_png(path: str | Path, state: GameState, agent_names: list[str] | None = None):
    out = Path(path)
    original_driver = os.environ.get("SDL_VIDEODRIVER")
    if original_driver is None:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    try:
        pygame.init()
        square = Dimension.SQUARE_SIZE.value
        armies = np.array(state.armies)
        ownership = np.array(state.ownership, dtype=bool)
        cities = np.array(state.cities, dtype=bool)
        mountains = np.array(state.mountains, dtype=bool)
        generals = np.array(state.generals, dtype=bool)
        neutral = np.array(state.ownership_neutral, dtype=bool)

        h, w = armies.shape
        width = w * square
        height = h * square
        screen = pygame.Surface((width, height))

        mountain_img = pygame.image.load(str(AssetPath.MOUNTAIN_PATH))
        general_img = pygame.image.load(str(AssetPath.GENERAL_PATH))
        city_img = pygame.image.load(str(AssetPath.CITY_PATH))
        font = pygame.font.Font(AssetPath.FONT_PATH, 18)

        for row in range(h):
            for col in range(w):
                rect = pygame.Rect(col * square, row * square, square, square)
                if mountains[row, col]:
                    color = MOUNTAIN
                elif ownership[0, row, col]:
                    color = PLAYER_COLORS[0]
                elif ownership[1, row, col]:
                    color = PLAYER_COLORS[1]
                elif neutral[row, col] and cities[row, col]:
                    color = CITY_NEUTRAL
                else:
                    color = EMPTY

                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)

                if mountains[row, col]:
                    screen.blit(mountain_img, mountain_img.get_rect(center=rect.center))
                elif cities[row, col]:
                    screen.blit(city_img, city_img.get_rect(center=rect.center))

                if generals[row, col]:
                    screen.blit(general_img, general_img.get_rect(center=rect.center))

                if armies[row, col] > 0:
                    text = font.render(str(int(armies[row, col])), True, WHITE)
                    screen.blit(text, text.get_rect(center=rect.center))

        pygame.image.save(screen, str(out))
    finally:
        pygame.quit()
        if original_driver is None:
            os.environ.pop("SDL_VIDEODRIVER", None)
