"""
Template custom agent for LAN play.

Usage:
    uv run python examples/lan_client.py \
        --host <server-ip> \
        --agent-custom examples/custom_agent.py:make_agent \
        --name MyBot
"""

import jax.numpy as jnp
import jax.random as jrandom

from generals.agents import Agent
from generals.core.action import compute_valid_move_mask
from generals.core.observation import Observation


class FirstMoveAgent(Agent):
    """
    Simple deterministic baseline for hacking on your own strategy.

    Picks the first legal move in scan order and never splits.
    """

    def __init__(self, id: str = "FirstMove"):
        super().__init__(id)

    def act(self, observation: Observation, key: jnp.ndarray) -> jnp.ndarray:
        del key

        valid_mask = compute_valid_move_mask(
            observation.armies,
            observation.owned_cells,
            observation.mountains,
        )
        h, w, _ = valid_mask.shape
        valid_positions = jnp.argwhere(valid_mask, size=h * w * 4, fill_value=-1)
        num_valid = jnp.sum(jnp.all(valid_positions >= 0, axis=-1))

        should_pass = num_valid == 0
        move = valid_positions[0]
        return jnp.array(
            [should_pass.astype(jnp.int32), move[0], move[1], move[2], jnp.int32(0)],
            dtype=jnp.int32,
        )


class NoisyFirstMoveAgent(Agent):
    """Example stochastic agent that flips between full and half moves."""

    def __init__(self, id: str = "NoisyFirstMove"):
        super().__init__(id)

    def act(self, observation: Observation, key: jnp.ndarray) -> jnp.ndarray:
        valid_mask = compute_valid_move_mask(
            observation.armies,
            observation.owned_cells,
            observation.mountains,
        )
        h, w, _ = valid_mask.shape
        valid_positions = jnp.argwhere(valid_mask, size=h * w * 4, fill_value=-1)
        num_valid = jnp.sum(jnp.all(valid_positions >= 0, axis=-1))

        should_pass = num_valid == 0
        move = valid_positions[0]
        split = jrandom.bernoulli(key, 0.5).astype(jnp.int32)
        return jnp.array(
            [should_pass.astype(jnp.int32), move[0], move[1], move[2], split],
            dtype=jnp.int32,
        )


def make_agent(name: str) -> Agent:
    return FirstMoveAgent(id=name)
