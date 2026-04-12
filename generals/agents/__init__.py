# JAX-compatible agents
from .agent import Agent
from .random_agent import RandomAgent
from .expander_agent import ExpanderAgent
from .graph_search_agent import GraphSearchAgent
from .loading import build_agent, load_agent_factory
from .strategic_agent import (
    BackdoorAgent,
    ChaosAgent,
    DefenseCounterAgent,
    GreedyCityAgent,
    MaterialAdvantageAgent,
    PunishAgent,
    ScoutPressureAgent,
    SniperAgent,
    SurroundPressureAgent,
    SwarmAgent,
    TurtleAgent,
)

# CLI name → (agent class, default display name)
BUILTIN_AGENTS: dict[str, tuple[type[Agent], str]] = {
    "expander": (ExpanderAgent, "Expander"),
    "random": (RandomAgent, "Random"),
    "graph": (GraphSearchAgent, "GraphSearch"),
    "material": (MaterialAdvantageAgent, "MaterialAdvantage"),
    "scout": (ScoutPressureAgent, "ScoutPressure"),
    "backdoor": (BackdoorAgent, "Backdoor"),
    "defense": (DefenseCounterAgent, "DefenseCounter"),
    "surround": (SurroundPressureAgent, "SurroundPressure"),
    "turtle": (TurtleAgent, "Turtle"),
    "punish": (PunishAgent, "Punish"),
    "swarm": (SwarmAgent, "Swarm"),
    "sniper": (SniperAgent, "Sniper"),
    "greedy_city": (GreedyCityAgent, "GreedyCity"),
    "chaos": (ChaosAgent, "Chaos"),
}


def build_builtin_agent(name: str, display_name: str | None = None) -> Agent:
    """Instantiate a built-in agent by CLI name."""
    if name not in BUILTIN_AGENTS:
        raise ValueError(f"Unknown built-in agent: {name!r} (available: {', '.join(BUILTIN_AGENTS)})")
    cls, default_name = BUILTIN_AGENTS[name]
    return cls(id=display_name or default_name)


__all__ = [
    "Agent",
    "RandomAgent",
    "ExpanderAgent",
    "GraphSearchAgent",
    "MaterialAdvantageAgent",
    "ScoutPressureAgent",
    "BackdoorAgent",
    "DefenseCounterAgent",
    "SurroundPressureAgent",
    "TurtleAgent",
    "PunishAgent",
    "SwarmAgent",
    "SniperAgent",
    "GreedyCityAgent",
    "ChaosAgent",
    "BUILTIN_AGENTS",
    "build_builtin_agent",
    "build_agent",
    "load_agent_factory",
]
