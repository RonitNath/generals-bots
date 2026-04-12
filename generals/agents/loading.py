from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

from generals.agents.agent import Agent


def _load_module_from_path(path: Path) -> ModuleType:
    spec = spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from path: {path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_agent_factory(target: str):
    """
    Load an agent factory from either:
    - `python.module:factory_name`
    - `/abs/or/rel/path/to/file.py:factory_name`

    The referenced attribute may be:
    - an Agent subclass, which will be instantiated with `id=...`
    - a callable returning an Agent instance
    """
    if ":" not in target:
        raise ValueError(
            "Agent target must be in the form 'module:factory' or '/path/to/file.py:factory'."
        )

    module_ref, attr_name = target.split(":", 1)
    if not module_ref or not attr_name:
        raise ValueError(f"Invalid agent target: {target}")

    module_path = Path(module_ref)
    if module_path.suffix == ".py" or module_path.exists():
        module = _load_module_from_path(module_path.resolve())
    else:
        module = import_module(module_ref)

    try:
        factory = getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(f"Module '{module_ref}' does not define '{attr_name}'.") from exc

    return factory


def build_agent(factory_target: str, name: str | None = None) -> Agent:
    factory = load_agent_factory(factory_target)

    if isinstance(factory, type) and issubclass(factory, Agent):
        return factory(id=name or factory.__name__)

    agent = factory(name or "CustomAgent")
    if not isinstance(agent, Agent):
        raise TypeError(f"Factory '{factory_target}' did not return an Agent instance.")
    return agent
