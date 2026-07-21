from .base import AbstractMonitor
from .deepseek import DeepSeekMonitor


_registry: dict[str, type[AbstractMonitor]] = {}


def register(name: str, cls: type[AbstractMonitor]):
    _registry[name] = cls


def get_monitor(name: str) -> AbstractMonitor:
    cls = _registry.get(name)
    if cls is None:
        choices = ", ".join(sorted(_registry))
        raise KeyError(f"unknown monitor '{name}'; available: {choices}")
    return cls()


def list_monitors() -> list[str]:
    return sorted(_registry)


register("deepseek", DeepSeekMonitor)
