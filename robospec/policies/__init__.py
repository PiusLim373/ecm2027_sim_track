"""Policy registry: short name -> "module:Class"."""
import importlib

REGISTRY = {
    "scripted_smoke": "robospec.policies.scripted_smoke:ScriptedSmoke",
}


def load(name: str):
    """Accept a registry name or a full "module:Class" path."""
    module, cls = REGISTRY.get(name, name).split(":")
    return getattr(importlib.import_module(module), cls)
