__version__ = "0.0.1"

import importlib

# Import subpackages
# NOTE: the FSM moved out to the standalone `statenav_fsm` package so that repos without
# statenav_global (IsaacLab, elevation_mapping) can use it; import it directly:
#     from statenav_fsm.NavigationFSM import NavigationFSM, FSMState
# Subpackages are imported LAZILY, on first attribute access (PEP 562 module __getattr__), so
# importing one light module (e.g. statenav_global.core.graph_layout) never loads the torch /
# ROS stack. Search order: world_model first, so `from statenav_global import
# CMDbasedMap` does not import planners.
_SUBPACKAGES = ("world_model", "utility", "inference", "planners")


def __getattr__(name):
    # Subpackage access (statenav_global.world_model, ...) imports only that subpackage.
    if name in _SUBPACKAGES:
        return importlib.import_module(f"{__name__}.{name}")
    # Automatically combine __all__ from subpackages
    # This way, changes in subpackages automatically propagate here
    if name == "__all__":
        names = list(_SUBPACKAGES) + [n for sub in _SUBPACKAGES for n in __getattr__(sub).__all__]
        globals()["__all__"] = names
        return names
    # Other dunder probes (__wrapped__, __file__-style introspection) must not import anything.
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Import all classes from subpackages (respects their __all__ definitions)
    for sub in _SUBPACKAGES:
        module = __getattr__(sub)
        if name in module.__all__:
            value = getattr(module, name)
            globals()[name] = value   # cache, so later lookups skip __getattr__
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
