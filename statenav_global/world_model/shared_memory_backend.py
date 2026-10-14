"""Back-compat alias: the shared-memory map backend moved to statenav_global.core.shared_memory_backend.

Replaces this module in sys.modules with the core one, so both import paths yield the
SAME module object (same classes, same private helpers, same atexit/signal registration).
New code should import statenav_global.core.shared_memory_backend directly: importing through
this path still runs statenav_global.world_model.__init__ and its torch / ROS stack.
"""
import sys

from statenav_global.core import shared_memory_backend as _core_shared_memory_backend

sys.modules[__name__] = _core_shared_memory_backend
