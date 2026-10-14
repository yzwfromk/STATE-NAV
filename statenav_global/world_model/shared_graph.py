"""Back-compat alias: the shared-memory graph moved to statenav_global.core.shared_graph.

Replaces this module in sys.modules with the core one, so both import paths yield the
SAME module object (same classes, same private helpers, same atexit/signal registration).
New code should import statenav_global.core.shared_graph directly: importing through this
path still runs statenav_global.world_model.__init__ and its torch / ROS stack.
"""
import sys

from statenav_global.core import shared_graph as _core_shared_graph

sys.modules[__name__] = _core_shared_graph
