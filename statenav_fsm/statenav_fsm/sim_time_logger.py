"""Sim-time console logging for every statenav / gsa ROS node.

Every ROS 2 log line is prefixed by rcutils, e.g.

    [INFO] [1786660114.575760334] [vlm_processor_node]: VLM evaluation signal received...

That timestamp comes from the `{time}` token of RCUTILS_CONSOLE_OUTPUT_FORMAT, which
rcutils fills with rcutils_system_time_now() — wall clock, always.  Logging lives BELOW
rclpy, in a layer that has no access to any ROS clock, so `use_sim_time` does not and
cannot change it: there is no parameter, no environment variable and no output-format
token that makes the prefix sim time.  When the pipeline is driven by a simulator whose
/clock is paused, throttled or replayed, those wall-clock stamps cannot be lined up with
the simulated run at all.

The only place sim time can go is the MESSAGE, which is what this module does: it wraps
the node logger in a proxy that prepends the node's ROS clock reading to every message.

    [INFO] [1786660114.575760334] [vlm_processor_node]: [sim   1523.480] VLM evaluation...
           ^ system time (rcutils, unavoidable)          ^ sim time (this module)

Usage — list the mixin FIRST so its get_logger() wins the MRO over Node's:

    from statenav_fsm.sim_time_logger import SimTimeLoggerMixin, turn_banner

    class MyNode(SimTimeLoggerMixin, Node):
        def __init__(self):
            super().__init__('my_node', parameter_overrides=[
                Parameter('use_sim_time', Parameter.Type.BOOL, True)])

No call site changes are needed: every existing self.get_logger().info(...) is stamped
from then on.  The mixin defines no __init__, so the cooperative super().__init__(...)
chain above still reaches Node.__init__ untouched.

NOTE: importing this module imports NO rclpy and no ROS message packages (only `os`), so
unlike qos_profiles it is safe to import at module scope even from modules that must stay
importable without rclpy.  rclpy is touched lazily, on first proxy construction, and only
to register this file as an "internal caller" (see _register_as_internal_caller).

CAVEAT: sim time only advances while /clock is published.  With use_sim_time=True and no
/clock publisher, every line reads `[sim      0.000]` forever — that is the intended tell
that sim time is not flowing, not a bug in this module.  Check with `ros2 topic hz /clock`.
"""

import os

# Width of the sim-time field, e.g. "[sim   1523.480]".  Fixed so the message text of
# consecutive lines stays column-aligned in the terminal.
_SIM_FIELD_WIDTH = 10

# Set once the first proxy has registered this file with rclpy's caller detection.
_caller_registered = False


def _register_as_internal_caller():
    """Hide this module from rclpy's log-caller detection.

    rclpy's RcutilsLogger.log() walks up the stack for the first frame outside rclpy and
    uses it for (a) the {function_name}/{file_path}/{line_number} console tokens and
    (b) the per-call-site key of the `once` / `throttle` / `skip_first` filter contexts.
    A proxy sits between the real call site and rclpy, so without this every stamped line
    would be attributed to sim_time_logger.py and — far worse — all `once=`/`throttle=`
    calls in the process would share ONE filter context and suppress each other.  rclpy
    keeps `_internal_callers` exactly so wrappers can opt out, and its own lazy populate
    step extends that list rather than replacing it, so this entry survives.

    NEVER raises: logging must not be able to break a node.  A failure here only degrades
    caller attribution of filtered log calls, so it is reported once and then ignored.
    """
    global _caller_registered
    if _caller_registered:
        return
    _caller_registered = True  # set first: one attempt, whatever the outcome
    try:
        from rclpy.impl import rcutils_logger
        rcutils_logger._internal_callers.append(os.path.realpath(__file__))
    except Exception as e:  # rclpy internals renamed / not importable
        import warnings
        warnings.warn(
            f'statenav_fsm.sim_time_logger: could not register with rclpy caller '
            f'detection ({type(e).__name__}: {e}). Sim-time stamping still works; only '
            f'the file/line attribution and once=/throttle_duration_sec= filter contexts '
            f'of stamped log calls are affected.'
        )


class SimTimeLogger:
    """Node-logger proxy that prefixes the node's ROS (sim) clock to every message.

    Wraps rather than subclasses rclpy's RcutilsLogger: the wrapped object is whatever
    Node.get_logger() returned, and anything not overridden here is delegated untouched
    (see __getattr__), so `.name`, `.set_level()`, `.get_child()` etc. keep working.
    Messages logged through a child obtained via .get_child() are NOT stamped — that
    child is a plain rclpy logger.
    """

    __slots__ = ('_inner', '_clock_fn')

    def __init__(self, inner, clock_fn):
        """
        Args:
            inner: the real logger to forward to (rclpy RcutilsLogger).
            clock_fn: zero-arg callable returning the node's rclpy Clock (normally the
                bound `node.get_clock`).  A callable rather than a Clock so the clock is
                resolved per call: rclpy's TimeSource attaches/replaces the node clock
                while use_sim_time is applied, and a reference captured too early could
                be the wrong (system-time) one.
        """
        self._inner = inner
        self._clock_fn = clock_fn
        _register_as_internal_caller()

    def _stamp(self, message):
        """Return `message` prefixed with the current sim time, e.g. '[sim   1523.480] ...'.

        NEVER raises: a failing clock read (clock not attached yet, node already
        destroyed) must not take the log line down with it, so the field degrades to the
        exception's class name instead of propagating.
        """
        try:
            # .nanoseconds is an int since the clock's epoch.  Sim clocks start at 0, so
            # this is seconds-since-sim-start under Isaac/Gazebo, not a Unix timestamp.
            field = f'{self._clock_fn().now().nanoseconds * 1e-9:{_SIM_FIELD_WIDTH}.3f}'
        except Exception as e:
            field = f'{type(e).__name__[:_SIM_FIELD_WIDTH]:>{_SIM_FIELD_WIDTH}}'
        return f'[sim {field}] {message}'

    # ── Severity methods: stamp, then forward verbatim (kwargs carry the rclpy log
    #    filters throttle_duration_sec / skip_first / once, which must pass through). ──
    def debug(self, message, **kwargs):
        return self._inner.debug(self._stamp(message), **kwargs)

    def info(self, message, **kwargs):
        return self._inner.info(self._stamp(message), **kwargs)

    def warn(self, message, **kwargs):
        return self._inner.warn(self._stamp(message), **kwargs)

    def warning(self, message, **kwargs):
        return self._inner.warning(self._stamp(message), **kwargs)

    def error(self, message, **kwargs):
        return self._inner.error(self._stamp(message), **kwargs)

    def fatal(self, message, **kwargs):
        return self._inner.fatal(self._stamp(message), **kwargs)

    def log(self, message, severity, **kwargs):
        return self._inner.log(self._stamp(message), severity, **kwargs)

    def __getattr__(self, name):
        """Delegate everything not overridden above to the real logger.

        __slots__ is defined, so this only fires on genuine misses.  The explicit slot
        guard stops an infinite recursion if an attribute is read before __init__ has
        assigned _inner (e.g. by copy/pickle probing for __deepcopy__ / __getstate__).
        """
        if name in SimTimeLogger.__slots__:
            raise AttributeError(name)
        return getattr(self._inner, name)


class SimTimeLoggerMixin:
    """Mixin for rclpy Node subclasses: makes get_logger() return a SimTimeLogger.

    List it FIRST in the base list, before Node, or Node.get_logger() wins the MRO and
    nothing is stamped.  It deliberately defines no __init__, so putting it first does
    not disturb the node's cooperative super().__init__(...) chain — super() passes
    straight through to Node.__init__.
    """

    # Class attribute rather than an __init__ assignment (there is no __init__): rclpy
    # calls get_logger() from inside Node.__init__, i.e. before any subclass __init__
    # body runs, so this name must already resolve at that point.
    _sim_logger = None

    def get_logger(self):
        if self._sim_logger is None:
            try:
                # `self.get_clock` (bound, uncalled) — see SimTimeLogger.__init__.
                self._sim_logger = SimTimeLogger(super().get_logger(), self.get_clock)
            except Exception:
                # Called too early (node internals not built yet): use the plain logger
                # for this one line and retry the wrap on the next call.
                return super().get_logger()
        return self._sim_logger


def turn_banner(title='', width=78, fill='='):
    """Return a rule line that separates one processing turn from the previous one.

    Logged at the start of a turn so a multi-second block of turn output is visually
    delimited in a console shared by several nodes.

    Args:
        title: text centred in the rule; omit for a plain rule.
        width: total character width of the rule (a title longer than `width` is not
            truncated — the rule grows instead, so the title is never mangled).
        fill: character the rule is drawn with.

    Returns:
        e.g. turn_banner('GSAM turn 7') ->
        '================================ GSAM turn 7 ================================='
    """
    if not title:
        return fill * width
    label = f' {title} '
    # At least two fill characters per side, even when the label overruns `width`.
    pad = max(width - len(label), 4)
    left = pad // 2
    return f'{fill * left}{label}{fill * (pad - left)}'
