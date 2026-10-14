"""The cost registry: what one tick's objective is made of, and how it is bound.

A cost term used to be named three times -- computed, summed, then stored under its
trace key -- in a method that also had to know which layer owned it.  Here a term is
REGISTERED once, by the layer that owns it, and the solver only iterates:

    process start   CostTermEntry("CtrlEffort", weight, reward, make)   x1 per process
       __init__     inside MPPICore._cost_table
                            |
    each tick       BoundTerm("CtrlEffort", weight, fn, reward)         x~20 per second
     entrypoint     inside CostManager, inside MPPIProblem
                            |
    each batch      weight * fn(CostContext(states, controls, data))    x2-3 per tick
      evaluate()

The three lifetimes are separate on purpose.  Structure never changes, so validating it
per tick is waste and a structural bug (a duplicate name, a bad weight) must raise at
construction rather than on a control tick.  The bound objective differs per tick -- so it
is passed BY VALUE inside MPPIProblem and never stored on the planner, which is what keeps
one tick's inputs from leaking into the next.  And one bound objective is evaluated over
two different batches within
a tick: the (K,T,.) sample batch in solve(), then the settled batch-of-one for the trace.

Nothing here imports the planner, and no layer of the planner holds anything defined here
beyond the immutable CostTermTable built at __init__.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import math

import numpy as np
import torch

from statenav_mppi.mppi_core.mppi_validity import ValidityManager, ValidityVerdict, screen
from statenav_mppi.mppi_core.mppi_types import MPPIInput


# fn(ctx) -> (K,) UNWEIGHTED cost for this batch; the manager applies the weight.
CostFn = Callable[["CostContext"], torch.Tensor]
# (data, reference_points) -> the CostFn bound to this run.  reference_points is the
# (M, 2) local-reference tensor, or None for a run that has no reference.
CostMake = Callable[[MPPIInput, Optional[torch.Tensor]], CostFn]




####################################### PER-EVALUATION SCRATCH #######################################

@dataclass
class CostContext:
    """Per-evaluation scratch.  Lives for ONE CostManager.evaluate() call and no longer.

    Deliberately not a planner field: two batches share one bound objective within a tick
    (the sample batch and the settled batch-of-one), and each gets its own context, so a
    memoized intermediate can never leak from one batch into the other.
    """

    states: torch.Tensor       # (K,T,5) predicted states of this batch
    controls: torch.Tensor     # (K,T,2) projected commands of this batch
    data: MPPIInput            # this tick's inputs
    # Intermediates shared by several terms of one batch, memoized by key.  The path
    # terms share one (K,T) projection through it; recomputing it per term would run the
    # batched nearest-segment search four times over the same positions.
    cache: dict = field(default_factory=dict)




####################################### TERM STRUCTURE #######################################

@dataclass(frozen=True)
class CostTermEntry:
    """One term's structure.  Built once at planner construction, never per tick."""

    name: str          # trace name and terms-dict key
    weight: float      # resolved from the frozen MPPIConfig at __init__, never re-read
    reward: bool       # True -> the manager SUBTRACTS this term (Progress only, today)
    make: CostMake     # (data, reference_points) -> the bound fn for this tick


@dataclass(frozen=True)
class BoundTerm:
    """One term of one tick's objective: structure plus the fn bound to that tick."""

    name: str
    weight: float
    fn: CostFn
    reward: bool


class CostTermTable:
    """Immutable, ordered term structure.  Composed bottom layer up via _build_cost_table.

    Each planner layer overrides _build_cost_table(), calls super() FIRST and appends the
    terms it owns, so the composed order runs core -> terrain -> tracking -> task and a new
    term is one edit in the file that owns it.
    """

    def __init__(self) -> None:
        self._entries: list[CostTermEntry] = []
        self._names: set[str] = set()

    @property
    def entries(self) -> tuple[CostTermEntry, ...]:
        """Registered terms in evaluation order."""
        return tuple(self._entries)

    def add(
        self,
        name: str,
        weight: float,
        *,
        reward: bool = False,
        fn: Optional[CostFn] = None,
        make: Optional[CostMake] = None,
    ) -> None:
        """Register one term.  Exactly one of ``fn=`` (run-independent) or ``make=``.

        Raises on a duplicate name, a non-finite or negative weight, or on receiving both
        or neither of fn/make.  All three are construction errors, not tick errors: a
        control tick must never be the place a mis-registered objective is discovered.
        """
        if not isinstance(name, str) or not name:
            raise ValueError(f"cost term name must be a non-empty string, got {name!r}")
        if name in self._names:
            raise ValueError(
                f"cost term {name!r} is already registered; term names are the trace "
                f"keys and must be unique (registered so far: {sorted(self._names)})")
        value = float(weight)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"weight for cost term {name!r} must be finite and non-negative, "
                f"got {weight!r}")
        if (fn is None) == (make is None):
            raise ValueError(
                f"cost term {name!r} needs exactly one of fn= (run-independent) or "
                f"make= (bound per run), got "
                f"fn={'set' if fn is not None else 'None'}, "
                f"make={'set' if make is not None else 'None'}")
        # One always-populated `make`: a run-independent fn is simply a maker that
        # ignores the run, which keeps bind() a single uniform loop.
        bound_make: CostMake = (
            make if make is not None else (lambda data, reference, fn=fn: fn))
        self._entries.append(
            CostTermEntry(name=name, weight=value, reward=bool(reward),
                          make=bound_make))
        self._names.add(name)

    def bind(
        self, data: MPPIInput, reference_points: Optional[torch.Tensor],
    ) -> "CostManager":
        """Attach this tick's inputs.  Allocates N closures; validates nothing.

        ``reference_points`` is the ``(M, 2)`` local-reference tensor this run's path terms
        measure against, or None for a run with no reference.
        """
        return CostManager(tuple(
            BoundTerm(name=entry.name, weight=entry.weight,
                      fn=entry.make(data, reference_points), reward=entry.reward)
            for entry in self._entries))




####################################### ONE TICK'S OBJECTIVE #######################################

class CostManager:
    """One tick's objective: the weighted terms bound to that tick's inputs."""

    def __init__(self, terms: tuple[BoundTerm, ...]) -> None:
        self.terms = terms

    def evaluate(
        self, ctx: CostContext, terms: Optional[dict] = None,
    ) -> torch.Tensor:
        """Return the ``(K,)`` total cost: the weighted sum of the terms, and nothing else.

        No validity masking happens here any more.  That is ValidityManager's job,
        and screen() is what folds the two together -- see MPPIProblem.score().  This
        method means one thing again: sum the weighted terms.

        ``terms``, when given, is filled with each term's ``(K,)`` WEIGHTED contribution
        under its registered name.  An out-parameter rather than a second return value so
        the caller decides whether to pay for the dict at all.
        """
        total = torch.zeros(
            ctx.states.shape[0], device=ctx.states.device, dtype=ctx.states.dtype)
        for term in self.terms:
            value = term.weight * term.fn(ctx)      # (K,) unweighted -> weighted
            if term.reward:
                value = -value
            total = total + value
            if terms is not None:
                terms[term.name] = value
        return total

    def debug_lines(self) -> tuple[str, ...]:
        """Extra trace lines from terms that expose ``debug_line()``.

        A term whose LEVEL alone is unreadable (0.00 can mean "switched off" or "already
        achieved", with opposite tuning responses) publishes its own line here instead of
        leaving a snapshot on the planner for the trace to find.
        """
        lines = []
        for term in self.terms:
            emit = getattr(term.fn, "debug_line", None)
            if emit is None:
                continue
            line = emit()
            if line is not None:
                lines.append(line)
        return tuple(lines)

    def signature(self) -> str:
        """``(name, weight)`` summary, so an objective change is legible in the trace.

        Reward terms carry a leading ``-`` because that is how the manager applies them.
        """
        return " ".join(
            f"{'-' if term.reward else ''}{term.name}*{term.weight:g}"
            for term in self.terms)




####################################### ONE TICK'S PROBLEM STATEMENT #######################################

@dataclass(frozen=True)
class MPPIProblem:
    """One tick's problem statement.  solve() derives none of it.

    Preconditions solve() relies on and does NOT check:
      1. ``costs`` is fully bound (reference tensor, all resolved);
      2. any map snapshot the terms or the command constraints will read is already pinned.
    Both are the per-run entrypoint's job -- PathTrackingMPPI._prepare_problem for the
    task entrypoints.  Precondition 2 is guarded at use by
    TerrainMPPI._require_map_snapshot(), called from _term_Travmap.

    Passed by value: the objective is never a planner field, so one tick's bound inputs
    can never be read by the next.
    """

    data: MPPIInput
    costs: CostManager
    # This run's validity test, bound to the same inputs as `costs`.  A sibling of it, not
    # a part of it: a veto carries no weight, so a zero-weighted map cost still rejects a
    # cell that cannot bound a command.  Carried by value for the same reason `costs` is.
    validity: ValidityManager
    # (M, 2) world-frame local reference this run's path terms were bound to.  Path-flavoured
    # on an otherwise generic type, and carried only for the result's target endpoint and the
    # trace's reference section; solve() never projects against it.  None for a run that
    # tracks no reference.
    reference: Optional[np.ndarray] = None

    def score(
        self, ctx: CostContext, terms: Optional[dict] = None,
    ) -> tuple[torch.Tensor, "ValidityVerdict"]:
        """Price one batch and screen it: the ONE place costs and validity compose.

        Returns the ``(K,)`` total with every INVALID rollout replaced by ``+inf``, plus
        the verdict that says which validity term eliminated which rollout.

        A method rather than three copies at the three call sites (the sample batch, the
        settled batch-of-one behind the trace, and the offline _costs wrapper) because the
        composition carries an ordering requirement that is easy to get wrong and silent
        when it is: both managers MUST see the same CostContext instance, since the
        footprint lookup the terrain cost and the terrain veto share is memoized on
        ``ctx.cache`` and a second context would run that lookup twice per batch.

        ``terms`` is forwarded to CostManager.evaluate: the unscreened weighted components.
        """
        total = self.costs.evaluate(ctx, terms=terms)
        return screen(total, self.validity.evaluate(ctx))

    def __post_init__(self) -> None:
        """Every shape and finiteness check solve() used to run, in ONE place.

        Raises ValueError with the same reason strings the solver used to return, so the
        entrypoint that catches this still surfaces the message the operator knows.
        """
        # Structural, not numeric: an unbound objective or an absent validity set is a
        # wiring bug at the entrypoint, and a missing veto would silently let a rollout
        # over an unusable cell win the update.
        if not isinstance(self.costs, CostManager):
            raise ValueError(
                f"MPPIProblem.costs must be a bound CostManager, got "
                f"{type(self.costs).__name__}")
        if not isinstance(self.validity, ValidityManager):
            raise ValueError(
                f"MPPIProblem.validity must be a bound ValidityManager, got "
                f"{type(self.validity).__name__}")
        pose = np.asarray(self.data.pose, dtype=np.float32)
        velocity = np.asarray(self.data.velocity, dtype=np.float32)
        path = np.asarray(self.data.path, dtype=np.float32)
        if pose.shape != (3,) or velocity.shape != (2,):
            raise ValueError("invalid pose/velocity shape")
        if path.ndim != 2 or path.shape[1:] != (2,) or path.shape[0] < 2:
            raise ValueError("empty or invalid path")
        if (
            not np.all(np.isfinite(pose))
            or not np.all(np.isfinite(velocity))
            or not np.all(np.isfinite(path))
        ):
            raise ValueError("non-finite input state")
        scalars = (
            self.data.control_dt, self.data.cmd_v_limit, self.data.cmd_w_limit)
        if (
            not all(math.isfinite(float(value)) for value in scalars)
            or self.data.control_dt <= 0.0
        ):
            raise ValueError("invalid timing or command limits")
        if self.data.cmd_v_limit <= 0.0 or self.data.cmd_w_limit <= 0.0:
            raise ValueError("non-positive command limits")
