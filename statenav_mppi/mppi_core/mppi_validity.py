"""The validity registry: which rolled-out samples the update is allowed to use.

A VALIDITY test judges a rollout AFTER the fact and produces a boolean mask -- "is this
sample usable at all".  An invalid sample must not merely be expensive, it must be
UNSELECTABLE, so its total cost is replaced with +inf rather than penalized.

That is a different mechanism from a CONSTRAINT, which shapes the command BEFORE the
rollout by projecting it onto an admissible set -- see mppi_constraints.py:

    constraint (there)     command -> admissible command      projection, per knot
    validity   (here)      rollout -> True / False            veto, per batch

Structurally this mirrors CostTermTable exactly, deliberately, because it is the same
three-lifetime shape:

    process start   ValidityTermEntry("MapUsable", make)          x1 per process
       __init__     inside MPPICore._validity_table
                            |
    each tick       BoundValidity("MapUsable", fn)                x~20 per second
     entrypoint     inside ValidityManager, inside MPPIProblem
                            |
    each batch      fn(CostContext(states, controls, data)) -> (K,) bool
      evaluate()

A veto used to be a side channel on the COST context (ctx.reject), which meant it lived
inside a cost term and a tick where every sample died was indistinguishable in the trace
from a tick where the map was merely tight.  Here every veto is named, so it can be
counted and attributed.

Nothing here imports the planner, and nothing here imports mppi_costs at runtime:
mppi_costs imports THIS module (MPPIProblem carries a ValidityManager), so the runtime
dependency runs one way only, validity <- costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from statenav_mppi.mppi_core.mppi_types import MPPIInput

if TYPE_CHECKING:
    # Type-only: a runtime import would close the cycle described in the module
    # docstring.  Every use below is a string annotation, which is never resolved.
    from statenav_mppi.mppi_core.mppi_costs import CostContext




####################################### ROLLOUT VALIDITY REGISTRY #######################################

# fn(ctx) -> (K,) or (K,T) bool for this batch; True = ADMISSIBLE.  A (K,T) mask is
# reduced with all(dim=-1): a rollout is valid only if every knot of it is.
ValidityFn = Callable[["CostContext"], torch.Tensor]
# (data, reference_points) -> the ValidityFn bound to this run.  Same signature as
# CostMake, so a validity test that needs the run's reference is registered the same way a
# cost that needs it is.
ValidityMake = Callable[[MPPIInput, Optional[torch.Tensor]], ValidityFn]

# Reserved trace key for the one validity test that cannot be a ctx-only predicate: a
# non-finite TOTAL COST.  It depends on the cost sum, so registering it would force the
# validity manager to run after the cost manager and to read the total off a shared
# scratch field -- an implicit ordering dependency between two managers.  It is folded in
# by screen() instead, and the name is reserved here so nothing else can claim the key.
FINITE_COST_TERM = "FiniteCost"




@dataclass(frozen=True)
class ValidityTermEntry:
    """One validity term's structure.  Built once at planner construction, never per tick."""

    name: str              # trace key and violation-report key
    make: ValidityMake   # (data, reference_points) -> the bound fn for this tick


@dataclass(frozen=True)
class BoundValidity:
    """One validity term of one tick: its name plus the fn bound to that tick."""

    name: str
    fn: ValidityFn


class ValidityTermTable:
    """Immutable, ordered validity structure.  Composed bottom layer up.

    Each planner layer overrides _build_validity_table(), calls super() FIRST and
    appends the vetoes it owns, exactly as _build_cost_table does for costs, so
    adding a validity term is one edit in the file that owns it.
    """

    def __init__(self) -> None:
        self._entries: list[ValidityTermEntry] = []
        self._names: set[str] = set()

    @property
    def entries(self) -> tuple[ValidityTermEntry, ...]:
        """Registered validity terms in evaluation order."""
        return tuple(self._entries)

    def add(
        self,
        name: str,
        *,
        fn: Optional[ValidityFn] = None,
        make: Optional[ValidityMake] = None,
    ) -> None:
        """Register one validity term.  Exactly one of ``fn=`` (run-independent) or ``make=``.

        No weight, deliberately: a veto is not priced, which is what makes it independent
        of any cost weight BY CONSTRUCTION rather than by the accident of how one function
        unpacked a tuple.

        Raises on a duplicate name, the reserved FiniteCost name, or on receiving both or
        neither of fn/make.  All are construction errors, not tick errors: a control tick
        must never be the place a mis-registered validity term is discovered.
        """
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"validity term name must be a non-empty string, got {name!r}")
        if name == FINITE_COST_TERM:
            raise ValueError(
                f"validity term name {FINITE_COST_TERM!r} is reserved for the total-cost "
                f"finiteness test that screen() folds in; it cannot be registered because "
                f"it depends on the cost total rather than on the context alone")
        if name in self._names:
            raise ValueError(
                f"validity term {name!r} is already registered; term names are the "
                f"trace and violation-report keys and must be unique (registered so far: "
                f"{sorted(self._names)})")
        if (fn is None) == (make is None):
            raise ValueError(
                f"validity term {name!r} needs exactly one of fn= (run-independent) or "
                f"make= (bound per run), got "
                f"fn={'set' if fn is not None else 'None'}, "
                f"make={'set' if make is not None else 'None'}")
        # One always-populated `make`: a run-independent fn is simply a maker that
        # ignores the run, which keeps bind() a single uniform loop.
        bound_make: ValidityMake = (
            make if make is not None else (lambda data, reference, fn=fn: fn))
        self._entries.append(ValidityTermEntry(name=name, make=bound_make))
        self._names.add(name)

    def bind(
        self, data: MPPIInput, reference_points: Optional[torch.Tensor],
    ) -> "ValidityManager":
        """Attach this tick's inputs.  Allocates N closures; validates nothing."""
        return ValidityManager(tuple(
            BoundValidity(name=entry.name, fn=entry.make(data, reference_points))
            for entry in self._entries))




@dataclass(frozen=True)
class ValidityVerdict:
    """Which rollouts of ONE batch are valid, and which term said otherwise."""

    valid: torch.Tensor      # (K,) bool, True = usable by the update
    per_term: dict                # {name: (K,) bool}, the same convention, for attribution

    def with_term(self, name: str, mask: torch.Tensor) -> "ValidityVerdict":
        """Return a new verdict with one more named term ANDed in.

        The mechanism screen() uses to fold in FiniteCost without registering it.  Frozen
        in, frozen out: the original verdict is never mutated, so a caller holding one
        cannot have it change underneath.
        """
        if name in self.per_term:
            raise ValueError(
                f"validity term {name!r} is already in this verdict; a term may be folded "
                f"in only once (present: {sorted(self.per_term)})")
        return ValidityVerdict(
            valid=self.valid & mask,
            per_term={**self.per_term, name: mask})

    def rejection_counts(self) -> dict:
        """``{name: rollouts this validity term eliminated}``, only where non-zero.

        Per-term and therefore NOT a partition: one rollout failing three terms is
        counted by all three, so the values need not sum to the rejected total.  That is
        the useful reading -- "how tight is each test", not "who owns each corpse".
        """
        counts = {}
        for name, mask in self.per_term.items():
            rejected = int(torch.count_nonzero(~mask))
            if rejected:
                counts[name] = rejected
        return counts

    def dominant_reason(self) -> Optional[str]:
        """The registered validity term that eliminated the most rollouts; None if all passed.

        Ties break on the name, so the reported reason is reproducible across runs
        rather than following dict order.
        """
        counts = self.rejection_counts()
        if not counts:
            return None
        return max(counts, key=lambda name: (counts[name], name))




class ValidityManager:
    """One tick's validity test: the vetoes bound to that tick's inputs."""

    def __init__(self, terms: tuple[BoundValidity, ...]) -> None:
        self.terms = terms

    def evaluate(self, ctx: "CostContext") -> ValidityVerdict:
        """Judge one batch.  Returns the ``(K,)`` admissibility plus per-term attribution.

        Must be handed the SAME CostContext instance the cost manager was given: the
        footprint lookup TerrainMPPI's map cost and map veto share is memoized on
        ``ctx.cache``, and a second context would run that lookup twice per batch.
        """
        valid = torch.ones(
            ctx.states.shape[0], device=ctx.states.device, dtype=torch.bool)
        per_term: dict = {}
        for term in self.terms:
            mask = term.fn(ctx)
            # (K,T) -> (K,): a rollout is valid only where every knot of it is.
            mask = mask if mask.ndim == 1 else torch.all(mask, dim=-1)
            per_term[term.name] = mask
            valid = valid & mask
        return ValidityVerdict(valid=valid, per_term=per_term)

    def debug_lines(self) -> tuple[str, ...]:
        """Extra trace lines from vetoes that expose ``debug_line()``.

        The mirror of CostManager.debug_lines(), and for the same reason: a veto whose
        rejection COUNT alone is unreadable publishes its own line.  "Corridor 1024" can
        mean "the plan is wandering" or "the bound is mis-sized and the term abstained",
        and the tuning response to the two is opposite.  Terms without the attribute -- all
        of the structural ones -- contribute nothing and pay nothing.
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
        """``"FiniteState & FiniteControl & MapUsable"`` -- the validity set, for the trace."""
        return " & ".join(term.name for term in self.terms)




def screen(
    total: torch.Tensor, verdict: ValidityVerdict,
) -> tuple[torch.Tensor, ValidityVerdict]:
    """Replace every invalid rollout's total with ``+inf``, folding in FiniteCost.

    ``total`` is the ``(K,)`` weighted cost sum and ``verdict`` the validity manager's
    judgement of the SAME batch.  Returns the screened costs and the verdict extended with
    the reserved FiniteCost term, so the caller can report what actually eliminated what.

    An invalid rollout must not merely be expensive, it must be UNSELECTABLE, so its
    total is replaced rather than penalized.
    """
    verdict = verdict.with_term(FINITE_COST_TERM, torch.isfinite(total))
    return (
        torch.where(verdict.valid, total, torch.full_like(total, float("inf"))),
        verdict,
    )
