"""Class-level declarations of a System (ADR 0001, decision 2).

A System is written as flat declarations in its class body, with no
``__init__``::

    class ValveSystem(System):
        valvePosition = Input()
        valvePosition = Output()
        waterFlowRate = Output()
        waterFlowRateMax = Parameter(0.016, lb=5e-3, ub=1.0, scaling="log")

        def forward(self, x, inputs, p, step_size):
            ...

The metaclass's ``__prepare__`` hands the class body a namespace that files
each declaration into a registry as it is written, instead of storing it as a
class attribute.  So one name may serve one input and one output, nothing is
renamed, and ``nn.Module`` can register the runtime parameter under the same
name on each instance.  Any other repeated name raises ``TypeError``.

The registries of a class, merged along its bases (a subclass may redeclare),
are ``cls._declarations``: ``{"input": {...}, "output": {...}, "state": {...},
"parameter": {...}, "setting": {...}, "child": {...}}``, each in declaration
order.  :class:`~twin4build.systems.saref4syst.system.System` builds the
runtime ports, parameters, states and children from them.
"""

from __future__ import annotations

# Standard library imports
import functools
import itertools
from typing import Any, Dict, Optional, Sequence

#: The kinds of declaration, in the order the base builds them.
KINDS = ("setting", "input", "output", "parameter", "state", "child")

_ORDER = itertools.count()


class _Required:
    """The default of a setting the constructor must be given."""

    def __repr__(self) -> str:
        return "REQUIRED"


#: The default of a :class:`Setting` that has none: constructing the System
#: without it raises ``TypeError``.
REQUIRED = _Required()


class Declaration:
    """One class-level declaration of a System. Its name is the attribute name
    it is written under in the class body."""

    kind = ""

    def __init__(self, doc: str = ""):
        self.name: Optional[str] = None
        self.doc = doc
        self._order = next(_ORDER)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


def _check_size(size, where: str) -> None:
    if size is None or size == "wired" or isinstance(size, (Input, Output, Setting)):
        return
    if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
        return
    raise TypeError(
        f"{where}: size= takes an int, 'wired', or another port or setting of the class, got {size!r}"
    )


def _check_init(init, where: str, allow_state: bool) -> None:
    if init is None or isinstance(init, Setting) or (allow_state and isinstance(init, State)):
        return
    if isinstance(init, (int, float)) and not isinstance(init, bool):
        return
    kinds = "a number, a setting or a state" if allow_state else "a number or a setting"
    raise TypeError(f"{where}: init= takes {kinds} of the class, got {init!r}")


def _check_per_slot(per_slot, per_slot_if, where: str) -> None:
    if per_slot is not None and not isinstance(per_slot, Input):
        raise TypeError(f"{where}: per_slot= takes an input of the class, got {per_slot!r}")
    if per_slot_if is not None:
        if per_slot is None:
            raise TypeError(f"{where}: per_slot_if= needs per_slot=")
        if not isinstance(per_slot_if, Setting):
            raise TypeError(f"{where}: per_slot_if= takes a setting of the class, got {per_slot_if!r}")


class Input(Declaration):
    """An input port.

    Args:
        size: ``None`` for a scalar port. For a vector port: an int, ``"wired"``
            (as many slots as its connections fill), or another port or setting
            of the class (the same size; a setting holding a sequence gives its
            length).
        default: What an unwired input holds: a number, or another input of
            the class, whose value it then reads (a make-up air stream that is
            outdoor air unless something says otherwise:
            ``makeUpAirTemperature = Input(default=outdoorTemperature)``).
            ``None``: the input must be wired.
        alias: An old name of the port that still connects, with a
            ``DeprecationWarning``.
        doc: What the port carries, with its unit.
    """

    kind = "input"

    def __init__(self, size=None, default=None, alias: Optional[str] = None, doc: str = ""):
        super().__init__(doc)
        _check_size(size, "Input")
        if default is not None and not isinstance(default, Input) and (
            isinstance(default, bool) or not isinstance(default, (int, float))
        ):
            raise TypeError(f"Input: default= takes a number or another input of the class, got {default!r}")
        self.size = size
        self.default = default
        self.alias = alias


class Output(Declaration):
    """An output port.

    Args:
        size: As for :class:`Input`.
        init: The value the output holds before the first step, which a
            reader one step behind sees: a number, a setting of the class, or a
            state of the class (an output that publishes the state starts at
            the state's initial value).  ``None``: 0.
        alias: An old name of the port that still connects and reads, with a
            ``DeprecationWarning``.
        doc: What the port carries, with its unit.
    """

    kind = "output"

    def __init__(self, size=None, init=None, alias: Optional[str] = None, doc: str = ""):
        super().__init__(doc)
        _check_size(size, "Output")
        _check_init(init, "Output", allow_state=True)
        self.size = size
        self.init = init
        self.alias = alias


class State(Declaration):
    """A continuous state the System carries from one step to the next.

    Args:
        size: The number of state variables: an int, or another port or setting
            of the class.
        init: The value it starts from when the model sets no initial state: a
            number, or a setting of the class (a number, or one per state
            variable).
        lb, ub: Physical bounds, used to box the state in collocation.
        names: One name per state variable, for diagnostics.
        doc: What the state is, with its unit.
    """

    kind = "state"

    def __init__(self, size=1, init: float = 0.0, lb=None, ub=None,
                 names: Optional[Sequence[str]] = None, doc: str = ""):
        super().__init__(doc)
        if size == "wired":
            raise TypeError("State: size= takes an int or another port or setting of the class, not 'wired'")
        _check_size(size, "State")
        _check_init(init, "State", allow_state=False)
        self.size = size
        self.init = init
        self.lb = lb
        self.ub = ub
        self.names = None if names is None else list(names)


class Parameter(Declaration):
    """A value of the System that can be estimated.

    Args:
        value: The default value, or a function of the System that computes it
            once the System has its settings and its other parameters (a
            radiator's UA from its nominal power).  ``None``: the parameter is
            absent until it is given (``System(c=...)`` or ``system.c = ...``).
            An absent
            parameter is not in ``forward``'s ``p``, is not estimated and is not
            saved; the System says what follows from its absence in ``derive``.
        lb, ub: The range the estimator searches. A parameter without both is
            not estimated unless the caller gives bounds.
        scaling: ``"linear"`` or ``"log"`` (equal steps are equal ratios; needs
            ``lb > 0``).
        relative: ``(low, high)`` factors: once something sizes the parameter
            (:meth:`System.size_parameter`), its range is ``value * low`` to
            ``value * high``.
        per_slot: An input of the class: the parameter holds one value per
            slot of that input (per copy), as :class:`Child`'s ``per_slot``.
        per_slot_if: A boolean setting of the class: ``per_slot`` holds only
            while the setting is true; otherwise one value per copy.
        doc: What the parameter is, with its unit.
    """

    kind = "parameter"

    def __init__(self, value, lb=None, ub=None, scaling: str = "linear",
                 relative: Optional[Sequence[float]] = None, per_slot=None, per_slot_if=None,
                 doc: str = ""):
        super().__init__(doc)
        if scaling not in ("linear", "log"):
            raise TypeError(f"Parameter: scaling= is 'linear' or 'log', got {scaling!r}")
        _check_per_slot(per_slot, per_slot_if, "Parameter")
        self.value = value
        self.lb = lb
        self.ub = ub
        self.scaling = scaling
        self.relative = None if relative is None else tuple(relative)
        self.per_slot = per_slot
        self.per_slot_if = per_slot_if

    def bounds(self) -> Optional[Dict[str, Any]]:
        """The entry of the System's bounds table, or ``None`` without bounds."""
        if self.lb is None and self.ub is None and self.relative is None:
            return None
        spec: Dict[str, Any] = {}
        if self.lb is not None:
            spec["lb"] = self.lb
        if self.ub is not None:
            spec["ub"] = self.ub
        if self.relative is not None:
            spec["relative"] = self.relative
        return spec


class Setting(Declaration):
    """A constructor value that is never estimated: a size, a modelling choice,
    a constant.

    Args:
        default: The value when the constructor is not given one.
            :data:`REQUIRED` (the default): the constructor must be given it.
        convert: Applied to the value the constructor is given (a reader from
            the dict a saved model holds, ...).
        save: Whether a saved model holds the value (``False`` for a value
            that cannot be written down, such as a function).
        doc: What the setting chooses.
    """

    kind = "setting"

    def __init__(self, default=REQUIRED, convert=None, save: bool = True, doc: str = ""):
        super().__init__(doc)
        self.default = default
        self.convert = convert
        self.save = save


class Child(Declaration):
    """A child System of a composite: built with the composite, stepped by the
    composite's ``forward``.

    Args:
        cls: The child's class.
        per_slot: An input of the composite: the child is stepped once per slot
            of that input (``n_c`` times its width copies; an air handling
            unit's damper, one per branch), so each of its parameters holds a
            value per slot.
        **values: The child's constructor values. A user overrides them with
            ``name=dict(...)`` on the composite's constructor.
    """

    kind = "child"

    def __init__(self, cls, per_slot=None, **values):
        super().__init__(values.pop("doc", ""))
        _check_per_slot(per_slot, None, "Child")
        self.cls = cls
        self.per_slot = per_slot
        self.values = values


class _Namespace(dict):
    """The class body's namespace: declarations go into registries."""

    def __init__(self, qualname: str):
        super().__init__()
        self.qualname = qualname
        self.declarations: Dict[str, Dict[str, Declaration]] = {kind: {} for kind in KINDS}

    def _kind_of(self, name: str) -> Optional[str]:
        for kind, book in self.declarations.items():
            if name in book:
                return kind
        return None

    def __setitem__(self, name, value):
        if isinstance(value, Declaration):
            book = self.declarations[value.kind]
            if name in book:
                raise TypeError(f"{self.qualname}: {type(value).__name__} {name!r} is declared twice")
            for kind, other in self.declarations.items():
                if kind != value.kind and name in other and {kind, value.kind} != {"input", "output"}:
                    raise TypeError(
                        f"{self.qualname}: {name!r} is already declared as a {kind}; only one input "
                        f"and one output may share a name"
                    )
            if dict.__contains__(self, name) and not name.startswith("__"):
                raise TypeError(f"{self.qualname}: {name!r} is already a class attribute")
            value.name = name
            book[name] = value
            return
        kind = self._kind_of(name) if isinstance(name, str) else None
        if kind is not None and not name.startswith("__"):
            raise TypeError(f"{self.qualname}: {name!r} is already declared as a {kind}")
        super().__setitem__(name, value)

    def __missing__(self, name):
        # Lets a declaration refer to an earlier one (``size=otherPort``). A
        # name that is both an input and an output refers to the input.
        for kind in ("input", "output", "setting", "parameter", "state", "child"):
            book = self.declarations[kind]
            if name in book:
                return book[name]
        raise KeyError(name)


def _merge(cls) -> Dict[str, Dict[str, Declaration]]:
    merged: Dict[str, Dict[str, Declaration]] = {kind: {} for kind in KINDS}
    for klass in reversed(cls.__mro__):
        own = klass.__dict__.get("_own_declarations")
        if not own:
            continue
        for kind, book in own.items():
            for name, decl in book.items():
                for other_kind in KINDS:
                    if other_kind != kind and {kind, other_kind} != {"input", "output"}:
                        merged[other_kind].pop(name, None)
                merged[kind][name] = decl
    return merged


def _check_references(cls, merged) -> None:
    own = {
        id(decl)
        for klass in cls.__mro__
        for book in klass.__dict__.get("_own_declarations", {}).values()
        for decl in book.values()
    }

    def known(ref) -> bool:
        return id(ref) in own and ref.name in merged.get(ref.kind, {})

    for kind in ("input", "output", "state"):
        for name, decl in merged[kind].items():
            ref = decl.size
            if isinstance(ref, Declaration) and not known(ref):
                raise TypeError(
                    f"{cls.__qualname__}.{name}: size= refers to {ref!r}, which the class does not declare"
                )
    for name, decl in merged["input"].items():
        ref = decl.default
        if isinstance(ref, Declaration) and not known(ref):
            raise TypeError(
                f"{cls.__qualname__}.{name}: default= refers to {ref!r}, which the class does not declare"
            )
    for kind in ("output", "state"):
        for name, decl in merged[kind].items():
            ref = decl.init
            if isinstance(ref, Declaration) and not known(ref):
                raise TypeError(
                    f"{cls.__qualname__}.{name}: init= refers to {ref!r}, which the class does not declare"
                )
    for kind in ("parameter", "child"):
        for name, decl in merged[kind].items():
            for keyword in ("per_slot", "per_slot_if"):
                ref = getattr(decl, keyword, None)
                if ref is not None and not known(ref):
                    raise TypeError(
                        f"{cls.__qualname__}.{name}: {keyword}= refers to {ref!r}, which the class does not declare"
                    )


def _starts_a_run(initialize):
    """Wrap a System's own ``initialize`` (a System written the old way): after
    it, the base starts the run -- it records the step size and forgets the
    derived values of the last run (see ``System._forward_params``)."""
    if getattr(initialize, "_starts_a_run", False):
        return initialize

    @functools.wraps(initialize)
    def wrapper(self, *args, **kwargs):
        result = initialize(self, *args, **kwargs)
        step_size = kwargs.get("step_size", args[2] if len(args) > 2 else None)
        self._start_run(step_size)
        return result

    wrapper._starts_a_run = True
    return wrapper


class SystemMeta(type):
    """The metaclass of :class:`System`: files class-level declarations into
    registries (see the module docstring)."""

    @classmethod
    def __prepare__(mcls, name, bases, **kwargs):
        return _Namespace(name)

    def __new__(mcls, name, bases, namespace, **kwargs):
        declarations = getattr(namespace, "declarations", None)
        own_initialize = namespace.get("initialize") if "initialize" in namespace else None
        if callable(own_initialize) and bases:
            namespace = dict(namespace)
            namespace["initialize"] = _starts_a_run(own_initialize)
        cls = super().__new__(mcls, name, bases, dict(namespace), **kwargs)
        cls._own_declarations = {k: dict(v) for k, v in (declarations or {}).items() if v}
        merged = _merge(cls)
        _check_references(cls, merged)
        cls._declarations = merged
        cls._is_declared = any(merged[kind] for kind in KINDS)
        if cls._is_declared and "PARAM_NAMES" not in cls.__dict__:
            names = list(merged["parameter"])
            shared = getattr(cls, "SHARED_UNIT_PARAMS", {})
            for child_name, child in merged["child"].items():
                for sub in getattr(child.cls, "PARAM_NAMES", ()):
                    if f"{child_name}.{sub}" not in shared:  # read from its source
                        names.append(f"{child_name}.{sub}")
            cls.PARAM_NAMES = tuple(names)
        # A renamed port's old name connects through the class's alias tables
        # (``System.resolve_output_port`` / ``resolve_input_port``).
        for kind, table in (("output", "OUTPUT_PORT_ALIASES"), ("input", "INPUT_PORT_ALIASES")):
            aliases = {decl.alias: name for name, decl in merged[kind].items() if decl.alias}
            if aliases and table not in cls.__dict__:
                setattr(cls, table, aliases)
        return cls
