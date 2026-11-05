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


class Input(Declaration):
    """An input port.

    Args:
        size: ``None`` for a scalar port. For a vector port: an int, ``"wired"``
            (as many slots as its connections fill), or another port or setting
            of the class (the same size; a setting holding a sequence gives its
            length).
        default: The value an unwired input holds (a number). ``None``: the
            input must be wired.
        alias: An old name of the port that still connects, with a
            ``DeprecationWarning``.
        doc: What the port carries, with its unit.
    """

    kind = "input"

    def __init__(self, size=None, default=None, alias: Optional[str] = None, doc: str = ""):
        super().__init__(doc)
        _check_size(size, "Input")
        if default is not None and (isinstance(default, bool) or not isinstance(default, (int, float))):
            raise TypeError(f"Input: default= takes a number, got {default!r}")
        self.size = size
        self.default = default
        self.alias = alias


class Output(Declaration):
    """An output port.

    Args:
        size: As for :class:`Input`.
        alias: An old name of the port that still connects and reads, with a
            ``DeprecationWarning``.
        doc: What the port carries, with its unit.
    """

    kind = "output"

    def __init__(self, size=None, alias: Optional[str] = None, doc: str = ""):
        super().__init__(doc)
        _check_size(size, "Output")
        self.size = size
        self.alias = alias


class State(Declaration):
    """A continuous state the System carries from one step to the next.

    Args:
        size: The number of state variables: an int, or another port or setting
            of the class.
        init: The value it starts from when the model sets no initial state.
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
        self.size = size
        self.init = init
        self.lb = lb
        self.ub = ub
        self.names = None if names is None else list(names)


class Parameter(Declaration):
    """A value of the System that can be estimated.

    Args:
        value: The default value.
        lb, ub: The range the estimator searches. A parameter without both is
            not estimated unless the caller gives bounds.
        scaling: ``"linear"`` or ``"log"`` (equal steps are equal ratios; needs
            ``lb > 0``).
        relative: ``(low, high)`` factors: once something sizes the parameter
            (:meth:`System.size_parameter`), its range is ``value * low`` to
            ``value * high``.
        doc: What the parameter is, with its unit.
    """

    kind = "parameter"

    def __init__(self, value, lb=None, ub=None, scaling: str = "linear",
                 relative: Optional[Sequence[float]] = None, doc: str = ""):
        super().__init__(doc)
        if scaling not in ("linear", "log"):
            raise TypeError(f"Parameter: scaling= is 'linear' or 'log', got {scaling!r}")
        self.value = value
        self.lb = lb
        self.ub = ub
        self.scaling = scaling
        self.relative = None if relative is None else tuple(relative)

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
        doc: What the setting chooses.
    """

    kind = "setting"

    def __init__(self, default=REQUIRED, doc: str = ""):
        super().__init__(doc)
        self.default = default


class Child(Declaration):
    """A child System of a composite: built with the composite, stepped by the
    composite's ``forward``.

    Args:
        cls: The child's class.
        **values: The child's constructor values. A user overrides them with
            ``name=dict(...)`` on the composite's constructor.
    """

    kind = "child"

    def __init__(self, cls, **values):
        super().__init__(values.pop("doc", ""))
        self.cls = cls
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


class SystemMeta(type):
    """The metaclass of :class:`System`: files class-level declarations into
    registries (see the module docstring)."""

    @classmethod
    def __prepare__(mcls, name, bases, **kwargs):
        return _Namespace(name)

    def __new__(mcls, name, bases, namespace, **kwargs):
        declarations = getattr(namespace, "declarations", None)
        cls = super().__new__(mcls, name, bases, dict(namespace), **kwargs)
        cls._own_declarations = {k: dict(v) for k, v in (declarations or {}).items() if v}
        merged = _merge(cls)
        _check_references(cls, merged)
        cls._declarations = merged
        cls._is_declared = any(merged[kind] for kind in KINDS)
        if cls._is_declared and "PARAM_NAMES" not in cls.__dict__:
            names = list(merged["parameter"])
            for child_name, child in merged["child"].items():
                for sub in getattr(child.cls, "PARAM_NAMES", ()):
                    names.append(f"{child_name}.{sub}")
            cls.PARAM_NAMES = tuple(names)
        return cls
