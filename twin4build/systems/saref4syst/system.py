from __future__ import annotations

# Standard library imports
import datetime
import inspect
from typing import Any, FrozenSet, Hashable, List, Optional, Tuple, Union

import copy

# Third party imports
import numpy as np
import torch
import torch.nn as nn

import twin4build.utils.types as tps

# from twin4build.utils.plot.simulation_result import SimulationResult
from prettytable import PrettyTable

# Local application imports
import twin4build.utils.types as tps
from twin4build.systems.saref4syst.declarations import REQUIRED, Declaration, SystemMeta
from twin4build.utils.deprecation import deprecate_name, reject_unexpected_kwargs
from twin4build.utils.logger import LOGGER
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.rhasattr import rhasattr
from twin4build.utils.simulation_time import get_simulation_timesteps
from twin4build.utils.state_marker import StateMarker


def _setting_reader(name: str):
    """A state's ``init_fn`` reading its initial value from the setting ``name``."""

    def read(component):
        return torch.as_tensor(getattr(component, name), dtype=tps.float_dtype())

    return read


class AliasedPorts(dict):
    """
    The ports of a component, which also answer to the deprecated names of
    renamed ports.

    This is a plain ``dict`` of the ports under their current names:
    iteration, ``keys()``, ``values()``, ``items()`` and ``len()`` see the
    current names only, so everything that walks the ports of a component
    (initialization, serialization, the functional rollout) sees one name per
    port.  A lookup by a deprecated name (``ports[name]``, ``ports.get(name)``)
    returns the port that replaced it and emits a ``DeprecationWarning``;
    ``name in ports`` is true for a deprecated name and does not warn.

    Args:
        ports: The ports, keyed by their current names.
        aliases: Deprecated name -> current name.
        owner: The name of the owner of the ports (its class name), used in
            the warning.
    """

    #: Class-level defaults: a copy or an unpickled instance is filled item
    #: by item before its attributes are restored.
    aliases: dict = {}
    owner: str = ""

    def __init__(
        self,
        ports: Union[dict, None] = None,
        aliases: Union[dict, None] = None,
        owner: str = "",
    ):
        super().__init__({} if ports is None else ports)
        self.aliases = dict({} if aliases is None else aliases)
        self.owner = owner

    def resolve(self, name: str, stacklevel: int = 3) -> str:
        """
        The current name of a port.

        A deprecated name is replaced by the name that stands for it, with a
        ``DeprecationWarning``; any other name is returned unchanged.

        Args:
            name: The name of the port, current or deprecated.
            stacklevel: The frame the warning points at, counted from this
                method (``2`` is its caller, ``3`` the caller of the caller,
                which is the user's code for ``ports[name]``).

        Returns:
            str: The current name of the port.
        """
        new = self.aliases.get(name) if isinstance(name, str) else None
        if new is None or dict.__contains__(self, name):
            return name
        prefix = f"{self.owner}." if self.owner else ""
        deprecate_name(f"{prefix}{name}", f"{prefix}{new}", stacklevel=stacklevel + 1)
        return new

    def __getitem__(self, name):
        return dict.__getitem__(self, self.resolve(name))

    def __contains__(self, name):
        if dict.__contains__(self, name):
            return True
        return isinstance(name, str) and dict.__contains__(
            self, self.aliases.get(name)
        )

    def get(self, name, default=None):
        if name in self:
            return dict.__getitem__(self, self.resolve(name))
        return default

    def __getattr__(self, name):
        # Attribute access to a port (``damper.output.airFlowRate``); only
        # reached when no dict attribute has the name.
        if name.startswith("__") or name in ("aliases", "owner"):
            raise AttributeError(name)
        if name in self:
            return self[name]
        raise AttributeError(f"{self.owner or 'ports'} has no port {name!r}")


def _concatenate_capacities(parts, n_c):
    """Units' ``(n_c | 1, width)`` capacities side by side, ``(n_c, sum)``;
    a unit without state contributes nothing."""
    parts = [part.detach().cpu() for part in parts if part is not None]  # a NaN part is made on the CPU
    if not parts:
        return None
    dtype = parts[0].dtype
    return torch.cat(
        [part.to(dtype).expand(n_c, part.shape[-1]) if part.shape[0] == 1 else part.to(dtype) for part in parts],
        dim=-1,
    )


_UNSIZED_WARNED = set()


def _warn_unsized(owner, leaf: str) -> None:
    """Warn once per class and attribute that a size quantity kept its
    absolute fallback bounds (nothing sized it, see
    :meth:`System.size_parameter`)."""
    key = (type(owner).__name__, leaf)
    if key in _UNSIZED_WARNED:
        return
    _UNSIZED_WARNED.add(key)
    LOGGER.warning(
        "%s.%s is a size quantity nothing sized: it keeps the class's absolute bounds; "
        "size it (System.size_parameter) for bounds relative to its size",
        key[0], leaf,
    )


class System(nn.Module, metaclass=SystemMeta):
    """
    A base-class representing a component model used as part of a simulation model.

    A System is written as class-level declarations plus one ``forward``
    (ADR 0001)::

        class ValveSystem(System):
            valvePosition = Input()
            valvePosition = Output()
            waterFlowRate = Output()
            waterFlowRateMax = Parameter(0.016, lb=5e-3, ub=1.0, scaling="log")

            def forward(self, x, inputs, p, step_size):
                ...

    See :mod:`twin4build.systems.saref4syst.declarations`.  The base builds
    the ports, parameters, settings, states and children from the
    declarations, sizes the vector ports, and steps the System through
    ``forward``.  Systems written the old way, with their own ``__init__``,
    ``initialize`` and ``do_step``, keep working until they are converted.

    Tensor-based subclasses should also implement one pure ``forward`` function
    and delegate ``do_step`` mathematics to it. See
    :doc:`/manual/differentiable_system_models` for the complete transform,
    device, cache, and structural-support contract.

    Args:
        connects_at: A list of connection points (inputs) of the system.
        connected_through: A list of Connection objects (outputs) of the system.
        input: A dictionary of inputs to the system.
        output: A dictionary of outputs from the system.
        id: The id of the system.
    """

    #: Deprecated output port names and the names that replaced them
    #: (``{"old": "new"}``).  A class that renames an output port lists the
    #: old name here and holds its outputs in an :class:`AliasedPorts`; the
    #: old name then still connects, loads and reads, with a
    #: ``DeprecationWarning``, and the model holds the new name only (see
    #: :meth:`resolve_output_port`).  Empty on the base class.
    OUTPUT_PORT_ALIASES: dict = {}

    #: Deprecated input port names and the names that replaced them
    #: (``{"old": "new"}``), the input-side counterpart of
    #: :attr:`OUTPUT_PORT_ALIASES`: a class that renames an input port lists
    #: the old name here and holds its inputs in an :class:`AliasedPorts`
    #: (see :meth:`resolve_input_port`).  Empty on the base class.
    INPUT_PORT_ALIASES: dict = {}

    #: Deprecated constructor keywords the class translates itself
    #: (``{"old": "new"}``, ``None`` for one that is only dropped), with a
    #: ``DeprecationWarning`` (``deprecate_args``).  Part of
    #: :meth:`constructor_keywords`, so a model saved with an old name loads.
    #: Empty on the base class.
    DEPRECATED_KEYWORDS: dict = {}

    #: Per class, the keyword arguments its constructor takes (see
    #: :meth:`constructor_keywords`).
    _CONSTRUCTOR_KEYWORDS: dict = {}

    @classmethod
    def constructor_keywords(cls) -> FrozenSet[str]:
        """The keyword arguments the constructor of the class takes.

        The named parameters of every ``__init__`` along the class's MRO
        that hands its ``**kwargs`` on, down to :class:`System`'s own, which
        rejects the rest, and the deprecated names each of those classes
        translates (:attr:`DEPRECATED_KEYWORDS`).  The translator and the model loader build a
        component from a node's literals, which also name properties the
        class does not take; they pass only these.

        Returns:
            FrozenSet[str]: The keyword names.
        """
        cached = System._CONSTRUCTOR_KEYWORDS.get(cls)
        if cached is not None:
            return cached
        names = set()
        for klass in cls.__mro__:
            init = klass.__dict__.get("__init__")
            if init is None:
                continue
            params = list(inspect.signature(init).parameters.values())
            names.update(
                p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            )
            names.update(klass.__dict__.get("DEPRECATED_KEYWORDS", {}))
            if klass is System or not any(p.kind is p.VAR_KEYWORD for p in params):
                break
        names.discard("self")
        declared = getattr(cls, "_declarations", {})
        for kind in ("setting", "parameter", "child"):
            names.update(declared.get(kind, {}))
        if getattr(cls, "_is_declared", False):
            # a declared class has no __init__ of its own: the base
            # translates its deprecated names
            names.update(getattr(cls, "DEPRECATED_KEYWORDS", {}))
        System._CONSTRUCTOR_KEYWORDS[cls] = frozenset(names)
        return System._CONSTRUCTOR_KEYWORDS[cls]

    def __str__(self):
        t = PrettyTable(field_names=["input", "output"], divider=True)
        title = f"Component overview    id: {self.id}"
        t.title = title
        input_list = list(self.input.keys())
        output_list = list(self.output.keys())
        n = max(len(input_list), len(output_list))
        for j in range(n):
            i = input_list[j] if j < len(input_list) else ""
            o = output_list[j] if j < len(output_list) else ""
            t.add_row([i, o], divider=True if j == len(input_list) - 1 else False)

        return t.get_string()

    def __init__(
        self,
        connects_at: Union[list, None] = None,
        connected_through: Union[list, None] = None,
        input: Union[dict, None] = None,
        output: Union[dict, None] = None,
        id: Union[str, None] = None,
        **kwargs,
    ):
        """
        Initialize a System object.

        Args:
            connects_at: A list of connection points (inputs) of the system.
            connected_through: A list of Connection objects (outputs) of the system.
            input: A dictionary of inputs to the system.
            output: A dictionary of outputs from the system.
            id: The id of the system.

        Raises:
            TypeError: On a keyword argument that neither the component's
                class nor :class:`System` takes (a misspelled keyword would
                otherwise be dropped without a word).
        """
        if "_parameters" not in self.__dict__:
            nn.Module.__init__(self)
        declared_values = self._take_declared_values(kwargs) if type(self)._is_declared else None
        for name in kwargs:
            removed = type(self).REMOVED_ATTRIBUTES.get(name)
            if removed is not None:
                raise TypeError(f"{type(self).__name__}({name}=...) is gone in 2.0: {removed}")
        reject_unexpected_kwargs(type(self).__name__, kwargs)
        assert isinstance(connects_at, list) or connects_at is None, (
            'Attribute "connects_at" is of type "'
            + str(type(connects_at))
            + '" but must be of type "'
            + str(list)
            + '"'
        )
        assert isinstance(connected_through, list) or connected_through is None, (
            'Attribute "connected_through" is of type "'
            + str(type(connected_through))
            + '" but must be of type "'
            + str(list)
            + '"'
        )
        assert isinstance(input, dict) or input is None, (
            'Attribute "input" is of type "'
            + str(type(input))
            + '" but must be of type "'
            + str(dict)
            + '"'
        )
        assert isinstance(output, dict) or output is None, (
            'Attribute "output" is of type "'
            + str(type(output))
            + '" but must be of type "'
            + str(dict)
            + '"'
        )
        assert isinstance(id, str), (
            'Attribute "id" is of type "'
            + str(type(id))
            + '" but must be of type "'
            + str(str)
            + '"'
        )
        if connects_at is None:
            connects_at = []
        if connected_through is None:
            connected_through = []
        if input is None:
            input = {}
        if output is None:
            output = {}
        self._connects_at = connects_at
        self._connected_through = connected_through
        self._input = input
        self._output = output
        self._id = id
        self._n_c = 1  # Number of parallel components (for vectorization)
        if declared_values is not None:
            self._build_declared(declared_values)

    #: Attributes a class no longer has, and what replaced them: reading or
    #: setting one raises with the replacement (setting it would otherwise
    #: quietly do nothing).
    REMOVED_ATTRIBUTES: dict = {}

    def _renamed(self, name: str):
        """The current name of a declared System's attribute ``name`` when it
        is an old keyword the class still translates (``DEPRECATED_KEYWORDS``),
        with a ``DeprecationWarning``; else ``None``."""
        if not type(self).__dict__.get("_is_declared", getattr(type(self), "_is_declared", False)):
            return None
        new = getattr(type(self), "DEPRECATED_KEYWORDS", {}).get(name)
        if not new or "." in new:
            return None
        cls_name = type(self).__name__
        deprecate_name(f"{cls_name}.{name}", f"{cls_name}.{new}", stacklevel=4)
        return new

    def __getattr__(self, name):
        removed = type(self).REMOVED_ATTRIBUTES.get(name)
        if removed is not None:
            raise AttributeError(f"{type(self).__name__}.{name} is gone in 2.0: {removed}")
        if not name.startswith("_"):
            new = self._renamed(name)
            if new is not None:  # an old name reads the current one
                return getattr(self, new)
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        removed = type(self).REMOVED_ATTRIBUTES.get(name)
        if removed is not None:
            raise AttributeError(f"{type(self).__name__}.{name} is gone in 2.0: {removed}")
        if not name.startswith("_"):
            new = self._renamed(name)
            if new is not None:  # an old name sets the current one (it set nothing before)
                name = new
        # A property of the class runs its setter, as on any Python object.
        # nn.Module would otherwise register a module or parameter value under
        # the property's name and skip the setter (a sensor's
        # ``time_series_input``).
        if isinstance(getattr(type(self), name, None), property):
            object.__setattr__(self, name, value)
            return
        # A declared setting converts every value it is given (a reader from
        # the dict a saved model's parameter file holds).
        setting = type(self).__dict__.get("_declarations", type(self)._declarations)["setting"].get(name)
        if setting is not None and setting.convert is not None:
            current = self.__dict__.get(name)
            if isinstance(value, dict) and hasattr(current, "to_config") and current.to_config() == value:
                return  # the reader it already holds (a DataFrame reader's dict has no data)
            value = setting.convert(value)
        parameter = type(self)._declarations["parameter"].get(name)
        if parameter is not None and value is not None and not isinstance(
            value, (tps.Parameter, tps.TensorParameter)
        ):
            # a number for a declared parameter (``damper.c = -1.9``): the
            # parameter, with its declared scaling
            if isinstance(value, torch.Tensor):
                data = value.detach().clone()
            else:
                data = torch.as_tensor(value, dtype=tps.float_dtype()).clone()
            value = tps.Parameter(data.to(tps.float_dtype()), requires_grad=False, scaling=parameter.scaling)
        # Replacing a registered parameter or child with something torch does
        # not register (a ``TensorParameter`` -- what ``set_parameters`` leaves
        # on a parameter that is set but not estimated -- or a plain value)
        # unregisters the name, as on a System that is no module.
        for registry, kind in (("_parameters", nn.Parameter), ("_modules", nn.Module)):
            registered = self.__dict__.get(registry)
            if registered is not None and name in registered and value is not None and not isinstance(value, kind):
                del registered[name]
        super().__setattr__(name, value)

    # -- Declared Systems (ADR 0001) ------------------------------------------

    def _take_declared_values(self, kwargs: dict) -> dict:
        """Pop the declared settings, parameter values and child overrides
        from the constructor's keywords; a missing required setting raises."""
        declared = type(self)._declarations
        for old, new in getattr(type(self), "DEPRECATED_KEYWORDS", {}).items():
            if old in kwargs:
                value = kwargs.pop(old)
                cls_name = type(self).__name__
                if new is not None:
                    deprecate_name(f"{cls_name}({old}=...)", f"{cls_name}({new}=...)", stacklevel=4)
                    child, _, leaf = new.partition(".")
                    if leaf:  # a value of a child: into the child's dict
                        given = dict(kwargs.get(child) or {})
                        given.setdefault(leaf, value)
                        kwargs[child] = given
                    else:
                        kwargs.setdefault(new, value)
        values = {"setting": {}, "parameter": {}, "child": {}}
        for name, decl in declared["setting"].items():
            if name in kwargs:
                values["setting"][name] = kwargs.pop(name)
            elif decl.default is REQUIRED:
                raise TypeError(f"{type(self).__name__}() needs the setting {name!r}")
            else:
                values["setting"][name] = copy.deepcopy(decl.default)
            if decl.convert is not None:
                values["setting"][name] = decl.convert(values["setting"][name])
        for name, decl in declared["parameter"].items():
            values["parameter"][name] = kwargs.pop(name, decl.value)
        for name in declared["child"]:
            override = kwargs.pop(name, None)
            if override is not None and not isinstance(override, dict):
                raise TypeError(
                    f"{type(self).__name__}(): {name}= takes a dict of the child's values, got {override!r}"
                )
            values["child"][name] = override or {}
        return values

    def _build_declared(self, values: dict) -> None:
        """Build the runtime settings, ports, parameters, states and
        children from the class's declarations."""
        cls = type(self)
        declared = cls._declarations
        for name, value in values["setting"].items():
            setattr(self, name, value)
        self._input = AliasedPorts(
            {name: self._declared_port(decl) for name, decl in declared["input"].items()},
            aliases={d.alias: n for n, d in declared["input"].items() if d.alias},
            owner=cls.__name__,
        )
        series_outputs = cls._is_data_source()
        self._output = AliasedPorts(
            {name: self._declared_port(decl, leaf=series_outputs) for name, decl in declared["output"].items()},
            aliases={d.alias: n for n, d in declared["output"].items() if d.alias},
            owner=cls.__name__,
        )
        bounds = {}
        computed = []
        for name, decl in declared["parameter"].items():
            value = values["parameter"][name]
            if callable(value) and not isinstance(value, (torch.Tensor, nn.Module)):
                computed.append((name, value))  # a default computed from the System
                value = None
            setattr(self, name, value)  # None: absent until given
            spec = decl.bounds()
            if spec is not None:
                bounds[name] = spec
        self.parameter = bounds
        for name, compute in computed:
            setattr(self, name, compute(self))
        for name, decl in declared["state"].items():
            size = decl.size if isinstance(decl.size, int) else None
            if size is None and getattr(decl.size, "kind", None) == "setting":
                size = self._setting_size(decl.size.name)
            from_setting = isinstance(decl.init, Declaration)
            setattr(
                self,
                name,
                tps.State(
                    n_v=size,
                    init_value=0.0 if from_setting else decl.init,
                    init_fn=_setting_reader(decl.init.name) if from_setting else None,
                    lb=decl.lb,
                    ub=decl.ub,
                    names=decl.names,
                ),
            )
        config = [n for n, d in declared["setting"].items() if d.save] + list(declared["parameter"])
        self._child_overrides = {k: v for k, v in values["child"].items() if v}
        for name, decl in declared["child"].items():
            child = decl.cls(id=f"{self._id}_{name}", **{**decl.values, **values["child"][name]})
            setattr(self, name, child)
            config.extend(f"{name}.{sub}" for sub in child.config.get("parameters", ()))
        self._config = {"parameters": config}
        self.INITIALIZED = False

    @classmethod
    def _is_data_source(cls) -> bool:
        """A System whose outputs are a series known before the run: it
        defines ``series(timestamps, step_size)`` and no ``forward``."""
        return getattr(cls, "series", None) is not None and getattr(cls, "forward", None) is nn.Module.forward

    @staticmethod
    def _declared_port(decl, leaf: bool = False):
        default = getattr(decl, "default", None)
        if decl.size is None:
            return tps.Scalar(default, optional=default is not None, is_leaf=leaf)
        n_v = decl.size if isinstance(decl.size, int) else None
        return tps.Vector(default, n_v=n_v, optional=default is not None, is_leaf=leaf)

    def _setting_size(self, name: str) -> int:
        value = getattr(self, name)
        if value is None:
            return 0
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return len(value)

    def declared_size(self, kind: str, name: str, _seen=None) -> Optional[int]:
        """The width of a declared vector port (``kind`` ``"input"`` or
        ``"output"``) or state, from its ``size=`` and the connections;
        ``None`` for a scalar port."""
        decl = type(self)._declarations[kind][name]
        size = decl.size
        if size is None:
            return None
        if isinstance(size, int):
            return size
        if size == "wired":
            from twin4build.utils.slots import wired_width

            if kind != "input":
                return 0
            # Nothing wired: keep a width set by hand (a System stepped on its own).
            return wired_width(self, name) or (self.input[name]._n_v or 0)
        if size.kind == "setting":
            return self._setting_size(size.name)
        seen = set() if _seen is None else _seen
        key = (size.kind, size.name)
        if key in seen:
            raise TypeError(f"{type(self).__name__}.{name}: size= refers to itself through {size.name!r}")
        seen.add(key)
        width = self.declared_size(size.kind, size.name, seen)
        return 0 if width is None else width

    def size_declared_ports(self) -> None:
        """Set the width of every declared vector port and state from its
        ``size=`` and the connections, and check that the wiring fits: an
        input's wired slots lie within its size."""
        from twin4build.utils.slots import wired_width

        declared = type(self)._declarations
        for kind, ports in (("input", self.input), ("output", self.output)):
            for name, decl in declared[kind].items():
                if decl.size is None:
                    continue
                n_v = self.declared_size(kind, name)
                if kind == "input" and decl.size != "wired":
                    wired = wired_width(self, name)
                    if wired > n_v:
                        raise ValueError(
                            f"{type(self).__name__} '{self.id}': input {name!r} is wired up to slot "
                            f"{wired - 1}, but its size is {n_v}"
                        )
                ports[name]._n_v = n_v
        for name, decl in declared["state"].items():
            if not isinstance(decl.size, int):
                getattr(self, name)._n_v = self.declared_size("state", name)
        self.__dict__.pop("_state_size_cache", None)

    def _initialize_declared(self, n_t: int, n_s: int, n_c: int, run=None) -> None:
        """Run start of a declared System: size and allocate the ports,
        expand the parameters to one value per copy, start the states, and
        fill a data source's outputs from its ``series`` (``run`` is the
        run's ``(timestamps, own steps per period, step size)``)."""
        self.n_c = n_c
        self.size_declared_ports()
        data_source = type(self)._is_data_source()
        for name, decl in type(self)._declarations["output"].items():
            if decl.init is not None:
                self.output[name].init_value = self._initial_value(decl.init)
        for ports in (self.input, self.output):
            for port in ports.values():
                if data_source and ports is self.output:
                    continue  # filled from ``series`` below
                if isinstance(port, tps.Vector):
                    port.initialize(n_t=n_t, n_s=n_s, n_c=n_c, n_v=port._n_v)
                else:
                    port.initialize(n_t=n_t, n_s=n_s, n_c=n_c)
        declared = type(self)._declarations
        for name, decl in declared["parameter"].items():
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.expand_to_n_c(n_c * self.slots_per_copy(decl)))
        for name in declared["state"]:
            getattr(self, name).initialize(n_s=n_s, n_c=n_c, component=self, force=True)
        for name, decl in declared["child"].items():
            child = getattr(self, name)
            if type(child)._is_declared:
                child._initialize_declared(n_t, n_s, n_c * self.slots_per_copy(decl), run)
        if data_source:
            self._fill_from_series(n_t, n_s, n_c, *run)
        self.INITIALIZED = True

    def _initial_value(self, init) -> float:
        """An output's ``init=``: a number, a setting's value, or a state's
        initial value (one number)."""
        if isinstance(init, Declaration) and init.kind == "state":
            init = init.init
        if isinstance(init, Declaration):
            init = getattr(self, init.name)
        value = torch.as_tensor(init, dtype=tps.float_dtype()).reshape(-1)
        if value.numel() != 1:
            raise ValueError(
                f"{type(self).__name__} '{self.id}': an output starts at one number, got {value.numel()} values"
            )
        return float(value[0])

    def slots_per_copy(self, decl) -> int:
        """How many values a declared parameter, or copies a declared child,
        one copy of this System holds: the width of its ``per_slot`` input
        (while its ``per_slot_if`` setting is true), else 1.  Read once the
        ports are sized."""
        port = getattr(decl, "per_slot", None)
        condition = getattr(decl, "per_slot_if", None)
        if port is None or (condition is not None and not getattr(self, condition.name)):
            return 1
        return max(int(self.input[port.name]._n_v or 0), 1)

    def _fill_from_series(self, n_t, n_s, n_c, timestamps, own_steps, step_size) -> None:
        """Call ``series`` and fill every output with its values for the whole
        run. A period's steps past its own end hold its last value."""
        values = self.series(timestamps, step_size)
        for name, port in self.output.items():
            if name not in values:
                raise KeyError(f"{type(self).__name__} '{self.id}': series() gave no values for the output {name!r}")
            v = values[name]
            v = v.to(tps.float_dtype()) if isinstance(v, torch.Tensor) else torch.as_tensor(
                np.asarray(v, dtype=float), dtype=tps.float_dtype()
            )
            v = v.clone()
            if v.shape[:2] != (n_s, n_t):
                raise ValueError(
                    f"{type(self).__name__} '{self.id}': series() gave {name!r} of shape {tuple(v.shape)}, "
                    f"expected ({n_s}, {n_t}, ...)"
                )
            for b, own in enumerate(own_steps):
                own = int(own)
                if 0 < own < n_t:
                    v[b, own:] = v[b, own - 1]
            if isinstance(port, tps.Vector):
                n_v = v.shape[2] if v.dim() == 3 else 1
                v = v.reshape(n_s, n_t, n_v).permute(1, 0, 2).unsqueeze(2).expand(n_t, n_s, n_c, n_v)
                port.initialize(n_t=n_t, n_s=n_s, n_c=n_c, n_v=n_v, values=v)
            else:
                port.initialize(n_t=n_t, n_s=n_s, n_c=n_c, values=v.T.unsqueeze(-1).expand(n_t, n_s, n_c))

    @staticmethod
    def run_timestamps(start_time, end_time, step_size):
        """The ``(n_s, n_t)`` timestamps ``series`` is called with, the own
        steps of each period and the step size: a shorter period's row ends
        in ``None`` after its last step."""
        _, date_steps, n_t, own_steps = get_simulation_timesteps(start_time, end_time, step_size)
        timestamps = np.array(date_steps, dtype=object).reshape(len(start_time), n_t)
        for b, own in enumerate(own_steps):
            timestamps[b, int(own):] = None
        return timestamps, [int(n) for n in own_steps], System._scalar_step_size(step_size)

    @staticmethod
    def periods(timestamps, step_size):
        """The ``(start_time, end_time, step_size)`` lists of the periods in
        ``timestamps`` (as ``series`` gets them): a period ends one step after
        its last timestamp."""
        starts, ends = [], []
        for row in timestamps:
            own = [t for t in row if t is not None]
            starts.append(own[0])
            ends.append(own[-1] + datetime.timedelta(seconds=float(step_size)))
        return starts, ends, [step_size] * len(starts)

    @staticmethod
    def _fold(value: torch.Tensor, n_s: int, n_c: int) -> torch.Tensor:
        """``(n_s, n_c, ...)`` -> ``(n_s * n_c, ...)``: the one form
        ``forward`` sees, its first dimension the copies being stepped.  A
        port held once for every simulation or copy is broadcast."""
        if value.shape[0] != n_s or value.shape[1] != n_c:
            value = value.expand(n_s, n_c, *value.shape[2:])
        return value.reshape(n_s * n_c, *value.shape[2:])

    @staticmethod
    def _fold_parameter(value: torch.Tensor, n_s: int, n_c: int) -> torch.Tensor:
        """A parameter of one value per copy, ``(n_c, ...)``, repeated for
        every simulation: ``(n_s * n_c, ...)``. A shared value is left as is."""
        if n_s == 1 or value.dim() == 0 or value.shape[0] != n_c:
            return value
        return value.unsqueeze(0).expand(n_s, *value.shape).reshape(n_s * n_c, *value.shape[1:])

    @classmethod
    def _fold_tree(cls, value, n_s: int, n_c: int):
        """:meth:`_fold_parameter` on every tensor of a dict / tuple / list."""
        if isinstance(value, torch.Tensor):
            return cls._fold_parameter(value, n_s, n_c)
        if isinstance(value, dict):
            return {k: cls._fold_tree(v, n_s, n_c) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(cls._fold_tree(v, n_s, n_c) for v in value)
        return value

    def _declared_do_step(self, step_size, step_index: int) -> None:
        """One object-mode step of a declared System through ``forward``."""
        n_c = self.n_c
        n_s = next(iter({**self.input, **self.output}.values())).n_s
        inputs = {name: self._fold(port.get(), n_s, n_c) for name, port in self.input.items()}
        params = self._fold_tree(self._forward_params(step_size), n_s, n_c)
        stateful = bool(self.collect_states())
        x = self._fold(self.get_state(), n_s, n_c) if stateful else None
        x_next, outs = self.forward(x, inputs, params, self._scalar_step_size(step_size))
        for name, port in self.output.items():
            value = outs[name]
            port._set(value.reshape(n_s, n_c, *value.shape[1:]), i_t=step_index)
        if stateful:
            self.set_state(x_next.reshape(n_s, n_c, x_next.shape[-1]))

    @property
    def config(self) -> dict:
        """What the System saves with the model: the attribute paths under
        ``"parameters"``."""
        return self._config

    @config.setter
    def config(self, value: dict) -> None:
        self._config = value

    @property
    def _batch_init_kwargs(self) -> dict:
        """The constructor values a batched copy is built with: a declared
        System's settings and child overrides."""
        if type(self)._is_declared:
            values = {name: getattr(self, name) for name in type(self)._declarations["setting"]}
            values.update(self.__dict__.get("_child_overrides", {}))
            return values
        if "_batch_init_kwargs_value" in self.__dict__:
            return self.__dict__["_batch_init_kwargs_value"]
        raise AttributeError("_batch_init_kwargs")

    @_batch_init_kwargs.setter
    def _batch_init_kwargs(self, value: dict) -> None:
        self.__dict__["_batch_init_kwargs_value"] = value

    @property
    def connects_at(self) -> list:
        """
        Get the connection points.
        """
        return self._connects_at

    @property
    def connected_through(self) -> list:
        """
        Get the list of Connection objects (outputs) of the system.
        """
        return self._connected_through

    @property
    def input(self) -> dict:
        """
        Get the input of the system.
        """
        return self._input

    @property
    def output(self) -> dict:
        """
        Get the output of the system.
        """
        return self._output

    @property
    def id(self) -> str:
        """
        Get the id of the system.
        """
        return self._id

    @input.setter
    def input(self, value: dict) -> None:
        """
        Set the input of the system.
        """
        self._input = value

    @output.setter
    def output(self, value: dict) -> None:
        """
        Set the output of the system.
        """
        self._output = value

    @connects_at.setter
    def connects_at(self, value: list) -> None:
        """
        Set the connection points.
        """
        self._connects_at = value

    @connected_through.setter
    def connected_through(self, value: list) -> None:
        """
        Set the list of Connection objects (outputs) of the system.
        """
        self._connected_through = value

    @id.setter
    def id(self, value: str) -> None:
        """
        Set the id of the system.
        """
        self._id = value

    def resolve_output_port(self, output_port: str, stacklevel: int = 3) -> str:
        """
        The current name of an output port of the system.

        A deprecated name (a key of :attr:`OUTPUT_PORT_ALIASES`) is replaced
        by the name that stands for it, with a ``DeprecationWarning``; any
        other name is returned unchanged.  Everything that takes a
        ``(component, output port)`` pair from the user resolves the name
        here first, so connections, objectives and serialized models hold
        the current name only.

        Args:
            output_port: The name of the output port, current or deprecated.
            stacklevel: The frame the warning points at, counted from this
                method (``2`` is its caller, ``3`` the caller of the caller).

        Returns:
            str: The current name of the output port.
        """
        new = self.OUTPUT_PORT_ALIASES.get(output_port) if isinstance(output_port, str) else None
        if new is None or dict.__contains__(self.output, output_port):
            return output_port
        name = type(self).__name__
        deprecate_name(f"{name}.{output_port}", f"{name}.{new}", stacklevel=stacklevel + 1)
        return new

    def resolve_input_port(self, input_port: str, stacklevel: int = 3) -> str:
        """
        The current name of an input port of the system.

        A deprecated name (a key of :attr:`INPUT_PORT_ALIASES`) is replaced
        by the name that stands for it, with a ``DeprecationWarning``; any
        other name is returned unchanged.  Everything that takes a
        ``(component, input port)`` pair from the user (connections,
        ``fill_missing_inputs``, signature patterns) resolves the name here
        first, so the model holds the current name only.

        Args:
            input_port: The name of the input port, current or deprecated.
            stacklevel: The frame the warning points at, counted from this
                method (``2`` is its caller, ``3`` the caller of the caller).

        Returns:
            str: The current name of the input port.
        """
        new = self.INPUT_PORT_ALIASES.get(input_port) if isinstance(input_port, str) else None
        if new is None or dict.__contains__(self.input, input_port):
            return input_port
        name = type(self).__name__
        deprecate_name(f"{name}.{input_port}", f"{name}.{new}", stacklevel=stacklevel + 1)
        return new

    @property
    def n_c(self) -> int:
        """
        Get the number of parallel components (for n_c vectorization).
        """
        return self._n_c

    @n_c.setter
    def n_c(self, value: int) -> None:
        """
        Set the number of parallel components (for n_c vectorization).
        """
        self._n_c = value

    def get_n_v_from_connections(self, input_port_name: str) -> int:
        """
        Determine the required n_v (vector size) for a given input port by examining
        connection points and finding the largest input_port_index.

        Args:
            input_port_name: Name of the input port to check.

        Returns:
            int: Required n_v (max index + 1), or None if no connections found.
        """
        max_index = 0
        found_connection = False

        for cp in self.connects_at:
            if cp.input_port == input_port_name:
                # Get all indices from the input_port_index dict
                for index in cp.input_port_index.values():
                    found_connection = True
                    if isinstance(index, int):
                        idx_val = index
                    elif hasattr(index, "max"):
                        idx_val = int(index.max().item())
                    else:
                        idx_val = int(index)
                    max_index = max(max_index, idx_val)

        n_v = max_index + 1
        if found_connection == False:
            n_v = None
        return n_v  # Default to None if no connections

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
    ) -> None:
        """
        Initialize the system.

        Args:
            start_time (datetime.datetime): The start time of the simulation.
            end_time (datetime.datetime): The end time of the simulation.
            step_size (int): The step size of the simulation in seconds.
        """
        _, _, max_timesteps, _ = get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)
        if type(self)._is_declared:
            n_c = int(getattr(self, "_n_c_batched", 1) or 1)
            run = self.run_timestamps(start_time, end_time, step_size) if type(self)._is_data_source() else None
            self._initialize_declared(max_timesteps, batch_size, max(n_c, self.n_c), run)
            self._start_run(step_size)
            return
        for input in self.input.values():
            input.initialize(n_t=max_timesteps, n_s=batch_size, n_c=self.n_c)
        for output in self.output.values():
            output.initialize(n_t=max_timesteps, n_s=batch_size, n_c=self.n_c)

    def do_step(
        self,
        second_time: Optional[List[float]] = None,
        date_time: Optional[List[datetime.datetime]] = None,
        step_size: Optional[List[int]] = None,
        step_index: int = 0,
    ) -> None:
        """
        Do a single step of the system.

        Args:
            second_time (List[float]): The current time in seconds.
            date_time (List[datetime.datetime]): The current date and time.
            step_size (List[int]): The step size of the simulation in seconds.
            step_index (int): The current step index.
        """
        if type(self)._is_data_source():
            for port in self.output.values():
                port._set(i_t=step_index)
        elif type(self)._is_declared:
            self._declared_do_step(step_size, step_index)

    # -- Continuous state (multiple-shooting / collocation estimation) --------
    # State is a first-class ``tps.State`` member (declared next to parameters /
    # ports).  A System is "stateful" iff it owns any -- directly or inside an
    # owned sub-system.  These methods gather/scatter those states generically,
    # so no component needs bespoke get/set code; it just declares ``tps.State``.
    # ``tps.State`` is imported lazily to avoid the core <-> utils.types cycle.

    def collect_states(self, _visited=None) -> List[Any]:
        """Ordered list of owned ``tps.State`` objects.

        Walks this System's own ``tps.State`` attributes, then recurses into
        *owned* sub-systems -- plain-attribute ``System`` instances (e.g. an
        ``ss_model``) and registered ``nn.Module`` children (e.g. a composite's
        ``thermal`` / ``mass``).  Deterministic by attribute / child order;
        cycle-guarded.  Connected (upstream/downstream) components are never
        followed -- they live inside ``Connection`` / ``ConnectionPoint`` objects,
        not as direct System attributes -- so only genuinely owned state is
        collected.
        """

        if _visited is None:
            _visited = set()
        if id(self) in _visited:
            return []
        _visited.add(id(self))

        states: List[Any] = []
        subs = []
        for attr in vars(self).values():
            if isinstance(attr, StateMarker):
                states.append(attr)
            elif isinstance(attr, System) and attr is not self:
                subs.append(attr)
        named_children = getattr(self, "named_children", None)
        if callable(named_children):
            for _name, child in named_children():
                subs.append(child)
        for sub in subs:
            collect = getattr(sub, "collect_states", None)
            if callable(collect):
                states.extend(sub.collect_states(_visited))
        return states

    def is_stateful(self) -> bool:
        """True iff this System owns any continuous ``tps.State``."""
        return len(self.collect_states()) > 0

    @staticmethod
    def _scalar_step_size(step_size) -> float:
        """Scalar step size from the simulator's per-simulation step-size list.

        Components whose ``forward`` discretizes with a single step size
        cannot batch simulations with different step sizes; keep the explicit
        guard the old state-space ``do_step`` had.
        """
        if isinstance(step_size, (list, tuple)):
            assert all(
                s == step_size[0] for s in step_size
            ), "Component supports only a single step size across batched simulations."
            return float(step_size[0])
        return float(step_size)

    # The old name, for the Roomwise extension's do_step overrides until they
    # become forward-only Systems; removed with do_step (ADR 0001).
    _scalar_sample_time = _scalar_step_size

    def _forward_params(self, step_size=None) -> dict:
        """The ``forward`` params dict in object mode: the System's own
        parameter values (:attr:`PARAM_NAMES`; dotted paths reach into owned
        sub-objects) and, for a System that defines ``derive(p, step_size)``,
        what it derives from them.

        The derived values are computed once per run, at the first step
        (``forward`` caches nothing; ADR 0001, decision 5): the run start
        forgets the last run's (:meth:`_start_run`).
        """
        names = getattr(self, "PARAM_NAMES", ())
        params = {}
        for name in names:
            value = rgetattr(self, name)
            if value is not None:  # an absent (optional) parameter
                params[name] = value.get()
        if getattr(type(self), "derive", None) is None:
            return params
        derived = self.__dict__.get("_derived")
        if derived is None:
            if step_size is None:
                step_size = self.__dict__.get("_run_step_size")
            elif not isinstance(step_size, (int, float)):
                step_size = self._scalar_step_size(step_size)
            derived = self.derive(params, step_size)
            self.__dict__["_derived"] = derived
        return {**params, **derived}

    def full_params(self, p: dict, step_size) -> dict:
        """``p`` plus what this System derives from it (``derive``): the
        params its ``forward`` expects.  A composite's ``derive`` hands each
        child its full params this way."""
        if getattr(type(self), "derive", None) is None:
            return p
        return {**p, **self.derive(p, step_size)}

    def _start_run(self, step_size) -> None:
        """Run start, as far as the base is concerned for a System written the
        old way: record the run's step size and forget the derived values of
        the last run, for this System and every System it owns."""
        if step_size is not None and not isinstance(step_size, torch.Tensor):
            try:
                self.__dict__["_run_step_size"] = self._scalar_step_size(step_size)
            except (AssertionError, TypeError, ValueError):
                self.__dict__.pop("_run_step_size", None)
        self.__dict__.pop("_derived", None)
        for owned in self._owned_systems():
            owned.__dict__.pop("_derived", None)
            if step_size is not None and "_run_step_size" in self.__dict__:
                owned.__dict__["_run_step_size"] = self.__dict__["_run_step_size"]

    def _owned_systems(self, _visited=None) -> list:
        """Every System this one owns (attributes and registered children),
        recursively; connected Systems are never followed."""
        if _visited is None:
            _visited = {id(self)}
        out = []
        candidates = [v for v in vars(self).values() if isinstance(v, System)]
        candidates.extend(child for _, child in self.named_children() if isinstance(child, System))
        for sub in candidates:
            if id(sub) in _visited:
                continue
            _visited.add(id(sub))
            out.append(sub)
            out.extend(sub._owned_systems(_visited))
        return out

    #: Ports through which this component may be fused into a monolithic
    #: state-space block (see ``FusedStateSpaceSystem``).  A connection is
    #: *fusable* -- eliminable exactly at the matrix level -- iff its output
    #: port is in the sender's ``FUSABLE_OUTPUT_PORTS`` and its input port is
    #: in the receiver's ``FUSABLE_INPUT_PORTS``.  Empty on the base class:
    #: components opt in explicitly.
    FUSABLE_INPUT_PORTS: frozenset = frozenset()
    FUSABLE_OUTPUT_PORTS: frozenset = frozenset()

    def _ss_units(self):
        """The linear/bilinear state-space *leaf* units owned by this component,
        as ``[(prefix, unit), ...]`` in state order (the order
        :meth:`collect_states` walks them).

        Each unit exposes ``_build_matrices(p)``, ``_ss_layout()`` and
        ``_ss_support()``.  The latter declares conservative structural
        support for the D/E/F matrices so fusion never infers graph structure
        from a parameter-dependent tensor value.  The prefix is the attribute
        path used to route prefixed parameter names (empty for a leaf component
        that is its own unit).  Composites override this to enumerate their
        submodels.
        """
        if hasattr(self, "_build_matrices"):
            return [("", self)]
        raise NotImplementedError(
            f"{type(self).__name__} declares fusable ports but has no "
            "state-space units (_ss_units)."
        )

    def state_size(self) -> int:
        """Total continuous-state width ``D`` (sum of owned state widths).

        Cached after the first call: state markers are declared at
        construction, and ``collect_states`` walks ``vars(self)``, which a
        compiled forward (``torch.compile`` of the functional step) cannot
        trace.  Forwards that need the width every step read the cached int.
        """
        cached = self.__dict__.get("_state_size_cache")
        if cached is None:
            cached = sum(int(s.n_v) for s in self.collect_states())
            self.__dict__["_state_size_cache"] = cached
        return cached

    def get_state(self) -> torch.Tensor:
        """Current continuous state, ``(n_s, n_c, D)``, differentiable.

        Concatenates the owned ``tps.State`` values along the last axis.
        """
        parts = [s.get() for s in self.collect_states()]
        if not parts:
            raise ValueError(
                f"System '{self.id}' has no continuous state (no tps.State declared)."
            )
        return torch.cat(parts, dim=-1)

    def state_heat_capacities(self):
        """The heat capacity of every continuous state [J/K], ``(n_c, D)`` in
        state order: the heat a jump of one kelvin in the state creates or
        destroys.  ``NaN`` for a state that holds no heat (a concentration,
        a controller's memory).

        A leaf unit declares its own with ``_state_heat_capacities()`` (from
        its parameters' current values); a composite of several state-space
        units (:meth:`_ss_units`) concatenates its units'.  ``None`` for a
        component without state.  Detached and on the CPU: a tolerance, not
        a parameter path (the multiple-shooting tolerance by energy reads it
        and moves it where it needs it).
        """
        width = self.state_size()
        if width == 0:
            return None
        caps = None
        own = getattr(self, "_state_heat_capacities", None)
        if own is not None:
            caps = own()
        else:
            try:
                units = self._ss_units()
            except NotImplementedError:
                units = []
            if units and not (len(units) == 1 and units[0][1] is self):
                caps = _concatenate_capacities([unit.state_heat_capacities() for _, unit in units], self.n_c)
        if caps is None:
            return torch.full((self.n_c, width), float("nan"), dtype=tps.float_dtype())
        caps = caps.detach().cpu()
        if caps.shape[0] == 1 and self.n_c > 1:
            caps = caps.expand(self.n_c, caps.shape[-1])
        if tuple(caps.shape) != (self.n_c, width):
            raise ValueError(
                f"{type(self).__name__} '{self.id}': heat capacities of shape {tuple(caps.shape)} "
                f"for a state of ({self.n_c}, {width})"
            )
        return caps

    def state_output_ports(self) -> List[Optional[str]]:
        """The output port that publishes each continuous state as it is,
        one entry per state of a copy (``state_size()``) in state order;
        ``None`` for a state no output publishes as it is.

        A leaf unit declares ``STATE_OUTPUT_PORTS = {port: state index}``; a
        composite of several state-space units (:meth:`_ss_units`)
        concatenates its units' (it publishes their outputs under their
        names).  A sensor on such a port measures the state, and the
        multiple-shooting tolerance takes that sensor's sd as the state's
        scale.
        """
        width = self.state_size()
        if width == 0:
            return []
        own = getattr(type(self), "STATE_OUTPUT_PORTS", None)
        if own is not None:
            ports: List[Optional[str]] = [None] * width
            for port, k in own.items():
                ports[k] = port
            return ports
        try:
            units = self._ss_units()
        except NotImplementedError:
            units = []
        if units and not (len(units) == 1 and units[0][1] is self):
            ports = [p for _, unit in units for p in unit.state_output_ports()]
            if len(ports) == width:
                return ports
        return [None] * width

    def set_state(self, x: torch.Tensor) -> None:
        """Write continuous state from a ``(n_s, n_c, D)`` tensor.

        Splits ``x`` across the owned ``tps.State`` objects; gradients are
        preserved (no detach) so a decision-variable slice can be written in.
        """
        offset = 0
        for s in self.collect_states():
            w = int(s.n_v)
            s.set(x[..., offset : offset + w])
            offset += w

    def state_names(self) -> List[str]:
        """Per-dimension names of the concatenated state (for diagnostics)."""
        names: List[str] = []
        for s in self.collect_states():
            names.extend(s.names())
        return names

    def get_estimable_parameters(
        self,
    ) -> List[Tuple[Any, str, float, float, float]]:
        """Default ``parameters="auto"`` contract for a System.

        Walks the attribute paths in ``self._config["parameters"]`` (the
        project-wide registry of every tunable knob) and emits one
        ``(component, attr_path, x0, lb, ub)`` tuple for each entry that

          * resolves to a :class:`torch.nn.Parameter` (e.g. our
            :class:`twin4build.utils.types.Parameter`) -- skips plain
            Python floats stored as construction nominals, and
          * has both ``"lb"`` and ``"ub"`` in its owner's ``parameter``
            map, and
          * is not named by the owner's optional ``_inactive_parameters()``
            (parameters the current wiring leaves without effect).

        Owner resolution: ``"thermal.C_air"`` looks up
        ``self.thermal.parameter["C_air"]`` for bounds; an unprefixed
        path looks up ``self.parameter[leaf]``.

        A size quantity declares ``"relative": (lo, hi)`` next to its
        absolute ``lb`` / ``ub``: once :meth:`size_parameter` has sized it,
        ``lb`` / ``ub`` are ``lo`` / ``hi`` times the sized value; unsized,
        the absolute bounds stay and a warning names the attribute (once
        per class and attribute).  Subclasses with a
        bespoke parameter space (e.g.
        :class:`ControllerIdentificationSystem`) override this with
        their own discovery logic.

        Returns:
            List of ``(component, attr_path, x0, lb, ub)`` tuples,
            possibly empty when the system has no estimable knobs or has
            not been built yet.
        """
        cfg = getattr(self, "_config", None)
        if not isinstance(cfg, dict):
            return []
        paths = cfg.get("parameters", [])
        if not isinstance(paths, (list, tuple)):
            return []
        out: List[Tuple[Any, str, float, float, float]] = []
        for path in paths:
            if not isinstance(path, str):
                continue
            *prefix, leaf = path.split(".")
            try:
                owner = rgetattr(self, ".".join(prefix)) if prefix else self
            except AttributeError:
                continue
            param = getattr(owner, leaf, None)
            # Duck-typed by ``nn.Parameter`` rather than tps.Parameter so
            # this module avoids the ``core <-> utils.types`` import
            # cycle that pulling :class:`tps.Parameter` in here would
            # create.
            if not isinstance(param, torch.nn.Parameter):
                continue
            # A component may declare parameters that its current wiring
            # leaves without effect (e.g. the in-zone boundary wall of a
            # room whose ``boundaryTemperature`` port is unconnected); the
            # estimator would otherwise carry dead entries in theta.
            inactive = getattr(owner, "_inactive_parameters", None)
            if callable(inactive) and leaf in inactive():
                continue
            spec = getattr(owner, "parameter", None)
            if not isinstance(spec, dict):
                continue
            bounds = spec.get(leaf)
            if not isinstance(bounds, dict) or "lb" not in bounds or "ub" not in bounds:
                continue
            if "relative" in bounds and not bounds.get("sized", False):
                _warn_unsized(owner, leaf)
            # A batched meta's spec may hold one bound per instance
            # (``Model._stack_parameter_spec``); a scalar stays a float.
            lb = np.asarray(bounds["lb"], dtype=float).reshape(-1)
            ub = np.asarray(bounds["ub"], dtype=float).reshape(-1)
            lb = float(lb[0]) if lb.size == 1 else lb
            ub = float(ub[0]) if ub.size == 1 else ub
            # Log-scaled ``tps.Parameter`` instances assert ``lb > 0``
            # inside ``TensorParameter.__init__``.  Skip rather than
            # surface that assertion inside the estimator's
            # ``set_parameters`` call: the symptom there is opaque and
            # this is recoverable by either widening the bound on the
            # owning component or by leaving the parameter out of
            # estimation entirely.
            if getattr(param, "scaling", None) == "log" and float(np.min(lb)) <= 0.0:
                continue
            try:
                values = param.get().detach().reshape(-1)
                # A batched meta component holds one value per instance
                # (``n_c`` wide): the estimator takes the whole vector as
                # the start, with the spec's bounds (one per instance when
                # the batcher stacked differing specs).
                x0 = values.cpu().numpy().astype(float) if values.numel() > 1 else float(values[0].item())
            except Exception:  # noqa: BLE001
                continue
            out.append((self, path, x0, lb, ub))
        return out

    def curvature_group(self, path: str) -> Optional[Hashable]:
        """The curvature group of the estimable parameter at ``path``.

        A solver that models the curvature of a parameter block (a quasi-Newton
        matrix per independent block) may split the block into groups:
        entries of one group share a dense model, different groups none.  A
        component whose parameters are many and only locally coupled (a
        profile's knots, day by day) names a group per parameter; ``None``
        (the default) leaves the entry in its block's main group, with the
        windows' initial states and every component that names none.  Labels
        are compared within a block only.
        """
        return None

    def size_parameter(self, path: str, value=None) -> bool:
        """Mark the parameter at ``path`` as sized, its bounds relative.

        A size quantity (a nominal flow, a capacity, a volume) declares
        ``"relative": (lo, hi)`` in its owner's ``parameter`` spec next to
        its absolute ``lb`` / ``ub``.  Whatever sizes it (the translator
        mapping a value from the semantic model, a workflow's sizing rule)
        calls this: ``value`` (physical; one per instance or one for all)
        is set when given, and the instance's ``lb`` / ``ub`` become ``lo``
        / ``hi`` times the parameter's value.  The sized value is the
        anchor: a later value (a warm start, a fit's result) moves the
        parameter, not its bounds.

        Args:
            path: The attribute path, as in ``_config["parameters"]``
                (``"heat_recovery.primaryAirFlowRateMax"``).
            value: The sized value; ``None`` anchors at the current one.

        Returns:
            bool: Whether the bounds were made relative: ``False`` for an
            attribute without ``"relative"`` in its spec, or a value that
            is not positive (the absolute bounds then stay).
        """
        *prefix, leaf = path.split(".")
        owner = rgetattr(self, ".".join(prefix)) if prefix else self
        param = getattr(owner, leaf)
        if value is not None:
            current = param.get().detach()
            with torch.no_grad():
                param.set(
                    torch.as_tensor(value, dtype=current.dtype, device=current.device).expand_as(current).clone(),
                    normalized=False,
                )
        spec = getattr(owner, "parameter", None)
        entry = spec.get(leaf) if isinstance(spec, dict) else None
        if not isinstance(entry, dict) or "relative" not in entry:
            return False
        sized = np.asarray(param.get().detach().cpu(), dtype=float).reshape(-1)
        if not np.all(sized > 0):
            LOGGER.warning(
                "%s.%s: sized value %s is not positive; its absolute bounds stay",
                getattr(self, "id", type(self).__name__), path, sized,
            )
            return False
        lo, hi = entry["relative"]

        def bound(factor):
            b = factor * sized
            return float(b[0]) if b.size == 1 or np.all(b == b[0]) else b.tolist()

        # a new dict: the spec may be shared with another instance's
        owner.parameter = {**spec, leaf: {**entry, "lb": bound(lo), "ub": bound(hi), "sized": True}}
        return True

    def populate_config(self) -> dict:
        """
        Populate the config of the system.

        Returns:
            dict: The config of the system.
        """

        def extract_value(value):
            # ``tps.Parameter`` (an nn.Parameter) and ``tps.TensorParameter``
            # (what ``_write_parameters(overwrite=True)`` -- the estimator's
            # write path -- leaves behind) both publish their physical value
            # through ``get()``; serializing anything else would write the
            # object's repr into the graph.
            if hasattr(value, "to_config"):
                # A reader (a data source's ``source``): the dict it is
                # rebuilt from.
                return value.to_config()
            if callable(value) and not isinstance(value, (torch.nn.Module, type)):
                # A function (a transformation): its module:qualname
                # reference (None for one that cannot be imported again).
                from twin4build.utils.callable_ref import callable_to_ref

                return callable_to_ref(value, f"|CLASS: {type(self).__name__}|ID: {self.id}|")
            if isinstance(value, (tps.Parameter, tps.TensorParameter)) or (
                hasattr(value, "get") and hasattr(value, "detach")
            ):
                # Use .tolist() to convert numpy types to Python native types
                # This ensures values like np.float64(1.0) become 1.0
                return value.get().detach().cpu().numpy().flatten().tolist()
            elif isinstance(value, torch.Tensor):
                # A plain tensor (a parameter's bound, a normalisation
                # constant) is a list of numbers: the tensor itself is
                # neither a literal nor JSON.
                return value.detach().cpu().numpy().flatten().tolist()
            else:  # isinstance(value, (int, float, type(None))):
                return value

        d = {}
        for key, value in self.config.items():
            # Check if all keys in the value dict are valid attributes of the object
            cond = isinstance(value, dict) and all(
                [rhasattr(self, k) for k in value.keys()]
            )
            if cond:
                d[key] = self.populate_config(value)
            else:
                assert isinstance(
                    value, list
                ), f"Invalid config value type: {type(value)}. Must be a list of attributes."
                d[key] = {}
                for attr in value:
                    v = rgetattr(self, attr)
                    v = extract_value(v)
                    d[key][attr] = v
        return d
