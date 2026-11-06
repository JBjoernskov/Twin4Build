r"""What each output of a component depends on: the signal graph the partition walks.

A model is partitioned into worlds by signals, not by components
(:mod:`twin4build.model.partition`).  A world that needs a signal from outside
it needs the producer of that signal only as far as the signal depends on it:
a room's surplus air flow is its supply less its exhaust, so the world that
needs it needs the room's terminals, not the room's temperature.  A class
declares, per output port, what the output depends on:

.. code-block:: python

    OUTPUT_DEPENDENCIES = {
        "airFlowRate": {"inputs": ("damperPosition", "fanSpeed"),
                        "parameters": ("nominalAirFlowRate", "a", "c")},
    }

* ``"inputs"``: the input ports the output reads;
* ``"state"`` (default ``False``): whether it reads the component's own state;
* ``"parameters"``: the :attr:`~twin4build.systems.saref4syst.system.System.PARAM_NAMES`
  entries it depends on.

An output a class does not declare depends on everything: every input port,
the state if the component has one, every parameter (the partition's rule
before these declarations).  A declaration may be wider than the truth -- a
class whose configurations use different inputs declares the union -- at the
cost of a copy the partition did not need; it must never be narrower.
:func:`derive` finds what an output depends on from the component's
functional step (``forward``) with automatic differentiation; the classes'
tests check every declaration against it, so a change of the equations that
adds a dependency fails a test instead of a cut.

Slots.  A slot of an output reads every slot of a vector input, unless the
instance says otherwise with ``output_slot_dependencies(output, input)``: a
``{output slot: input slots}`` map, or ``None`` for every slot.  It is a
method because the grouping is the instance's configuration (which supply
terminals are an exhaust box's source), fixed once the model is translated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Tuple

import torch

import twin4build as tb
from twin4build.systems.utils.smooth_saturation import saturation_mode
from twin4build.utils.rgetattr import rgetattr

#: The keys a declaration may hold.
_KEYS = frozenset({"inputs", "state", "parameters"})


@dataclass(frozen=True)
class Dependency:
    """What one output depends on (module docstring)."""

    inputs: FrozenSet[str]
    state: bool
    parameters: FrozenSet[str]

    def covers(self, other: "Dependency") -> bool:
        """Whether this dependency includes everything ``other`` depends on."""
        return (
            self.inputs >= other.inputs
            and (self.state or not other.state)
            and self.parameters >= other.parameters
        )

    def union(self, other: "Dependency") -> "Dependency":
        return Dependency(self.inputs | other.inputs, self.state or other.state, self.parameters | other.parameters)


def parameter_names(component) -> Tuple[str, ...]:
    """The keys of the component's ``forward`` params dict: :attr:`PARAM_NAMES`, or for a composite without them
    (a room of a thermal and a CO2 model) the attribute paths of its estimable parameters (``"thermal.C_air"``),
    which its ``forward`` routes to its parts -- the keys the fit and the consensus use either way."""
    names = getattr(component, "PARAM_NAMES", None)
    if names is not None:
        return tuple(names)
    getter = getattr(component, "get_estimable_parameters", None)
    return tuple(str(e[1]) for e in getter()) if callable(getter) else ()


def forward_params(component) -> Dict[str, torch.Tensor]:
    """The component's parameters at their current values (:func:`parameter_names`); what the
    component derives from them (``derive``) is added where ``forward`` is called, so that the
    derivatives reach the parameters."""
    return {
        name: rgetattr(component, name).get()
        for name in parameter_names(component)
        if rgetattr(component, name) is not None  # an absent (optional) parameter
    }


def has_state(component) -> bool:
    """Whether ``component`` carries a state.

    A component creates its states when it is initialised, and :meth:`System.state_size` keeps its first answer:
    asked before, it would say 0 and keep saying it.  So it is asked only once the component is initialised; before,
    a class whose every output is declared free of the state (and that declares no state update) has none, and any
    other is taken to have one (the safe side: its outputs then read it)."""
    if getattr(component, "INITIALIZED", False):
        size = getattr(component, "state_size", None)
        try:
            return callable(size) and size() > 0
        except Exception:  # a component that cannot tell counts as having one
            return True
    cls = type(component)
    if getattr(cls, "STATE_DEPENDENCIES", None) is not None:
        return True
    table = getattr(cls, "OUTPUT_DEPENDENCIES", None) or {}
    outputs = list(getattr(component, "output", {}) or {})
    return not (outputs and all(p in table and not table[p].get("state", False) for p in outputs))


def everything(component) -> Dependency:
    """The dependency of an undeclared output: every input, the state if any, every parameter."""
    return Dependency(frozenset(getattr(component, "input", {}) or {}), has_state(component),
                      frozenset(parameter_names(component)))


#: The pseudo-port of a component's state update (``x_next`` of ``forward``): a copy that carries the state needs
#: what the update reads.  Declared by ``STATE_DEPENDENCIES`` (``"inputs"``, ``"parameters"``; the state itself is
#: implied), everything otherwise.
STATE = "__state__"


def _table(component) -> Dict[str, Mapping]:
    table = dict(getattr(type(component), "OUTPUT_DEPENDENCIES", None) or {})
    state = getattr(type(component), "STATE_DEPENDENCIES", None)
    if state is not None:
        table[STATE] = dict(state, state=True)
    return table


def declared(component) -> Dict[str, Dependency]:
    """``{output port: Dependency}`` of ``component``: its class's declarations, everything for the rest; with a
    state, also :data:`STATE` (what the state update reads).

    A declaration naming an input port or a parameter the component does not have is an error (a typo would
    otherwise narrow the output silently)."""
    table = _table(component)
    inputs = set(getattr(component, "input", {}) or {})
    params = set(parameter_names(component))
    ports = list(getattr(component, "output", {}) or {}) + ([STATE] if has_state(component) else [])
    out = {}
    for port in ports:
        spec = table.get(port)
        if spec is None:
            out[port] = everything(component)
            continue
        unknown = set(spec) - _KEYS
        if unknown:
            raise ValueError(f"{type(component).__name__} dependencies of {port!r}: unknown keys {sorted(unknown)}")
        ins, ps = frozenset(spec.get("inputs", ())), frozenset(spec.get("parameters", ()))
        if not ins <= inputs:
            raise ValueError(f"{type(component).__name__} dependencies of {port!r}: no input ports {sorted(ins - inputs)}")
        if not ps <= params:
            raise ValueError(f"{type(component).__name__} dependencies of {port!r}: no parameters {sorted(ps - params)}")
        out[port] = Dependency(ins, bool(spec.get("state", False)), ps)
    return out


def is_declared(component, port: str) -> bool:
    return port in _table(component)


def projection(component, port: str) -> Optional[type]:
    """The class that computes ``component``'s output ``port`` on its own, or ``None``.

    A class with a state may name, per output that depends on neither its state nor a parameter, a stateless class
    that computes it from the same input ports (``OUTPUT_PROJECTIONS = {port: class or its name in twin4build}``):
    a world that needs that output but not the component gets an instance of it in the component's place.  A class
    without a state needs none (it is copied whole).  An output the projection could not compute exactly -- one
    with a state or a parameter, or a port the projection lacks -- is an error."""
    table = getattr(type(component), "OUTPUT_PROJECTIONS", None) or {}
    cls = table.get(port)
    if cls is None:
        return None
    if isinstance(cls, str):

        cls = getattr(tb, cls)
    dep = declared(component)[port]
    if dep.state or dep.parameters:
        raise ValueError(f"{type(component).__name__}.OUTPUT_PROJECTIONS[{port!r}]: the output depends on "
                         f"{'its state' if dep.state else 'parameters'}; a projection computes only state- and "
                         "parameter-free outputs")
    probe = cls(id="__projection_probe")
    missing = dep.inputs - set(probe.input)
    if port not in probe.output or missing:
        raise ValueError(f"{type(component).__name__}.OUTPUT_PROJECTIONS[{port!r}]: {cls.__name__} lacks "
                         f"{'the output' if port not in probe.output else 'the inputs ' + str(sorted(missing))}")
    return cls


def slots(component, output: str, input_port: str) -> Optional[Dict[int, Tuple[int, ...]]]:
    """``{output slot: input slots}`` the instance declares for ``output`` reading ``input_port``; ``None``: every
    slot of the output reads every slot of the input."""
    method = getattr(component, "output_slot_dependencies", None)
    return method(output, input_port) if callable(method) else None


# ---------------------------------------------------------------------------
# Derivation by automatic differentiation
# ---------------------------------------------------------------------------
def _perturbed(value: torch.Tensor, generator: torch.Generator, spread: float) -> torch.Tensor:
    """``value`` scaled by a random factor in ``1 +- spread`` per element, signs kept; a zero moved off zero."""
    value = value.detach().to(torch.float64)
    factor = 1.0 + spread * (2.0 * torch.rand(value.shape, generator=generator, dtype=torch.float64) - 1.0)
    nudge = 1e-3 * torch.rand(value.shape, generator=generator, dtype=torch.float64)
    return torch.where(value == 0, nudge, value * factor)


#: A derivative counts when it moves the output by more than this share of the output's size (relative to a change
#: of the variable by its own size): a ratio that cancels exactly (a flow-weighted mean scaled by one factor) leaves
#: a rounding-level gradient that is no dependency.
RELATIVE_TOLERANCE = 1e-9


def _matters(grad: Optional[torch.Tensor], value: torch.Tensor, out: torch.Tensor) -> bool:
    if grad is None:
        return False
    scale = float(out.detach().abs().max()) if out.numel() else 0.0
    effect = (grad.detach() * value.detach().abs().clamp_min(1e-12)).abs()
    return bool((effect > RELATIVE_TOLERANCE * max(scale, 1e-12)).any())


def derive(
    component,
    inputs,
    *,
    state: Optional[torch.Tensor] = None,
    params: Optional[Mapping[str, torch.Tensor]] = None,
    step_size: float = 600.0,
    points: int = 4,
    spread: float = 0.4,
    seed: int = 0,
    forward_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Dependency], Dict[Tuple[str, str], Dict[int, FrozenSet[int]]]]:
    """What each output of ``component`` depends on, found by differentiating its ``forward``.

    ``inputs`` is one realistic value per input port (the shapes ``forward`` takes: ``(n_c,)`` for a scalar port,
    ``(n_c, n_v)`` for a vector one), or a list of such dicts to sample several operating regimes (a fan stopped and
    running: a gate saturated at every point of one regime has no derivative there); ``state`` the component's state (default :meth:`get_state` when it has one);
    ``params`` its ``forward`` params (default ``_forward_params()``).  At each of ``points`` random points around
    them (every value scaled by a factor in ``1 +- spread``) the gradient of every output with respect to every
    input, the state and every parameter is taken in the smooth saturation mode; whatever has a nonzero gradient
    at any point is a dependency.  A dependency whose derivative vanishes at every sampled point (a hard clamp
    sampled on its flat side) is missed: the samples should exercise the component's operating range.

    Returns ``(dependencies, slots)``: ``{output: Dependency}`` and, for every vector input,
    ``{(output, input): {output slot: input slots with a nonzero gradient}}`` (a scalar output has slot 0)."""

    generator = torch.Generator().manual_seed(seed)
    templates = [inputs] if isinstance(inputs, Mapping) else list(inputs)
    if params is None:
        params = forward_params(component)
    if state is None and has_state(component):
        state = component.get_state()
    deps: Dict[str, Dependency] = {}
    slot_map: Dict[Tuple[str, str], Dict[int, set]] = {}
    for i in range(points * len(templates)):
        x_in = {k: _perturbed(v, generator, spread).requires_grad_(True) for k, v in templates[i % len(templates)].items()}
        p_in = {k: _perturbed(v, generator, spread).requires_grad_(True) for k, v in params.items()}
        x0 = None if state is None else (_perturbed(state, generator, spread).requires_grad_(True))
        with saturation_mode("smooth"):
            x_next, outs = component.forward(
                x0, dict(x_in), component.full_params(dict(p_in), step_size), step_size, **(forward_kwargs or {})
            )
        outs = dict(outs)
        if x0 is not None and isinstance(x_next, torch.Tensor):
            outs[STATE] = x_next
        names_in, names_p = list(x_in), list(p_in)
        leaves = [x_in[k] for k in names_in] + [p_in[k] for k in names_p] + ([x0] if x0 is not None else [])
        for port, out in outs.items():
            if not isinstance(out, torch.Tensor) or not out.is_floating_point() or not out.requires_grad:
                deps.setdefault(port, Dependency(frozenset(), False, frozenset()))
                continue
            grads = torch.autograd.grad(out.sum(), leaves, allow_unused=True, retain_graph=True)
            found = Dependency(
                frozenset(k for k, g in zip(names_in, grads[: len(names_in)]) if _matters(g, x_in[k], out)),
                x0 is not None and _matters(grads[-1], x0, out),
                frozenset(k for k, g in zip(names_p, grads[len(names_in): len(names_in) + len(names_p)])
                          if _matters(g, p_in[k], out)),
            )
            deps[port] = found if port not in deps else deps[port].union(found)
            # slots: per output slot, the slots of each vector input it reads
            vector_inputs = [k for k in names_in if x_in[k].dim() >= 2 and k in found.inputs]
            if not vector_inputs:
                continue
            flat = out.reshape(out.shape[0], -1) if out.dim() >= 2 else out.reshape(-1, 1)
            for s in range(flat.shape[-1]):
                gs = torch.autograd.grad(flat[:, s].sum(), [x_in[k] for k in vector_inputs], allow_unused=True,
                                         retain_graph=True)
                scale = max(float(flat[:, s].detach().abs().max()), 1e-12)
                for k, g in zip(vector_inputs, gs):
                    if g is None:
                        continue
                    effect = (g.detach() * x_in[k].detach().abs().clamp_min(1e-12)).abs()
                    hit = {int(j) for j in torch.nonzero(
                        (effect > RELATIVE_TOLERANCE * scale).reshape(g.shape[0], -1).any(dim=0)).reshape(-1)}
                    slot_map.setdefault((port, k), {}).setdefault(s, set()).update(hit)
    return deps, {key: {s: frozenset(v) for s, v in m.items()} for key, m in slot_map.items()}


def independence_violations(
    component,
    inputs: Mapping[str, torch.Tensor],
    *,
    state: Optional[torch.Tensor] = None,
    params: Optional[Mapping[str, torch.Tensor]] = None,
    step_size: float = 600.0,
    forward_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, list]:
    """The declarations checked without derivatives: every input, parameter and the state that a declared output
    does not declare is changed (by a large factor, one at a time) and the output must not move -- also through a
    hard switch, which has no derivative to find.  Returns ``{output: [what moved it]}``, empty when consistent."""
    if params is None:
        params = forward_params(component)
    if state is None and has_state(component):
        state = component.get_state()

    def run(x_in, p_in, x0):
        x_next, outs = component.forward(
            x0, dict(x_in), component.full_params(dict(p_in), step_size), step_size, **(forward_kwargs or {})
        )
        outs = dict(outs)
        if x0 is not None and isinstance(x_next, torch.Tensor):
            outs[STATE] = x_next
        return outs

    def bumped(v):
        v = v.detach()
        return v * 3.0 + torch.sign(v) + (v == 0).to(v.dtype)

    base_in = {k: v.detach() for k, v in inputs.items()}
    base_p = {k: v.detach() for k, v in params.items()}
    base_x = None if state is None else state.detach()
    base = run(base_in, base_p, base_x)
    table = declared(component)
    violations: Dict[str, list] = {}
    for port, out in base.items():
        if not is_declared(component, port) or not isinstance(out, torch.Tensor):
            continue
        decl = table[port]
        trials = [("input", k, dict(base_in, **{k: bumped(v)}), base_p, base_x) for k, v in base_in.items()
                  if k not in decl.inputs]
        trials += [("parameter", k, base_in, dict(base_p, **{k: bumped(v)}), base_x) for k, v in base_p.items()
                   if k not in decl.parameters]
        if base_x is not None and not decl.state:
            trials.append(("state", "x", base_in, base_p, bumped(base_x)))
        for kind, name, x_in, p_in, x0 in trials:
            moved = run(x_in, p_in, x0).get(port)
            if moved is not None and not torch.allclose(moved, out, rtol=RELATIVE_TOLERANCE, atol=0.0):
                violations.setdefault(port, []).append(f"{kind} {name}")
    return violations


def check(component, derived: Mapping[str, Dependency], tight: bool = True) -> Dict[str, str]:
    """Problems of ``component``'s declarations against ``derived`` (:func:`derive`), by output: an output that
    depends on more than it declares (unsound, always), or, with ``tight``, a declared output that declares more
    than it depends on.  Undeclared outputs depend on everything and are not checked.  Empty when consistent."""
    problems = {}
    table = declared(component)
    for port, found in derived.items():
        if not is_declared(component, port):
            continue
        decl = table[port]
        if not decl.covers(found):
            problems[port] = (f"depends on more than declared: inputs {sorted(found.inputs - decl.inputs)}, "
                              f"state {found.state and not decl.state}, parameters {sorted(found.parameters - decl.parameters)}")
        elif tight and decl != found:
            problems[port] = (f"declares more than it depends on: inputs {sorted(decl.inputs - found.inputs)}, "
                              f"state {decl.state and not found.state}, parameters {sorted(decl.parameters - found.parameters)}")
    return problems


def proposal(derived: Mapping[str, Dependency], params: Iterable[str] = ()) -> str:
    """``derived`` written as an ``OUTPUT_DEPENDENCIES`` literal, for a class's source (the declarations' starting
    point; review it before pasting)."""
    lines = ["OUTPUT_DEPENDENCIES = {"]
    for port, dep in sorted(derived.items()):
        if port == STATE:
            continue
        parts = [f'"inputs": {tuple(sorted(dep.inputs))!r}']
        if dep.state:
            parts.append('"state": True')
        parts.append(f'"parameters": {tuple(sorted(dep.parameters))!r}')
        lines.append(f'    "{port}": {{{", ".join(parts)}}},')
    lines.append("}")
    if STATE in derived:
        dep = derived[STATE]
        lines.append(f'STATE_DEPENDENCIES = {{"inputs": {tuple(sorted(dep.inputs))!r}, '
                     f'"parameters": {tuple(sorted(dep.parameters))!r}}}')
    return "\n".join(lines)
