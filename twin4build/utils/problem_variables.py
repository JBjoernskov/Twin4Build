"""The variables of an estimation or optimization problem, resolved to the
tuples the :class:`~twin4build.estimator.estimator.Estimator` and the
:class:`~twin4build.optimizer.optimizer.Optimizer` validate.

A :class:`~twin4build.utils.types.Variable` (or a bare parameter) names its
target by the object; the problem classes key everything on ``(component,
attribute path)``.  A parameter is found by identity among the parameters
the model's components own: the paths their ``_config["parameters"]``
declares, their own attributes, registered ``nn.Module`` children and owned
sub-systems (connected components live in connection objects, never as
attributes, so they are not followed), and what ``get_estimable_parameters``
reports.  The lookup does not depend on a parameter's state (frozen,
unbuilt, a ``TensorParameter`` left by ``set_parameters(overwrite=True)``).
An output port is found among the components' ``output`` ports.

Default bounds are the ones the component declares for the attribute
(``owner.parameter[leaf]``, as ``parameters="auto"`` uses), else the
parameter's own ``min_value`` / ``max_value``.

Resolve before the problem initializes the model: an ``initialize`` may
replace a parameter object (a multi-branch component widens it), and the
path is what stays.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.types import TensorParameter, Variable

LOGGER = logging.getLogger(__name__)


def _is_parameter(obj) -> bool:
    return isinstance(obj, (torch.nn.Parameter, TensorParameter))


def _is_spec(entry) -> bool:
    return isinstance(entry, Variable) or _is_parameter(entry)


def _as_list(entries):
    """``entries`` as a list when it holds specs (a tuple of Variables is a
    list of them); anything else unchanged."""
    if isinstance(entries, tuple) and any(_is_spec(e) for e in entries):
        return list(entries)
    return entries


def _owned_parameters(component) -> Dict[int, str]:
    """``{id(parameter): attribute path}`` of the parameters ``component``
    owns: its own attributes, registered ``nn.Module`` children and owned
    sub-systems, recursively."""
    from twin4build.systems.saref4syst.system import System

    out: Dict[int, str] = {}
    seen = set()

    def visit(obj, prefix: str, depth: int) -> None:
        if id(obj) in seen or depth > 6:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.nn.Module):
            for name, p in obj.named_parameters(recurse=False):
                out.setdefault(id(p), prefix + name)
            for name, child in obj.named_children():
                visit(child, f"{prefix}{name}.", depth + 1)
        for name, value in list(getattr(obj, "__dict__", {}).items()):
            if name.startswith("_"):
                continue
            if _is_parameter(value):
                out.setdefault(id(value), prefix + name)
            elif isinstance(value, (System, torch.nn.Module)):
                visit(value, f"{prefix}{name}.", depth + 1)

    visit(component, "", 0)
    return out


def _estimable(component) -> List[str]:
    """The paths ``get_estimable_parameters`` reports; a failure is logged."""
    try:
        return [attr for _c, attr, *_ in (component.get_estimable_parameters() or [])]
    except Exception as exc:  # noqa: BLE001 - one component must not hide the others
        LOGGER.warning("%s.get_estimable_parameters() failed (%r); its parameters are found by attribute only", component.id, exc)
        return []


def _parameter_index(model) -> Dict[int, Tuple[Any, str]]:
    """``{id(parameter): (component, attribute path)}`` over the model's
    components: the declared paths first (the canonical names), then the
    owned attributes, then what ``get_estimable_parameters`` reports."""
    index: Dict[int, Tuple[Any, str]] = {}
    for component in model.components.values():
        cfg = getattr(component, "_config", None)
        declared = [p for p in (cfg.get("parameters", []) if isinstance(cfg, dict) else []) if isinstance(p, str)]
        for path in declared:
            try:
                obj = rgetattr(component, path)
            except AttributeError:
                continue
            if _is_parameter(obj):
                index.setdefault(id(obj), (component, path))
        for pid, path in _owned_parameters(component).items():
            index.setdefault(pid, (component, path))
        for path in _estimable(component):
            try:
                obj = rgetattr(component, path)
            except AttributeError:
                continue
            if _is_parameter(obj):
                index.setdefault(id(obj), (component, path))
    return index


def _port_owner(model, port) -> Tuple[Any, str]:
    for component in model.components.values():
        for name, candidate in getattr(component, "output", {}).items():
            if candidate is port:
                return component, name
    raise ValueError("a trajectory Variable's port is no output port of the model's components")


def _resolve(variable: Variable, index) -> List[Tuple[Any, str, Any]]:
    """``[(component, path, parameter)]`` of a parameter Variable, one
    attribute of its components."""
    if len({id(t) for t in variable.targets}) != len(variable.targets):
        raise ValueError("a Variable names the same parameter twice")
    found = []
    for target in variable.targets:
        hit = index.get(id(target))
        if hit is None:
            raise ValueError("a Variable's parameter is no parameter of the model's components")
        found.append((hit[0], hit[1], target))
    paths = {path for _c, path, _t in found}
    if len(paths) != 1:
        raise ValueError(f"the parameters of one Variable must be the same attribute of their components, got {sorted(paths)}")
    return found


def _plain(value):
    array = np.asarray(value.detach().cpu() if isinstance(value, torch.Tensor) else value, dtype=float).reshape(-1)
    return float(array[0]) if array.size == 1 else array


def _declared_bounds(component, path: str, target) -> Tuple[Any, Any]:
    *prefix, leaf = path.split(".")
    try:
        owner = rgetattr(component, ".".join(prefix)) if prefix else component
    except AttributeError:
        owner = None
    spec = getattr(owner, "parameter", None)
    entry = spec.get(leaf) if isinstance(spec, dict) else None
    if isinstance(entry, dict) and "lb" in entry and "ub" in entry:
        return entry["lb"], entry["ub"]
    lo, hi = getattr(target, "_min_value", None), getattr(target, "_max_value", None)
    if lo is not None and hi is not None:
        return _plain(lo), _plain(hi)
    raise ValueError(f"{component.id}.{path} declares no bounds: give the Variable lb and ub")


def _bounds(variable: Variable, component, path: str, target) -> Tuple[Any, Any]:
    if variable.lb is not None and variable.ub is not None:
        return variable.lb, variable.ub
    lb, ub = _declared_bounds(component, path, target)
    return (variable.lb if variable.lb is not None else lb, variable.ub if variable.ub is not None else ub)


def _same(a, b) -> bool:
    return bool(np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float)))


def estimator_parameters(entries, model) -> List[Any]:
    """``parameters`` with every :class:`Variable` or bare parameter as the
    Estimator's tuple ``(component(s), attr, x0, lb, ub, components)``:
    a private Variable gives one tuple per component (each with its own
    bounds), a shared one a single tuple.  Tuples pass through unchanged.
    Call before the Estimator initializes the model."""
    entries = _as_list(entries)
    if not isinstance(entries, list) or not any(_is_spec(e) for e in entries):
        return entries
    index = _parameter_index(model)
    out = []
    for entry in entries:
        if not _is_spec(entry):
            out.append(entry)
            continue
        variable = entry if isinstance(entry, Variable) else Variable(entry)
        if variable.is_trajectory:
            raise ValueError("the Estimator estimates parameters, not trajectories")
        if variable.periods != "shared":
            raise NotImplementedError("periods='per_period' for estimated parameters is not implemented yet (#235)")
        found = _resolve(variable, index)
        if variable.components == "private":
            for component, path, target in found:
                lb, ub = _bounds(variable, component, path, target)
                out.append((component, path, variable.x0, lb, ub, "private"))
            continue
        bounds = [_bounds(variable, c, p, t) for c, p, t in found]
        if any(not (_same(b[0], bounds[0][0]) and _same(b[1], bounds[0][1])) for b in bounds[1:]):
            raise ValueError("a shared Variable's components declare different bounds: give the Variable lb and ub")
        components = [c for c, _p, _t in found]
        target = components if len(components) > 1 else components[0]
        out.append((target, found[0][1], variable.x0, bounds[0][0], bounds[0][1], "shared"))
    return out


def optimizer_variables(entries, model) -> List[Any]:
    """``variables`` with every :class:`Variable` or bare parameter as the
    Optimizer's tuple ``(component, name, lb, ub)``, one per component.
    Tuples pass through unchanged.  The Optimizer starts from the
    parameter's current value, so a Variable's ``x0`` is refused."""
    entries = _as_list(entries)
    if not isinstance(entries, list) or not any(_is_spec(e) for e in entries):
        return entries
    index = None
    out = []
    for entry in entries:
        if not _is_spec(entry):
            out.append(entry)
            continue
        variable = entry if isinstance(entry, Variable) else Variable(entry)
        if variable.x0 is not None:
            raise ValueError("the Optimizer starts from the parameter's current value: set it on the parameter, not as x0")
        if variable.is_trajectory:
            if variable.components != "private":
                raise ValueError("components= names parameters of several components; a trajectory Variable names one port")
            if variable.periods != "per_period":
                raise NotImplementedError("periods='shared' for trajectory variables is not implemented yet (#235)")
            component, name = _port_owner(model, variable.targets[0])
            out.append((component, name, variable.lb, variable.ub))
            continue
        if variable.periods != "shared":
            raise NotImplementedError("periods='per_period' for parameter variables is not implemented yet (#235)")
        if len(variable.targets) > 1 and variable.components == "shared":
            raise NotImplementedError("components='shared' for parameter variables is not implemented yet (#235)")
        index = index if index is not None else _parameter_index(model)
        for component, path, target in _resolve(variable, index):
            lb, ub = _bounds(variable, component, path, target)
            if np.ndim(lb) or np.ndim(ub):
                lo, hi = np.asarray(lb, dtype=float).reshape(-1), np.asarray(ub, dtype=float).reshape(-1)
                if lo.min() != lo.max() or hi.min() != hi.max():
                    raise ValueError(f"{component.id}.{path} declares bounds per instance; the Optimizer takes one pair: give lb and ub")
                lb, ub = float(lo[0]), float(hi[0])
            out.append((component, path, lb, ub))
    return out
