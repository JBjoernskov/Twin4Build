"""The variables of an estimation or optimization problem, resolved to the
tuples the :class:`~twin4build.estimator.estimator.Estimator` and the
:class:`~twin4build.optimizer.optimizer.Optimizer` validate.

A :class:`~twin4build.utils.types.Variable` (or a bare
:class:`~twin4build.utils.types.Parameter`) names its target by the object;
the problem classes key everything on ``(component, attribute path)``.  A
parameter is found by identity among the parameters its component declares
(``_config["parameters"]`` and ``get_estimable_parameters()``, the registry
``parameters="auto"`` walks), an output port among the components'
``output`` ports.  Default bounds are the component's declared ones, as for
``parameters="auto"``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.types import Variable


def _as_variable(entry) -> Variable:
    return entry if isinstance(entry, Variable) else Variable(entry)


def _is_spec(entry) -> bool:
    return isinstance(entry, Variable) or isinstance(entry, torch.nn.Parameter)


def _declared(component) -> Dict[str, Tuple[Any, Any, Any]]:
    """``{attribute path: (x0, lb, ub)}`` the component declares estimable."""
    try:
        entries = component.get_estimable_parameters() or []
    except Exception:  # noqa: BLE001 - an unbuilt component declares nothing yet
        entries = []
    return {attr: (x0, lb, ub) for _component, attr, x0, lb, ub in entries}


def _parameter_index(model) -> Dict[int, Tuple[Any, str]]:
    """``{id(parameter): (component, attribute path)}`` over the model's
    declared parameters."""
    index: Dict[int, Tuple[Any, str]] = {}
    for component in model.components.values():
        cfg = getattr(component, "_config", None)
        paths = list(cfg.get("parameters", [])) if isinstance(cfg, dict) else []
        paths += [p for p in _declared(component) if p not in paths]
        for path in paths:
            if not isinstance(path, str):
                continue
            try:
                obj = rgetattr(component, path)
            except AttributeError:
                continue
            if isinstance(obj, torch.nn.Parameter):
                index.setdefault(id(obj), (component, path))
    return index


def _port_owner(model, port) -> Tuple[Any, str]:
    for component in model.components.values():
        for name, candidate in getattr(component, "output", {}).items():
            if candidate is port:
                return component, name
    raise ValueError("a trajectory Variable's port is no output port of the model's components")


def _resolve_parameters(variable: Variable, index) -> Tuple[List[Any], str]:
    found = []
    for target in variable.targets:
        hit = index.get(id(target))
        if hit is None:
            raise ValueError(
                "a Variable's parameter is not a declared parameter of the model's components "
                "(the components' _config['parameters'] / get_estimable_parameters())"
            )
        found.append(hit)
    paths = {path for _component, path in found}
    if len(paths) != 1:
        raise ValueError(f"the parameters of one Variable must be the same attribute of their components, got {sorted(paths)}")
    return [component for component, _path in found], paths.pop()


def _bounds(variable: Variable, component, path) -> Tuple[Any, Any]:
    if variable.lb is not None and variable.ub is not None:
        return variable.lb, variable.ub
    declared = _declared(component).get(path)
    if declared is None:
        raise ValueError(f"{component.id}.{path} declares no bounds: give the Variable lb and ub")
    return (variable.lb if variable.lb is not None else declared[1], variable.ub if variable.ub is not None else declared[2])


def estimator_parameters(entries, model) -> List[Any]:
    """``parameters`` with every :class:`Variable` or bare parameter as the
    Estimator's tuple ``(component(s), attr, x0, lb, ub, components)``;
    tuples pass through unchanged."""
    if not isinstance(entries, list) or not any(_is_spec(e) for e in entries):
        return entries
    index = _parameter_index(model)
    out = []
    for entry in entries:
        if not _is_spec(entry):
            out.append(entry)
            continue
        variable = _as_variable(entry)
        if variable.is_trajectory:
            raise ValueError("the Estimator estimates parameters, not trajectories")
        if variable.periods != "shared":
            raise NotImplementedError("periods='per_period' for estimated parameters is not implemented yet (#235)")
        components, path = _resolve_parameters(variable, index)
        lb, ub = _bounds(variable, components[0], path)
        target = components if len(components) > 1 else components[0]
        out.append((target, path, variable.x0, lb, ub, variable.components))
    return out


def optimizer_variables(entries, model) -> List[Any]:
    """``variables`` with every :class:`Variable` or bare parameter as the
    Optimizer's tuple ``(component, name, lb, ub)`` (one per component);
    tuples pass through unchanged."""
    if not isinstance(entries, list) or not any(_is_spec(e) for e in entries):
        return entries
    index = None
    out = []
    for entry in entries:
        if not _is_spec(entry):
            out.append(entry)
            continue
        variable = _as_variable(entry)
        if variable.is_trajectory:
            if variable.periods != "per_period":
                raise NotImplementedError("periods='shared' for trajectory variables is not implemented yet (#235)")
            component, name = _port_owner(model, variable.targets[0])
            out.append((component, name, variable.lb, variable.ub))
            continue
        if variable.periods != "shared":
            raise NotImplementedError("periods='per_period' for parameter variables is not implemented yet (#235)")
        index = index if index is not None else _parameter_index(model)
        components, path = _resolve_parameters(variable, index)
        if len(components) > 1 and variable.components == "shared":
            raise NotImplementedError("components='shared' for parameter variables is not implemented yet (#235)")
        for component in components:
            lb, ub = _bounds(variable, component, path)
            out.append((component, path, lb, ub))
    return out
