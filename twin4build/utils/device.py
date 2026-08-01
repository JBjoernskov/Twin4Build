"""Device/dtype movement for component object graphs.

``Model.to(device, dtype)`` must move every tensor a component owns:
``nn.Parameter``s and buffers (handled by ``nn.Module._apply``), the
normalization bounds that :class:`twin4build.utils.types.Parameter` carries
outside the module machinery, and the plain-tensor attributes (port tensors,
cached state-space matrices, schedule tables) that live in ordinary object
attributes.  Rather than enumerating every component class, we walk the
object graph generically: any ``torch.Tensor`` reachable through attributes
or plain containers of twin4build objects is moved.

Tensors allocated *after* the move land on the right device because
``SimulationModel.initialize`` runs under ``torch.device(model.device)`` and
the float dtype is resolved through ``tps.float_dtype()``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _move_tensor(
    t: torch.Tensor, device: torch.device, dtype: Optional[torch.dtype]
) -> torch.Tensor:
    """Move ``t`` to ``device``, converting dtype only for floating tensors."""
    if dtype is not None and t.is_floating_point():
        return t.to(device=device, dtype=dtype)
    return t.to(device=device)


def _is_twin4build_object(obj) -> bool:
    module = getattr(type(obj), "__module__", "") or ""
    return module.startswith("twin4build")


def _move_parameter_inplace(
    p: nn.Parameter, device: torch.device, dtype: Optional[torch.dtype]
) -> None:
    """Move an ``nn.Parameter`` (incl. tps.Parameter bounds) in place."""
    p.data = _move_tensor(p.data, device, dtype)
    if p.grad is not None:
        p.grad = _move_tensor(p.grad, device, dtype)
    # tps.Parameter stores its physical bounds outside the tensor data, so
    # nn.Module._apply does not see them.
    for name in ("_min_value", "_max_value"):
        bound = getattr(p, name, None)
        if isinstance(bound, torch.Tensor):
            setattr(p, name, _move_tensor(bound, device, dtype))


def _iter_attributes(obj):
    """Yield ``(container, key, value)`` for all attributes of ``obj``.

    ``container`` is either the object itself (setattr assignment) or a
    dict/list (item assignment).
    """
    d = getattr(obj, "__dict__", None)
    if d is not None:
        for name, value in list(d.items()):
            yield obj, name, value
    for klass in type(obj).__mro__:
        for name in getattr(klass, "__slots__", ()):
            if name == "__dict__" or not hasattr(obj, name):
                continue
            yield obj, name, getattr(obj, name)


def _assign(container, key, value) -> None:
    if isinstance(container, (dict, list)):
        container[key] = value
    else:
        setattr(container, key, value)


def move_object_tensors(
    root, device, dtype: Optional[torch.dtype] = None
) -> None:
    """Recursively move every tensor reachable from ``root`` to ``device``.

    Traverses attributes of twin4build objects and ``nn.Module``s plus plain
    dict/list/tuple containers.  ``nn.Parameter``s are moved in place (their
    identity must survive, they are registered in module ``_parameters``
    dicts and may be shared); plain tensors are replaced by moved copies.
    Non-floating tensors (masks, index tensors) keep their dtype.
    """
    device = torch.device(device)
    seen: set = set()
    stack = [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))

        if isinstance(obj, nn.Module):
            # Moves registered parameters/buffers of obj and all submodules
            # in place; custom attributes are handled by the walk below.
            for p in obj.parameters(recurse=True):
                _move_parameter_inplace(p, device, dtype)
            for buf_owner in obj.modules():
                for name, buf in list(buf_owner._buffers.items()):
                    if buf is not None:
                        buf_owner._buffers[name] = _move_tensor(buf, device, dtype)

        for container, key, value in _iter_attributes(obj):
            if isinstance(value, nn.Parameter):
                _move_parameter_inplace(value, device, dtype)
            elif isinstance(value, torch.Tensor):
                _assign(container, key, _move_tensor(value, device, dtype))
            elif isinstance(value, dict):
                for k, v in list(value.items()):
                    if isinstance(v, nn.Parameter):
                        _move_parameter_inplace(v, device, dtype)
                    elif isinstance(v, torch.Tensor):
                        value[k] = _move_tensor(v, device, dtype)
                    elif isinstance(v, (nn.Module,)) or _is_twin4build_object(v):
                        stack.append(v)
            elif isinstance(value, (list, tuple)):
                if isinstance(value, list):
                    for i, v in enumerate(value):
                        if isinstance(v, nn.Parameter):
                            _move_parameter_inplace(v, device, dtype)
                        elif isinstance(v, torch.Tensor):
                            value[i] = _move_tensor(v, device, dtype)
                        elif isinstance(v, nn.Module) or _is_twin4build_object(v):
                            stack.append(v)
                else:
                    # Tuples are immutable: recurse into object elements but
                    # replace the whole tuple if it holds tensors.
                    if any(isinstance(v, torch.Tensor) for v in value):
                        _assign(
                            container,
                            key,
                            tuple(
                                _move_tensor(v, device, dtype)
                                if isinstance(v, torch.Tensor)
                                else v
                                for v in value
                            ),
                        )
                    for v in value:
                        if isinstance(v, nn.Module) or _is_twin4build_object(v):
                            stack.append(v)
            elif isinstance(value, nn.Module) or _is_twin4build_object(value):
                stack.append(value)
