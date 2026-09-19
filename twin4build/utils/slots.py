"""Slot indices a component's ports are wired on.

A connection into a Vector port carries an input slot index and, from a
Vector output, an output slot index; a batched connection carries one
(sender instance, receiver instance, output slot, input slot) tuple per
pair as aligned tensors.  Components size their Vector ports from these,
so every reader must accept ints and tensors alike.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import torch


def _as_list(index: Any, n: int = 1) -> List[Any]:
    if index is None or isinstance(index, slice):
        return [index] * n
    if isinstance(index, torch.Tensor):
        values = index.reshape(-1).tolist()
        return values if len(values) != 1 or n == 1 else values * n
    return [int(index)] * n


def slot_pairs(connection_point, connection) -> List[Tuple[Any, Any, Any, Any]]:
    """``(sender_i_c, receiver_i_c, output_slot, input_slot)`` per pair of one
    connection; ``None`` where a side is scalar or unbatched (``slice`` for
    an unbatched component axis)."""
    out_v = connection_point.output_port_index.get(connection)
    in_v = connection_point.input_port_index.get(connection)
    s_ic = connection_point.output_component_index.get(connection)
    r_ic = connection_point.input_component_index.get(connection)
    n = max(
        int(v.numel()) if isinstance(v, torch.Tensor) else 1
        for v in (out_v, in_v, s_ic, r_ic)
    )
    return list(zip(_as_list(s_ic, n), _as_list(r_ic, n), _as_list(out_v, n), _as_list(in_v, n)))


def wired_slots(component, port: str) -> List[int]:
    """Every input slot index wired on ``component.input[port]``."""
    return [
        int(in_v)
        for cp in component.connects_at
        if cp.input_port == port
        for conn in cp.connects_system_through
        for _, _, _, in_v in slot_pairs(cp, conn)
        if in_v is not None and not isinstance(in_v, slice)
    ]


def wired_width(component, port: str) -> int:
    """``max slot + 1`` over the wired slots (0 when nothing is wired)."""
    return max(wired_slots(component, port), default=-1) + 1
