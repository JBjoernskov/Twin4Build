"""A measuring device's scoring mask, cut to one period, and the samples that
rest on filled inputs.

A sensor's ``scoring_mask`` says which samples of its series are scored
(``False`` = not scored).  The estimator's objective and the simulator's
error table both read it per period through :func:`period_mask`, and leave
unscored the samples :func:`filled_downstream` names: those of every scored
sensor downstream of an input step a declared fill gave a value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_ns(index) -> np.ndarray:
    """A DatetimeIndex (naive or tz-aware) as int64 nanoseconds since the epoch, UTC."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    # ``asi8`` gives the index's own unit (pandas 3 builds microsecond
    # indices from Python datetimes): compare in nanoseconds always
    return idx.as_unit("ns").asi8


def _receivers(component) -> list:
    return [
        cp.connection_point_of
        for conn in getattr(component, "connected_through", []) or []
        for cp in getattr(conn, "connects_system_at", []) or []
        if getattr(cp, "connection_point_of", None) is not None
    ]


def _downstream_scored(model, origin) -> list:
    """The ids of the scored sensors (sensors measuring a model output) the
    readings of ``origin`` reach through ``model``'s connections."""
    from twin4build.systems.sensor.sensor_system import SensorSystem

    reached, seen, todo = [], {id(origin)}, list(_receivers(origin))
    while todo:
        c = todo.pop()
        if id(c) in seen:
            continue
        seen.add(id(c))
        if isinstance(c, SensorSystem) and not c.is_leaf:
            reached.append(c.id)
        todo.extend(_receivers(c))
    return reached


def filled_downstream(model) -> dict:
    """``{scored sensor id: (n_s, n_t) bool}``: the samples of every scored
    sensor of ``model`` that rest on an input step a declared fill gave a
    value (``TimeSeriesInputSystem.filled``), after the model was initialised.

    The path from an input to what it reaches is read on the unbatched model:
    a batched model's metas join many instances, so its own wiring would reach
    nearly everything (``Model.batch_components`` keeps the source model, and
    every member of the batched model names the components it stands for).

    An input sensor with a ``fill_reach`` (sensor ids) unscores only those of
    the sensors it reaches: in a coupled building every input reaches every
    room, and the site may decide that a filled point of one room weighs on
    that room only.  One with ``fill_recovery_steps`` leaves them unscored
    that many steps after each filled step too.
    """
    from twin4build.systems.sensor.sensor_system import SensorSystem

    source = getattr(model, "_batch_source_model", None) or model
    sensors = [c for c in model.components.values() if isinstance(c, SensorSystem)]
    by_origin = {(getattr(s, "_source_component_ids", None) or (s.id,))[0]: s for s in sensors}
    out: dict = {}
    for s in sensors:
        tsi = getattr(s, "time_series_input", None)
        filled = getattr(tsi, "filled", None) if tsi is not None else None
        if filled is None or not s.is_leaf or not np.asarray(filled).any():
            continue
        origin = source.components.get((getattr(s, "_source_component_ids", None) or (s.id,))[0])
        if origin is None:
            continue
        reach = getattr(origin, "fill_reach", None)
        unscored = _with_recovery(np.asarray(filled, dtype=bool), int(getattr(origin, "fill_recovery_steps", 0) or 0))
        for rid in _downstream_scored(source, origin):
            if reach is not None and rid not in reach:
                continue
            r = by_origin.get(rid)
            if r is not None:
                out[r.id] = unscored | out.get(r.id, False)
    return out


def _with_recovery(filled: np.ndarray, steps: int) -> np.ndarray:
    """``filled`` (``(n_s, n_t)``) and the ``steps`` steps after each filled
    step: what rests on a filled input after its gap (a controller's
    integrator, say) recovers only some time after it."""
    out = filled.copy()
    for k in range(1, min(steps, filled.shape[-1] - 1) + 1):
        out[..., k:] |= filled[..., :-k]
    return out


def period_mask(mask, index, n_t: int) -> np.ndarray:
    """The device's ``scoring_mask`` for one period, ``n_t`` bools (True =
    scored).  A time-indexed ``pandas.Series`` is selected by the period's
    timestamps (a mask over a span of several periods; a time it does not
    cover is scored), a plain array is taken as the period's own steps
    (padded with True when short)."""
    if hasattr(mask, "reindex") and hasattr(mask, "index"):
        # sorted by time: a mask over periods given out of time order (a
        # winter span, then the summer before it) is a valid mask too, and a
        # binary search on an unsorted index finds nothing
        ns = to_ns(mask.index)
        order = np.argsort(ns, kind="stable")
        ns, values = ns[order], np.asarray(mask.to_numpy(), dtype=bool)[order]
        want = to_ns(index)[:n_t]
        pos = np.searchsorted(ns, want)
        pos = np.clip(pos, 0, max(len(ns) - 1, 0))
        hit = (len(ns) > 0) & (ns[pos] == want)
        keep = np.ones(len(want), dtype=bool)
        keep[hit] = values[pos[hit]]
        if len(want) < n_t:
            keep = np.concatenate([keep, np.ones(n_t - len(want), dtype=bool)])
        return keep
    keep = np.asarray(mask, dtype=bool)[:n_t]
    if keep.shape[0] < n_t:
        keep = np.concatenate([keep, np.ones(n_t - keep.shape[0], dtype=bool)])
    return keep
