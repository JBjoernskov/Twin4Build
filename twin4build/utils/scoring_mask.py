"""A measuring device's scoring mask, cut to one period.

A sensor's ``scoring_mask`` says which samples of its series are scored
(``False`` = not scored).  The estimator's objective and the simulator's
error table both read it per period through :func:`period_mask`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_ns(index) -> np.ndarray:
    """A DatetimeIndex (naive or tz-aware) as int64 nanoseconds since the epoch, UTC."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.asi8


def period_mask(mask, index, n_t: int) -> np.ndarray:
    """The device's ``scoring_mask`` for one period, ``n_t`` bools (True =
    scored).  A time-indexed ``pandas.Series`` is selected by the period's
    timestamps (a mask over a span of several periods; a time it does not
    cover is scored), a plain array is taken as the period's own steps
    (padded with True when short)."""
    if hasattr(mask, "reindex") and hasattr(mask, "index"):
        ns = to_ns(mask.index)
        want = to_ns(index)[:n_t]
        pos = np.searchsorted(ns, want)
        pos = np.clip(pos, 0, max(len(ns) - 1, 0))
        hit = (len(ns) > 0) & (ns[pos] == want)
        keep = np.ones(len(want), dtype=bool)
        keep[hit] = np.asarray(mask.to_numpy(), dtype=bool)[pos[hit]]
        if len(want) < n_t:
            keep = np.concatenate([keep, np.ones(n_t - len(want), dtype=bool)])
        return keep
    keep = np.asarray(mask, dtype=bool)[:n_t]
    if keep.shape[0] < n_t:
        keep = np.concatenate([keep, np.ones(n_t - keep.shape[0], dtype=bool)])
    return keep
