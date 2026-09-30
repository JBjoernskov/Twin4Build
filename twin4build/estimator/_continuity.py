r"""The jumps a multiple-shooting fit makes at its window boundaries.

With multiple shooting (``Estimator.estimate(..., multiple_shooting=...)``)
every window starts from a state the fit chooses, tied to the end of the
window before it by a continuity defect.  The fit may accept a mismatch
there when it buys a better fit of the following window: a *jump*, the
start of window :math:`p + 1` minus the end of window :math:`p`.  A
continuous simulation over all windows cannot make that jump.

Small jumps of either sign are the tolerance at work.  Jumps of one sign in
the same state at most boundaries are not: the model drifts where the
building does not, and the fit re-sets the state every window instead of
the parameters explaining it (a heat path the model lacks, a gain it
misses).  :func:`continuity_summary` ranks the states by both.

A result of such a fit carries ``continuity_jumps_instances`` (``{component
id: (P - 1, state_size)}``, ``NaN`` for states that were not variables) and
``continuity_tolerance_instances`` (``{component id: (1, state_size)}``).
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

#: The columns of :func:`continuity_summary`.
COLUMNS = [
    "component", "state", "boundaries", "mean jump", "mean |jump|", "max |jump|",
    "tolerance", "mean |jump| / tolerance", "one-sided",
]


def continuity_summary(jumps: Mapping, tolerance: Optional[Mapping] = None) -> pd.DataFrame:
    """One row per state that was a variable, the largest jumps first.

    Args:
        jumps: ``{component id: (P - 1, state_size)}`` as in a result's
            ``continuity_jumps_instances``.
        tolerance: ``{component id: (1, state_size)}`` as in a result's
            ``continuity_tolerance_instances``; ``None`` leaves the
            tolerance columns empty.

    Returns:
        A frame with :data:`COLUMNS`: ``state`` is the index in the
        component's state vector; ``mean jump`` keeps the sign (a state
        re-set downward at every boundary has a negative mean);
        ``one-sided`` is the share of boundaries whose jump has the sign of
        the mean (1: every jump points the same way; about 0.5: noise).
        Sorted by ``mean |jump| / tolerance`` when the tolerance is given,
        else by ``mean |jump|``.
    """
    rows = []
    for cid, block in jumps.items():
        values = np.asarray(block, dtype=float)
        values = values.reshape(values.shape[0], -1)
        sd = None
        if tolerance is not None and cid in tolerance:
            sd = np.asarray(tolerance[cid], dtype=float).reshape(-1)
        for k in range(values.shape[1]):
            column = values[:, k]
            column = column[np.isfinite(column)]
            if column.size == 0:
                continue
            mean = float(column.mean())
            same = float(np.mean(np.sign(column) == np.sign(mean))) if mean != 0.0 else 0.0
            tol = float(sd[k]) if sd is not None and k < sd.size and np.isfinite(sd[k]) else np.nan
            mean_abs = float(np.abs(column).mean())
            rows.append({
                "component": str(cid),
                "state": int(k),
                "boundaries": int(column.size),
                "mean jump": mean,
                "mean |jump|": mean_abs,
                "max |jump|": float(np.abs(column).max()),
                "tolerance": tol,
                "mean |jump| / tolerance": mean_abs / tol if np.isfinite(tol) and tol > 0 else np.nan,
                "one-sided": same,
            })
    table = pd.DataFrame(rows, columns=COLUMNS)
    key = "mean |jump| / tolerance" if tolerance is not None and table["mean |jump| / tolerance"].notna().any() else "mean |jump|"
    return table.sort_values(key, ascending=False, na_position="last").reset_index(drop=True)
