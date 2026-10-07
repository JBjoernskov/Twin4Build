r"""Balanced-ventilation input transform shared by the room air models.

The thermal (:class:`BuildingSpaceThermalSystem`) and CO2
(:class:`BuildingSpaceMassSystem`) room models receive a measured or
simulated supply flow :math:`\dot m_{sup}` and exhaust flow
:math:`\dot m_{exh}` from the VAV branches.  The two are never exactly equal
(one meter per side, leakage, estimation error), and a naive balance

.. math::

   \dot m_{sup} c_p T_{sup} - \dot m_{exh} c_p T_i

then carries a fictitious :math:`(\dot m_{sup} - \dot m_{exh}) c_p T_i`
term: mass that enters the room without ever leaving it (or leaves without
entering).  The room air mass is constant, so the imbalance is closed by
the envelope:

* **surplus supply** (:math:`\dot m_{sup} \ge \dot m_{exh}`): the excess
  leaves through cracks and doors *at room state*.  The whole supply flow is
  balanced by air leaving at :math:`T_i` (or :math:`C_i`):
  :math:`\dot m_{sup} c_p (T_{sup} - T_i)`.
* **deficit supply** (:math:`\dot m_{exh} > \dot m_{sup}`): the extra
  exhaust is made up by outdoor air drawn in through the envelope:
  :math:`\dot m_{sup} c_p (T_{sup} - T_i) + (\dot m_{exh} - \dot m_{sup})
  c_p (T_{out} - T_i)`.

Both cases are the same expression with the **make-up flow**

.. math::

   \dot m_{mu} = \max(\dot m_{exh} - \dot m_{sup}, 0)

in place of the raw exhaust flow: the supply term is balanced by room air
leaving, the make-up term brings outdoor air and is balanced the same way.
The room models keep their ``supplyAirFlowRate`` / ``exhaustAirFlowRate``
ports; :func:`make_up_air_flow` is applied to the exhaust slot at input
assembly time (object, functional and fused execution paths alike).

The constant infiltration :math:`\dot m_{inf}` (a parameter of both models)
is *additive* to the make-up flow, the EnergyPlus convention: it models
wind- and stack-driven exchange that exists regardless of the mechanical
imbalance, and is balanced by an equal outflow at room state.

The :math:`\max(\cdot, 0)` is the library's saturation
(:func:`~twin4build.systems.utils.smooth_saturation.clamp`), so it follows
the process-wide saturation mode.  A hard max has a corner at zero
imbalance, and a room whose exhaust follows its supply sits exactly on it:
the gradient there matches neither side, and an estimator fitting the
exhaust-to-supply ratio cannot move it.  The smooth mode (the default)
rounds the corner with a power curve that keeps a gradient at any
imbalance, at the price of a small make-up flow where the hard max is zero
(:data:`FLOW_CURVE_START` divided by :math:`\sqrt 3` at zero imbalance);
the hard mode (``saturation_mode("hard")``, the refinement stage) is the
exact max.
"""

import torch

from twin4build.systems.utils.smooth_saturation import clamp

#: Input ports read by :func:`balanced_flow_inputs`; the fused block asserts
#: they are external columns of the unit (an internal, substituted flow
#: would bypass the transform).
TRANSFORM_PORTS = ("supplyAirFlowRate", "exhaustAirFlowRate")

#: Width [kg/s] of the smooth mode's curve around zero flow (about 1 % of a
#: room's ventilation flow).
FLOW_CURVE_START = 1e-3
#: Exponent of the power curve; the curve's steepness is its inverse, so the
#: curve joins the straight part with slope 1 (continuously differentiable).
_POWER_EXP = 0.5
#: An upper bound no air flow reaches [kg/s] (the clamp is one-sided).
_UNBOUNDED = 1e6


def positive_flow(u: torch.Tensor) -> torch.Tensor:
    r""":math:`\max(u, 0)` of an air flow [kg/s] in the current saturation
    mode: exact under ``saturation_mode("hard")``; in the smooth mode a power
    curve below :data:`FLOW_CURVE_START`, continuously differentiable,
    positive everywhere (:math:`c/\sqrt 3` at zero, :math:`c\sqrt{c/2|u|}` for
    a negative :math:`u` far from it, with :math:`c` the curve start)."""
    return clamp(
        u,
        lower=0.0,
        upper=_UNBOUNDED,
        curve_start=FLOW_CURVE_START,
        steepness=1.0 / _POWER_EXP,
        curve_type="power",
        power_exp=_POWER_EXP,
    )


def make_up_air_flow(supply: torch.Tensor, exhaust: torch.Tensor) -> torch.Tensor:
    r"""Outdoor make-up flow :math:`\max(\dot m_{exh} - \dot m_{sup}, 0)`
    [kg/s] -- the exhaust in excess of the supply, drawn through the
    envelope at outdoor state (:func:`positive_flow`: exactly zero whenever
    the supply covers the exhaust in the hard saturation mode, a small flow
    in the smooth one).  Traceable under functorch/cuda graphs."""
    return positive_flow(exhaust - supply)


def balanced_flow_inputs(inputs: dict) -> dict:
    """Input transform of the room air models: replace the value on the
    ``exhaustAirFlowRate`` slot by the make-up flow.  Pure function of the
    *original* inputs; returns only the replaced entries (empty when a flow
    port is absent, e.g. a partially wired unit)."""
    if all(port in inputs for port in TRANSFORM_PORTS):
        return {
            "exhaustAirFlowRate": make_up_air_flow(
                inputs["supplyAirFlowRate"], inputs["exhaustAirFlowRate"]
            )
        }
    return {}
