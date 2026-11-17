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

The constant infiltration :math:`\dot m_{inf}` (one parameter shared by both
models of a :class:`BuildingSpaceSystem`) is *additive* to the make-up flow,
the EnergyPlus convention: it models wind- and stack-driven exchange that
exists regardless of the mechanical imbalance, always arrives at outdoor
state (also when the make-up stream comes from a neighbouring zone), and is
balanced by an equal outflow at room state.

The room publishes the two imbalance flows as outputs, so a neighbouring
zone can take them: the **surplus flow**
:math:`\dot m_{sur} = \max(\dot m_{sup} - \dot m_{exh}, 0)`, the supply the
exhaust does not take, which leaves the room at room state, and the make-up
flow.  :func:`surplus_air_flow` computes it as
:math:`\dot m_{sup} - \dot m_{exh} + \dot m_{mu}`, so the room's mechanical
streams balance exactly in either saturation mode:
:math:`\dot m_{sup} + \dot m_{mu} = \dot m_{exh} + \dot m_{sur}`.

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

from typing import Optional, Tuple

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


def surplus_air_flow(supply: torch.Tensor, exhaust: torch.Tensor) -> torch.Tensor:
    r"""Surplus flow :math:`\max(\dot m_{sup} - \dot m_{exh}, 0)` [kg/s] -- the
    supply the exhaust does not take, leaving the room at room state.  Computed
    as :math:`\dot m_{sup} - \dot m_{exh} + \dot m_{mu}` (:func:`make_up_air_flow`),
    the exact max in the hard saturation mode and the flow that closes the
    room's mass balance in the smooth one."""
    return supply - exhaust + make_up_air_flow(supply, exhaust)


#: The envelope's inflow at outdoor state [kg/s] (an envelope-leakage
#: component's share for the room): outdoor air that enters the room and
#: leaves it again at room state, a time-varying infiltration (the EnergyPlus
#: convention: additive to the make-up flow).  It is no part of the room's
#: mechanical balance: the surplus and make-up flows the room publishes, which
#: the extract boxes follow, stay those of its supply and exhaust (28 Oct 2026:
#: counted as supply, it travelled up the surplus chain and the boxes drew it).
#: Unwired it is zero and the balance is the one above, bit for bit.
INFILTRATION_PORT = "infiltrationAirFlowRate"


def mixed_state(flow_a: torch.Tensor, state_a: torch.Tensor, flow_b: torch.Tensor, state_b: torch.Tensor) -> torch.Tensor:
    """The flow-weighted state (temperature, concentration) of two streams, written as the first stream's state
    plus the second's share of the difference: exactly ``state_a`` when ``flow_b`` is zero (an unconnected inflow
    leaves the model bit for bit), and ``state_a`` where no air flows at all (it multiplies a zero flow)."""
    total = (flow_a + flow_b).clamp_min(1e-12)
    return state_a + (flow_b / total) * (state_b - state_a)


def imbalance_flows(supply: torch.Tensor, exhaust: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(surplus, make-up)`` of a room's mechanical balance (the envelope inflow is no part of it)."""
    return surplus_air_flow(supply, exhaust), make_up_air_flow(supply, exhaust)


def balanced_flow_inputs(inputs: dict, state_port: Optional[str] = None, outdoor_port: Optional[str] = None) -> dict:
    """Input transform of the room air models: replace the value on the
    ``exhaustAirFlowRate`` slot by the make-up flow.  Pure function of the
    *original* inputs; returns only the replaced entries (empty when a flow
    port is absent, e.g. a partially wired unit).

    With an envelope inflow (:data:`INFILTRATION_PORT`) the slot carries the
    make-up flow plus the inflow, at their flow-weighted state: ``state_port``
    (the make-up stream's temperature or CO2) mixed with ``outdoor_port`` (the
    inflow's).  A model whose make-up stream is outdoor air ignores
    ``state_port``'s slot: the mix is outdoor air then."""
    if not all(port in inputs for port in TRANSFORM_PORTS):
        return {}
    supply, exhaust = inputs["supplyAirFlowRate"], inputs["exhaustAirFlowRate"]
    inflow = inputs.get(INFILTRATION_PORT)
    if inflow is None:
        return {"exhaustAirFlowRate": make_up_air_flow(supply, exhaust)}
    make_up = make_up_air_flow(supply, exhaust)
    out = {"exhaustAirFlowRate": make_up + inflow}
    if state_port in inputs and outdoor_port in inputs:
        out[state_port] = mixed_state(make_up, inputs[state_port], inflow, inputs[outdoor_port])
    return out
