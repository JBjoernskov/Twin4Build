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
"""

import torch

#: Input ports read by :func:`balanced_flow_inputs`; the fused block asserts
#: they are external columns of the unit (an internal, substituted flow
#: would bypass the transform).
TRANSFORM_PORTS = ("supplyAirFlowRate", "exhaustAirFlowRate")


def make_up_air_flow(supply: torch.Tensor, exhaust: torch.Tensor) -> torch.Tensor:
    r"""Outdoor make-up flow :math:`\max(\dot m_{exh} - \dot m_{sup}, 0)`
    [kg/s] -- the exhaust in excess of the supply, drawn through the
    envelope at outdoor state.  Zero whenever the supply covers the exhaust.
    Differentiable (piecewise linear), traceable under functorch/cuda graphs."""
    return torch.clamp(exhaust - supply, min=0.0)


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
