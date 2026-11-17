r"""A space's mechanical air balance on its own: the surplus and the make-up flow of its supply and exhaust.

A room computes these two outputs from its flow ports alone (:mod:`~twin4build.systems.building_space.air_balance`),
not from its state or a parameter.  A world of a partitioned model that needs them from a room it does not simulate
(a storey's hallway taking the rooms' surplus) needs that room's terminals, not the room: the partition puts this
component there in the room's place (``OUTPUT_PROJECTIONS`` of the room classes,
:mod:`twin4build.model.signal_dependencies`).  Stateless and parameter-free, so its copies batch with each other.

.. math::

   \dot m_{sur} = \max(\dot m_{sup} - \dot m_{exh}, 0), \qquad \dot m_{mu} = \max(\dot m_{exh} - \dot m_{sup}, 0)

with :math:`\dot m_{sup}` the sum of the supply slots and :math:`\dot m_{exh}` the sum of the exhaust slots plus
the extract slots (what boxes draw from a hallway, :class:`StarNodeBuildingSpaceSystem`'s ``extractAirFlowRate``);
the maximum is the library's saturation (:func:`~twin4build.systems.building_space.air_balance.positive_flow`).
"""


import twin4build.core as core
from twin4build.systems.saref4syst.declarations import Input, Output
from twin4build.systems.building_space import air_balance

#: The flow ports, all vectors (one slot per stream); exhaust and extract are optional.
FLOW_PORTS = ("supplyAirFlowRate", "exhaustAirFlowRate", "extractAirFlowRate")


class AirImbalanceSystem(core.System):
    r"""The surplus and make-up flow of a space's supply and exhaust (module docstring).

    Inputs (vectors): ``supplyAirFlowRate``, ``exhaustAirFlowRate`` (optional), ``extractAirFlowRate`` (optional).
    Outputs: ``surplusAirFlowRate``, ``makeUpAirFlowRate`` [kg/s].
    """

    #: (:mod:`twin4build.model.signal_dependencies`)
    OUTPUT_DEPENDENCIES = {
        "surplusAirFlowRate": {"inputs": FLOW_PORTS},
        "makeUpAirFlowRate": {"inputs": FLOW_PORTS},
    }

    supplyAirFlowRate = Input(size="wired", doc="Supply air flow rates [kg/s], summed")
    exhaustAirFlowRate = Input(size="wired", default=0.0, doc="Exhaust air flow rates [kg/s], summed")
    extractAirFlowRate = Input(size="wired", default=0.0, doc="Extract air flow rates [kg/s], added to the exhaust")
    surplusAirFlowRate = Output(doc="Supply beyond the exhaust [kg/s]")
    makeUpAirFlowRate = Output(doc="Exhaust beyond the supply [kg/s]")

    def forward(self, x, inputs, p, step_size):
        """Stateless: the vectors summed, the extract added to the exhaust, the two imbalance flows."""
        supply = inputs["supplyAirFlowRate"].sum(dim=-1)
        exhaust = supply * 0.0
        for port in ("exhaustAirFlowRate", "extractAirFlowRate"):
            value = inputs.get(port)
            if value is not None:
                exhaust = exhaust + value.sum(dim=-1)
        surplus, make_up = air_balance.imbalance_flows(supply, exhaust)
        return x, {"surplusAirFlowRate": surplus, "makeUpAirFlowRate": make_up}

