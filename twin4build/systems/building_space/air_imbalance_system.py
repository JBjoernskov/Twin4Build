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

import datetime

import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.building_space import air_balance

#: The flow ports, all vectors (one slot per stream); exhaust and extract are optional.
FLOW_PORTS = ("supplyAirFlowRate", "exhaustAirFlowRate", "extractAirFlowRate")


class AirImbalanceSystem(core.System):
    r"""The surplus and make-up flow of a space's supply and exhaust (module docstring).

    Inputs (vectors): ``supplyAirFlowRate``, ``exhaustAirFlowRate`` (optional), ``extractAirFlowRate`` (optional).
    Outputs: ``surplusAirFlowRate``, ``makeUpAirFlowRate`` [kg/s].
    """

    PARAM_NAMES = ()

    #: (:mod:`twin4build.model.signal_dependencies`)
    OUTPUT_DEPENDENCIES = {
        "surplusAirFlowRate": {"inputs": FLOW_PORTS},
        "makeUpAirFlowRate": {"inputs": FLOW_PORTS},
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input = {
            "supplyAirFlowRate": tps.Vector(),
            "exhaustAirFlowRate": tps.Vector(optional=True),
            "extractAirFlowRate": tps.Vector(optional=True),
        }
        self.output = {"surplusAirFlowRate": tps.Scalar(0.0), "makeUpAirFlowRate": tps.Scalar(0.0)}
        self._config = {"parameters": []}
        self.INITIALIZED = False

    @property
    def config(self):
        return self._config

    def initialize(self, start_time: datetime.datetime, end_time: datetime.datetime, step_size: int, simulator=None) -> None:
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(start_time, end_time, step_size)
        n_s = len(start_time)
        for port in FLOW_PORTS:
            self.input[port].initialize(n_t=max_timesteps, n_s=n_s, n_v=self.get_n_v_from_connections(port) or 1)
        for output in self.output.values():
            output.initialize(n_t=max_timesteps, n_s=n_s)
        self.INITIALIZED = True

    def forward(self, x, inputs, params, step_size):
        """Pure algebraic map: the vectors summed, the extract added to the exhaust, the two imbalance flows."""
        supply = inputs["supplyAirFlowRate"].sum(dim=-1)
        exhaust = supply * 0.0
        for port in ("exhaustAirFlowRate", "extractAirFlowRate"):
            value = inputs.get(port)
            if value is not None:
                exhaust = exhaust + value.sum(dim=-1)
        surplus, make_up = air_balance.imbalance_flows(supply, exhaust)
        return x, {"surplusAirFlowRate": surplus, "makeUpAirFlowRate": make_up}

    def do_step(self, second_time=None, date_time=None, step_size=None, step_index=None) -> None:
        inputs = {port: self.input[port].get() for port in FLOW_PORTS}
        _, outs = self.forward(None, inputs, {}, self._scalar_step_size(step_size))
        for port, value in outs.items():
            self.output[port]._set(value, step_index)
