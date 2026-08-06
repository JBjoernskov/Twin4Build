# Standard library imports
import datetime
from typing import List, Optional

# Third party imports
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.constants as constants
import twin4build.utils.types as tps
from twin4build.systems.utils.smooth_saturation import clamp
from twin4build.systems.utils.time_series_input_system import TimeSeriesInputSystem


class _MassParams:
    """Lightweight container mirroring BuildingSpaceMassSystem parameter
    paths (``V``, ``G_occ``, ``m_inf``) so that the estimator can share them
    via ``rgetattr(component, "mass.V")`` etc."""

    def __init__(self, V: float, G_occ: float, m_inf: float):
        self.V = tps.Parameter(
            torch.tensor(V, dtype=tps.float_dtype()), requires_grad=False
        )
        self.G_occ = tps.Parameter(
            torch.tensor(G_occ, dtype=tps.float_dtype()), requires_grad=False
        )
        self.m_inf = tps.Parameter(
            torch.tensor(m_inf, dtype=tps.float_dtype()), requires_grad=False
        )


class _DamperParams(core.System, nn.Module):
    """Internal damper model for converting position to airflow.

    Mirrors ``DamperSystem`` attribute names (``a``, ``nominalAirFlowRate``)
    so that the estimator can share parameters directly::

        ([model_supply_damper, occ.supply_damper], "a", 1, 1, 10, "shared")
    """

    def __init__(self, id: str, a: float = 1.0, nominalAirFlowRate: float = 0.001):
        core.System.__init__(self, id=id)
        nn.Module.__init__(self)
        # Scalings MUST match ``DamperSystem`` (``a`` is log-scaled
        # there): the object-graph estimation path denormalizes each member
        # of a "shared" group with its OWN parameter scaling, so a scaling
        # mismatch silently gives the two members different physical values
        # for the same normalized theta (and different gradients).
        self.a = tps.Parameter(
            torch.tensor(a, dtype=tps.float_dtype()), requires_grad=False, scaling="log"
        )
        self.nominalAirFlowRate = tps.Parameter(
            torch.tensor(nominalAirFlowRate, dtype=tps.float_dtype()), requires_grad=False
        )

    def expand_to_n_c(self, n_c: int):
        self.a = self.a.expand_to_n_c(n_c)
        self.nominalAirFlowRate = self.nominalAirFlowRate.expand_to_n_c(n_c)

    def compute_airflow(self, position: torch.Tensor) -> torch.Tensor:
        """Convert damper position (0-1) to airflow [kg/s].

        Uses the same exponential characteristic as ``DamperSystem``:
        ``m = a * exp(b * u) + c``  where  ``c = -a``,
        ``b = ln((nominalAirFlowRate + a) / a)``.

        Delegates to :meth:`OccupancySystem._airflow` (single source of truth
        with the pure ``forward``).
        """
        return OccupancySystem._airflow(
            self.a.get(), self.nominalAirFlowRate.get(), position
        )


class OccupancySystem(core.System, nn.Module):
    r"""Estimate number of occupants from measured CO2 and damper positions.

    All measured inputs (indoor CO2, damper position) are read from CSV files
    so that no gradient feedback loop is created during calibration.

    The per-step math lives in the pure :meth:`forward` (``do_step`` is a thin
    port-I/O wrapper), so the component is composable by ``Simulator.compose``:
    the measured data is published on unconnected input ports
    (``indoorCo2Measured``, ``previousIndoorCo2Measured``,
    ``damperPositionMeasured``) that the fast paths capture per step.  At the
    first step the previous CO2 sample equals the current one (``dC = 0``), so
    the initial occupancy comes from the static balance.

    Internal ``supply_damper`` and ``exhaust_damper`` convert measured damper
    positions to airflows.  Their parameters (``a``, ``nominalAirFlowRate``)
    can be shared with the corresponding model dampers via the estimator::

        ([model_supply_damper, occ.supply_damper], "a", 1, 1, 10, "shared")

    Args:
        V: Room volume [m³].
        G_occ: CO2 generation rate per occupant [kg_CO2/s].
        m_inf: Infiltration mass flow rate [kg/s].
        supply_damper_a: Shape parameter for supply damper.
        supply_damper_nominalAirFlowRate: Nominal air flow [kg/s] for supply damper.
        exhaust_damper_a: Shape parameter for exhaust damper.
        exhaust_damper_nominalAirFlowRate: Nominal air flow [kg/s] for exhaust damper.
        co2_filename: Path to CSV with indoor CO2 measurements.
        co2_date_column: Date column index in the CO2 CSV.
        co2_value_column: Value column index in the CO2 CSV.
        damper_filename: Path to CSV with damper position measurements.
        damper_date_column: Date column index in the damper CSV.
        damper_value_column: Value column index in the damper CSV.
        **kwargs: Forwarded to ``core.System`` (must include ``id``).
            Also accepts deprecated ``co2_datecolumn`` / ``co2_valuecolumn`` /
            ``damper_datecolumn`` / ``damper_valuecolumn`` until 2.1.
    """

    def __init__(
        self,
        V: float = 100,
        G_occ: float = 5e-6,
        m_inf: float = 0.001,
        supply_damper_a: float = 1.0,
        supply_damper_nominalAirFlowRate: float = 0.001,
        exhaust_damper_a: float = 1.0,
        exhaust_damper_nominalAirFlowRate: float = 0.001,
        co2_filename: Optional[str] = None,
        co2_date_column: int = 0,
        co2_value_column: int = 1,
        damper_filename: Optional[str] = None,
        damper_date_column: int = 0,
        damper_value_column: int = 1,
        **kwargs,
    ):
        from twin4build.utils.deprecation import deprecate_args

        legacy = deprecate_args(
            [
                "co2_datecolumn",
                "co2_valuecolumn",
                "damper_datecolumn",
                "damper_valuecolumn",
            ],
            [
                "co2_date_column",
                "co2_value_column",
                "damper_date_column",
                "damper_value_column",
            ],
            [None, None, None, None],
            kwargs,
        )
        co2_date_column = legacy.get("co2_date_column", co2_date_column)
        co2_value_column = legacy.get("co2_value_column", co2_value_column)
        damper_date_column = legacy.get("damper_date_column", damper_date_column)
        damper_value_column = legacy.get("damper_value_column", damper_value_column)

        _id = kwargs.get("id", "occupancy")
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        self.mass = _MassParams(V=V, G_occ=G_occ, m_inf=m_inf)
        self.supply_damper = _DamperParams(
            id=f"{_id}_supply_damper",
            a=supply_damper_a,
            nominalAirFlowRate=supply_damper_nominalAirFlowRate,
        )
        self.exhaust_damper = _DamperParams(
            id=f"{_id}_exhaust_damper",
            a=exhaust_damper_a,
            nominalAirFlowRate=exhaust_damper_nominalAirFlowRate,
        )

        self.co2_filename = co2_filename
        self.co2_date_column = co2_date_column
        self.co2_value_column = co2_value_column
        self.damper_filename = damper_filename
        self.damper_date_column = damper_date_column
        self.damper_value_column = damper_value_column
        # Deprecated attribute aliases (removed in 2.1)
        self.co2_datecolumn = co2_date_column
        self.co2_valuecolumn = co2_value_column
        self.damper_datecolumn = damper_date_column
        self.damper_valuecolumn = damper_value_column

        self._input = {
            "outdoorCo2Concentration": tps.Scalar(),
            # Measured-data ports: NOT connected to any producer.  ``do_step``
            # publishes the CSV samples here each step so that the composed
            # fast paths (Simulator.compose) can capture them per step like
            # any exogenous signal -- freezing them is exact because they are
            # measured data, independent of any estimated parameter.
            "indoorCo2Measured": tps.Scalar(),
            "previousIndoorCo2Measured": tps.Scalar(),
            "damperPositionMeasured": tps.Scalar(),
        }
        self._output = {"scheduleValue": tps.Scalar()}
        self._config = {
            "parameters": [
                "mass.V",
                "mass.G_occ",
                "mass.m_inf",
                "supply_damper.a",
                "supply_damper.nominalAirFlowRate",
                "exhaust_damper.a",
                "exhaust_damper.nominalAirFlowRate",
                "co2_filename",
                "co2_date_column",
                "co2_value_column",
                "damper_filename",
                "damper_date_column",
                "damper_value_column",
            ]
        }
        self.INITIALIZED = False

    @property
    def config(self):
        return self._config

    def initialize(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: List[int],
    ) -> None:
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)

        for inp in self.input.values():
            inp.initialize(n_t=max_timesteps, n_s=batch_size)
        for out in self.output.values():
            out.initialize(n_t=max_timesteps, n_s=batch_size)

        assert self.co2_filename is not None, (
            f"|{self.__class__.__name__}|{self.id}|: " "co2_filename must be set."
        )
        assert self.damper_filename is not None, (
            f"|{self.__class__.__name__}|{self.id}|: " "damper_filename must be set."
        )

        self._co2_ts = TimeSeriesInputSystem(
            id=f"co2_ts_{self.id}",
            filename=self.co2_filename,
            date_column=self.co2_date_column,
            value_column=self.co2_value_column,
            use_spreadsheet=True,
        )
        self._co2_ts.initialize(start_time, end_time, step_size)

        self._damper_ts = TimeSeriesInputSystem(
            id=f"damper_ts_{self.id}",
            filename=self.damper_filename,
            date_column=self.damper_date_column,
            value_column=self.damper_value_column,
            use_spreadsheet=True,
        )
        self._damper_ts.initialize(start_time, end_time, step_size)

        self.mass.V = self.mass.V.expand_to_n_c(self.n_c)
        self.mass.G_occ = self.mass.G_occ.expand_to_n_c(self.n_c)
        self.mass.m_inf = self.mass.m_inf.expand_to_n_c(self.n_c)
        self.supply_damper.expand_to_n_c(self.n_c)
        self.exhaust_damper.expand_to_n_c(self.n_c)

        self.INITIALIZED = True

    PARAM_NAMES = (
        "mass.V",
        "mass.G_occ",
        "mass.m_inf",
        "supply_damper.a",
        "supply_damper.nominalAirFlowRate",
        "exhaust_damper.a",
        "exhaust_damper.nominalAirFlowRate",
    )

    @staticmethod
    def _airflow(
        a: torch.Tensor, nominal: torch.Tensor, position: torch.Tensor
    ) -> torch.Tensor:
        """Damper position (0-1) -> airflow [kg/s]; same exponential
        characteristic as ``DamperSystem`` (``_DamperParams.compute_airflow``
        expressed on explicit parameter tensors so ``forward`` stays pure)."""
        c = -a
        b = torch.log((nominal - c) / a)
        return a * torch.exp(b * position) + c

    def forward(self, x, inputs, params, sample_time):
        """Pure one-step occupancy estimate (functorch-safe, stateless).

        Inverts the zone CO2 balance: all data enters through ``inputs``
        (the measured-data ports plus ``outdoorCo2Concentration``), all
        estimable parameters through ``params``, so the composed fast paths
        thread theta gradients exactly.
        """
        C_indoor = inputs["indoorCo2Measured"]
        C_prev = inputs["previousIndoorCo2Measured"]
        damper_pos = inputs["damperPositionMeasured"]
        C_outdoor = inputs["outdoorCo2Concentration"]

        m_sup = self._airflow(
            params["supply_damper.a"],
            params["supply_damper.nominalAirFlowRate"],
            damper_pos,
        )
        m_exh = self._airflow(
            params["exhaust_damper.a"],
            params["exhaust_damper.nominalAirFlowRate"],
            damper_pos,
        )

        air_mass = params["mass.V"] * constants.RHO_AIR
        alpha = params["mass.G_occ"] * (constants.M_AIR / constants.M_CO2) * 1e6
        m_inf = params["mass.m_inf"]

        dC = C_indoor - C_prev
        N_occ = (
            air_mass * dC / sample_time
            + (m_inf + m_exh) * C_prev
            - (m_inf + m_sup) * C_outdoor
        ) / alpha
        N_occ = clamp(N_occ, lower=0.0, upper=1e6)
        return x, {"scheduleValue": N_occ}

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        C_indoor = self._co2_ts.values[step_index]  # (n_s, 1) - measured
        C_prev = (
            self._co2_ts.values[step_index - 1] if step_index > 0 else C_indoor
        )
        damper_pos = self._damper_ts.values[step_index]  # (n_s, 1) - measured

        # Publish the data samples on the (unconnected) measured-data input
        # ports: the composed fast paths capture input-port histories per
        # step, so this makes the data visible to Simulator.compose.
        self.input["indoorCo2Measured"]._set(C_indoor, i_t=step_index)
        self.input["previousIndoorCo2Measured"]._set(C_prev, i_t=step_index)
        self.input["damperPositionMeasured"]._set(damper_pos, i_t=step_index)

        inputs = {
            "indoorCo2Measured": C_indoor,
            "previousIndoorCo2Measured": C_prev,
            "damperPositionMeasured": damper_pos,
            "outdoorCo2Concentration": self.input["outdoorCo2Concentration"].get(),
        }
        _, outs = self.forward(
            None, inputs, self._forward_params(), self._scalar_sample_time(step_size)
        )
        self.output["scheduleValue"]._set(outs["scheduleValue"], i_t=step_index)
