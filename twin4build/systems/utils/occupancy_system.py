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
from twin4build.systems.utils.smooth_saturation import smooth_saturation
from twin4build.systems.utils.time_series_input_system import TimeSeriesInputSystem


class _MassParams:
    """Lightweight container mirroring BuildingSpaceMassTorchSystem parameter
    paths (``V``, ``G_occ``, ``m_inf``) so that the estimator can share them
    via ``rgetattr(component, "mass.V")`` etc."""

    def __init__(self, V: float, G_occ: float, m_inf: float):
        self.V = tps.Parameter(
            torch.tensor(V, dtype=torch.float64), requires_grad=False
        )
        self.G_occ = tps.Parameter(
            torch.tensor(G_occ, dtype=torch.float64), requires_grad=False
        )
        self.m_inf = tps.Parameter(
            torch.tensor(m_inf, dtype=torch.float64), requires_grad=False
        )


class _DamperParams(core.System, nn.Module):
    """Internal damper model for converting position to airflow.

    Mirrors ``DamperTorchSystem`` attribute names (``a``, ``nominalAirFlowRate``)
    so that the estimator can share parameters directly::

        ([model_supply_damper, occ.supply_damper], "a", 1, 1, 10, "shared")
    """

    def __init__(self, id: str, a: float = 1.0, nominalAirFlowRate: float = 0.001):
        core.System.__init__(self, id=id)
        nn.Module.__init__(self)
        self.a = tps.Parameter(
            torch.tensor(a, dtype=torch.float64), requires_grad=False
        )
        self.nominalAirFlowRate = tps.Parameter(
            torch.tensor(nominalAirFlowRate, dtype=torch.float64), requires_grad=False
        )

    def expand_to_n_c(self, n_c: int):
        self.a = self.a.expand_to_n_c(n_c)
        self.nominalAirFlowRate = self.nominalAirFlowRate.expand_to_n_c(n_c)

    def compute_airflow(self, position: torch.Tensor) -> torch.Tensor:
        """Convert damper position (0-1) to airflow [kg/s].

        Uses the same exponential characteristic as ``DamperTorchSystem``:
        ``m = a * exp(b * u) + c``  where  ``c = -a``,
        ``b = ln((nominalAirFlowRate + a) / a)``.
        """
        a = self.a.get()
        nom = self.nominalAirFlowRate.get()
        c = -a
        b = torch.log((nom - c) / a)
        return a * torch.exp(b * position) + c


class OccupancySystem(core.System, nn.Module):
    r"""Estimate number of occupants from measured CO2 and damper positions.

    All measured inputs (indoor CO2, damper position) are read from CSV files
    so that no gradient feedback loop is created during calibration.

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
        co2_datecolumn: Date column index in the CO2 CSV.
        co2_valuecolumn: Value column index in the CO2 CSV.
        damper_filename: Path to CSV with damper position measurements.
        damper_datecolumn: Date column index in the damper CSV.
        damper_valuecolumn: Value column index in the damper CSV.
        **kwargs: Forwarded to ``core.System`` (must include ``id``).
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
        co2_datecolumn: int = 0,
        co2_valuecolumn: int = 1,
        damper_filename: Optional[str] = None,
        damper_datecolumn: int = 0,
        damper_valuecolumn: int = 1,
        **kwargs,
    ):
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
        self.co2_datecolumn = co2_datecolumn
        self.co2_valuecolumn = co2_valuecolumn
        self.damper_filename = damper_filename
        self.damper_datecolumn = damper_datecolumn
        self.damper_valuecolumn = damper_valuecolumn

        self._input = {
            "outdoorCo2Concentration": tps.Scalar(),
        }
        self._output = {"scheduleValue": tps.Scalar()}
        self._config = {
            "parameters": [
                "mass.V", "mass.G_occ", "mass.m_inf",
                "supply_damper.a", "supply_damper.nominalAirFlowRate",
                "exhaust_damper.a", "exhaust_damper.nominalAirFlowRate",
                "co2_filename", "co2_datecolumn", "co2_valuecolumn",
                "damper_filename", "damper_datecolumn", "damper_valuecolumn",
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
            f"|{self.__class__.__name__}|{self.id}|: "
            "co2_filename must be set."
        )
        assert self.damper_filename is not None, (
            f"|{self.__class__.__name__}|{self.id}|: "
            "damper_filename must be set."
        )

        self._co2_ts = TimeSeriesInputSystem(
            id=f"co2_ts_{self.id}",
            filename=self.co2_filename,
            datecolumn=self.co2_datecolumn,
            valuecolumn=self.co2_valuecolumn,
            use_spreadsheet=True,
        )
        self._co2_ts.initialize(start_time, end_time, step_size)

        self._damper_ts = TimeSeriesInputSystem(
            id=f"damper_ts_{self.id}",
            filename=self.damper_filename,
            datecolumn=self.damper_datecolumn,
            valuecolumn=self.damper_valuecolumn,
            use_spreadsheet=True,
        )
        self._damper_ts.initialize(start_time, end_time, step_size)

        self.mass.V = self.mass.V.expand_to_n_c(self.n_c)
        self.mass.G_occ = self.mass.G_occ.expand_to_n_c(self.n_c)
        self.mass.m_inf = self.mass.m_inf.expand_to_n_c(self.n_c)
        self.supply_damper.expand_to_n_c(self.n_c)
        self.exhaust_damper.expand_to_n_c(self.n_c)

        self.C_prev = None
        self.INITIALIZED = True

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        C_indoor = self._co2_ts.values[step_index]          # (n_s, 1) - measured
        damper_pos = self._damper_ts.values[step_index]      # (n_s, 1) - measured
        C_outdoor = self.input["outdoorCo2Concentration"].get()

        m_sup = self.supply_damper.compute_airflow(damper_pos)
        m_exh = self.exhaust_damper.compute_airflow(damper_pos)

        if self.C_prev is None:
            self.C_prev = C_indoor.detach().clone()
            self.output["scheduleValue"]._set(
                torch.zeros_like(C_indoor), i_t=step_index
            )
            return

        V = self.mass.V.get()
        G_occ = self.mass.G_occ.get()
        m_inf = self.mass.m_inf.get()
        air_mass = V * constants.RHO_AIR
        alpha = G_occ * (constants.M_AIR / constants.M_CO2) * 1e6

        dt = torch.tensor(step_size, dtype=torch.float64).unsqueeze(-1)

        dC = C_indoor - self.C_prev
        N_occ = (
            air_mass * dC / dt
            + (m_inf + m_exh) * self.C_prev
            - (m_inf + m_sup) * C_outdoor
        ) / alpha
        N_occ = smooth_saturation(N_occ, lower=0.0, upper=1e6, curve_start=0.1)

        self.C_prev = C_indoor.detach().clone()
        self.output["scheduleValue"]._set(N_occ, i_t=step_index)
