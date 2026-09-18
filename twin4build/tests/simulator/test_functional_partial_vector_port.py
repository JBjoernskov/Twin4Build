"""A vector input port with a gap slot must not break functional simulation.

The AHU sizes its per-branch vector ports from the highest connection index
(``get_n_v_from_connections``).  A branch whose damper has no controller --
a translated BMS model where one room's loop did not match -- therefore
leaves an unconnected slot between connected ones.  The exogenous recording
used to ask "does this PORT have a source?" for such a slot, found the other
branches' controllers, and refused to isolate it (``cannot safely isolate
exogenous input ...supplyDamperPosition``); the question is per SLOT.
"""

# Standard library imports
import datetime
import unittest

# Third party imports
import torch
from dateutil import tz

# Local application imports
import twin4build as tb
from twin4build.systems.air_handling_unit.air_handling_unit_system import (
    AirHandlingUnitSystem,
)
from twin4build.systems.building_space.building_space_system import BuildingSpaceSystem
from twin4build.systems.schedule.schedule_system import ScheduleSystem

tb._IS_TESTING = True


def _schedule(id_, value):
    return ScheduleSystem(
        id=id_,
        weekday_ruleset={
            "ruleset_start_minute": [],
            "ruleset_end_minute": [],
            "ruleset_start_hour": [],
            "ruleset_end_hour": [],
            "ruleset_value": [],
            "ruleset_default_value": value,
        },
    )


class TestPartiallyConnectedVectorPort(unittest.TestCase):
    def test_gap_slot_keeps_its_initial_value(self):
        model = tb.Model(id="functional_partial_vector_port")
        ahu = AirHandlingUnitSystem(
            id="ahu",
            damper_kwargs={"a": 1.0, "nominalAirFlowRate": 0.5},
            heat_recovery_kwargs={
                "eps_75_h": 0.7, "eps_100_h": 0.75, "eps_75_c": 0.65, "eps_100_c": 0.7,
                "primaryAirFlowRateMax": 1.0, "secondaryAirFlowRateMax": 1.0,
            },
            supply_fan_kwargs={
                "nominalPowerRate": 1000.0, "nominalAirFlowRate": 1,
                "c1": 0, "c2": 0.2, "c3": 0.8, "c4": 0, "f_total": 1.0,
            },
            exhaust_fan_kwargs={
                "nominalPowerRate": 1000.0, "nominalAirFlowRate": 1,
                "c1": 0, "c2": 0.2, "c3": 0.8, "c4": 0, "f_total": 1.0,
            },
            n_branches=3,
        )
        # Branches 0 and 2 have a damper command; branch 1 has none.
        for slot, value in ((0, 0.8), (2, 0.4)):
            damper = _schedule(f"damper_{slot}", value)
            model.add_connection(damper, ahu, "scheduleValue", "supplyDamperPosition", input_port_index=slot)
            model.add_connection(damper, ahu, "scheduleValue", "exhaustDamperPosition", input_port_index=slot)
        model.add_connection(_schedule("t_sup", 18.0), ahu, "scheduleValue", "supplyAirTemperatureSetpoint")
        model.add_connection(_schedule("t_out", 5.0), ahu, "scheduleValue", "outdoorAirTemperature")
        # One stateful consumer on branch 0, so there is something to compose.
        room = BuildingSpaceSystem(
            id="room", C_wall=1e6, C_air=1e4, C_boundary=5e5, R_out=0.01, R_in=0.02, R_boundary=0.03,
            f_wall=0.5, f_air=0.3, Q_occ_gain=100.0, CO2_occ_gain=0.004, CO2_start=400.0, airVolume=100.0,
            T_wall_start=20.0, T_air_start=20.0, T_int_start=20.0, T_boundary_start=18.0,
        )
        model.add_connection(ahu, room, "supplyAirFlowRate", "supplyAirFlowRate", output_port_index=0)
        model.add_connection(ahu, room, "supplyAirTemperature", "supplyAirTemperature")
        model.add_connection(_schedule("t_out_room", 5.0), room, "scheduleValue", "outdoorTemperature")
        model.add_connection(_schedule("people", 0.0), room, "scheduleValue", "numberOfPeople")
        model.add_connection(_schedule("sun", 0.0), room, "scheduleValue", "globalIrradiation")
        model.add_connection(_schedule("gain", 0.0), room, "scheduleValue", "heatGain")
        model.load()

        start = datetime.datetime(2023, 1, 1, tzinfo=tz.UTC)
        kwargs = dict(start_time=start, end_time=start + datetime.timedelta(hours=1), step_size=600, show_progress_bar=False)

        # Reference: object-graph execution.
        tb.Simulator(model, execution_mode="object").simulate(**kwargs)
        flow_object = ahu.output["supplyAirFlowRate"]._history.detach().clone()

        tb.Simulator(model, execution_mode="functional").simulate(**kwargs)
        positions = ahu.input["supplyDamperPosition"]._history
        self.assertEqual(positions.shape[-1], 3)
        torch.testing.assert_close(positions[..., 0], torch.full_like(positions[..., 0], 0.8))
        torch.testing.assert_close(positions[..., 2], torch.full_like(positions[..., 2], 0.4))
        # The gap slot holds the port's initialised value in both engines.
        torch.testing.assert_close(positions[..., 1], torch.zeros_like(positions[..., 1]))
        torch.testing.assert_close(ahu.output["supplyAirFlowRate"]._history, flow_object)


if __name__ == "__main__":
    unittest.main()
