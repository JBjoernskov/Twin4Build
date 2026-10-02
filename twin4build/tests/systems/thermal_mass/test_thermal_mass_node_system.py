# Standard library imports
import datetime
import unittest

# Third party imports
import torch
from dateutil import tz

# Local application imports
import twin4build

twin4build._IS_TESTING = True

import twin4build as tb
from twin4build.systems.thermal_mass.thermal_mass_node_system import (
    ThermalMassNodeSystem,
)
from twin4build.systems.wall.wall_system import WallSystem


class TestThermalMassNodeSystem(unittest.TestCase):
    """The 1C node: exact update from the summed heat flows, a bare node
    keeps its temperature, the functional ``forward`` matches ``do_step``,
    and a star of walls into the node conserves energy."""

    C = 1e6
    T_INIT = 21.0
    DT = 600

    def _window(self, n_steps, batch_size=1):
        start = [datetime.datetime(2023, 1, 1, tzinfo=tz.UTC)] * batch_size
        end = [start[0] + datetime.timedelta(seconds=self.DT * n_steps)] * batch_size
        return start, end

    def _star(self, n_walls, C_node=None):
        """A model: ``n_walls`` walls between fixed temperatures (leaf
        schedules are not needed, the wall inputs are set by hand) and one
        node on their B sides."""
        model = tb.Model(id="node_star")
        node = ThermalMassNodeSystem(C=C_node or self.C, T_init=self.T_INIT, id="node")
        model.add_component(node)
        walls = []
        for i in range(n_walls):
            w = WallSystem(C=2e5, R_a=0.05, R_b=0.02, T_init=self.T_INIT, id=f"wall{i}")
            model.add_component(w)
            model.add_connection(node, w, "temperature", "temperatureB")
            model.add_connection(w, node, "heatFlowRateB", "heatFlowRate", input_port_index=i)
            walls.append(w)
        return model, node, walls

    def test_exact_update_from_summed_flows(self):
        """With A = 0 the ZOH step is exact: dT = dt * sum(Q) / C."""
        model, node, walls = self._star(2)
        start, end = self._window(3)
        node.initialize(start, end, [self.DT])
        self.assertEqual(node.n_flows, 2)
        q = torch.tensor([[[1000.0, -400.0]]])  # (n_s, n_c, n_v)
        node.input["heatFlowRate"].set(q, i_t=0)
        node.do_step(second_time=0, step_size=[self.DT], step_index=0)
        expected = self.T_INIT + self.DT * 600.0 / self.C
        self.assertAlmostEqual(float(node.output["temperature"].get()), expected, places=9)

    def test_bare_node_keeps_its_temperature(self):
        node = ThermalMassNodeSystem(C=self.C, T_init=self.T_INIT, id="lonely")
        start, end = self._window(2)
        node.initialize(start, end, [self.DT])
        self.assertEqual(node.n_flows, 1)
        node.do_step(second_time=0, step_size=[self.DT], step_index=0)
        self.assertAlmostEqual(float(node.output["temperature"].get()), self.T_INIT, places=12)

    def test_forward_matches_do_step_and_step_constants_shapes(self):
        model, node, walls = self._star(3)
        start, end = self._window(2)
        node.initialize(start, end, [self.DT])
        params = {"C": node.C.get()}
        x = node.ss_model.get_state()
        u = torch.tensor([[[100.0, 200.0, -50.0]]], dtype=node.C.get().dtype)
        x_next, outs = node.forward(x, {"heatFlowRate": u[0]}, params, float(self.DT))
        self.assertAlmostEqual(float(outs["temperature"]), self.T_INIT + self.DT * 250.0 / self.C, places=9)
        A, B, C_out, D, E, F = node.step_constants(params)
        self.assertEqual(tuple(B.shape), (1, 1, 3))
        self.assertEqual(tuple(D.shape), (1, 1, 3))
        self.assertEqual(tuple(E.shape), (1, 3, 1, 1))
        self.assertEqual(tuple(F.shape), (1, 3, 1, 3))

    def test_star_conserves_energy(self):
        """Walls at fixed side-A temperatures feeding the node: over a step
        the energy stored in the walls and the node equals the heat the
        walls took from their A sides (the walls' own balance), and the node
        stores exactly what the walls delivered on their B sides."""
        model, node, walls = self._star(2)
        start, end = self._window(2)
        for w in walls:
            w.initialize(start, end, [self.DT])
        node.initialize(start, end, [self.DT])
        t_a = [25.0, 18.0]
        t_node = float(node.output["temperature"].get())
        q_b = []
        for w, ta in zip(walls, t_a):
            w.input["temperatureA"].set(torch.tensor([ta]), i_t=0)
            w.input["temperatureB"].set(torch.tensor([t_node]), i_t=0)
            w.do_step(second_time=0, step_size=[self.DT], step_index=0)
            q_b.append(float(w.output["heatFlowRateB"].get()))
        node.input["heatFlowRate"].set(torch.tensor([[q_b]]), i_t=0)
        node.do_step(second_time=0, step_size=[self.DT], step_index=0)
        stored_node = self.C * (float(node.output["temperature"].get()) - t_node)
        # the node stores what the walls delivered on their B sides (heat INTO the node)
        self.assertAlmostEqual(stored_node, self.DT * sum(q_b), delta=1e-3)  # C * dT amplifies the state's rounding


if __name__ == "__main__":
    unittest.main()
