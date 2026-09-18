"""The occupancy inversion must be the exact inverse of the CO2 mass model.

``OccupancySystem.forward`` books the people that explain a measured CO2
step; ``BuildingSpaceMassSystem.forward`` is the ZOH of the same ODE via
``_discretize_onestep``.  A known occupancy pushed through the forward
model and inverted again must come back -- at AHU sample times
(``λ Δt ~ O(1)``), when supply and exhaust differ, and under both
``torch.matrix_exp`` (eager) and ``_expm_ss`` (transform mode).
"""

# Standard library imports
import inspect
import unittest

# Third party imports
import torch

# Local application imports
import twin4build
import twin4build as tb
import twin4build.utils.constants as constants
from twin4build.simulator.simulator import _has_triton
from twin4build.systems.building_space.building_space_mass_system import (
    ACTIVE_INPUT_SLICE,
    ACTIVE_INPUT_SLOTS,
    N_INPUTS,
    OCCUPANCY_SLOT,
    mass_matrices,
)
from twin4build.systems.utils.discrete_statespace_system import (
    _discretize_onestep,
    effective_matrices,
)

twin4build._IS_TESTING = True

DT = 1.0
DT_AHU = 600.0
V, G_OCC, M_INF = 120.0, 5e-6, 0.004
C_OUT = 400.0


def _t(v):
    return torch.tensor([float(v)], dtype=torch.float64)


def _invert_raw(m_sup, m_exh, c_prev, c_now, dt=DT, transform_mode=None):
    """``invert_zoh_occupancy`` on the zone's own parameter / input contract."""
    return tb.OccupancySystem.invert_zoh_occupancy(
        {"V": _t(V), "G_occ": _t(G_OCC), "m_inf": _t(M_INF)},
        {
            "supplyAirFlowRate": _t(m_sup),
            "exhaustAirFlowRate": _t(m_exh),
            "outdoorCO2": _t(C_OUT),
        },
        _t(c_prev),
        _t(c_now),
        dt,
        n_c=1,
        transform_mode=transform_mode,
    )


def _invert(m_sup, m_exh, c_prev, c_now, dt=DT, transform_mode=None):
    occ = tb.OccupancySystem(
        V=V, G_occ=G_OCC, m_inf=M_INF,
        supply_damper_nominalAirFlowRate=m_sup,
        exhaust_damper_nominalAirFlowRate=m_exh,
        id="occ",
    )
    occ._co2_from_sensor = False
    params = {
        "mass.V": _t(V), "mass.G_occ": _t(G_OCC), "mass.m_inf": _t(M_INF),
        "supply_damper.a": _t(1.0), "supply_damper.nominalAirFlowRate": _t(m_sup),
        "exhaust_damper.a": _t(1.0), "exhaust_damper.nominalAirFlowRate": _t(m_exh),
    }
    inputs = {
        "indoorCo2Measured": _t(c_now),
        "previousIndoorCo2Measured": _t(c_prev),
        "damperPositionMeasured": _t(1.0),  # fully open: flow = nominal
        "outdoorCo2Concentration": _t(C_OUT),
    }
    _, out = occ.forward(
        torch.zeros((1, 0), dtype=torch.float64),
        inputs,
        params,
        dt,
        transform_mode=transform_mode,
    )
    return float(out["scheduleValue"])


def _forward_step(m_sup, m_exh, c_prev, n_people, dt=DT, transform_mode=None):
    room = tb.BuildingSpaceMassSystem(V=V, G_occ=G_OCC, m_inf=M_INF, id="room")
    room.n_c = 1
    params = {"V": _t(V), "G_occ": _t(G_OCC), "m_inf": _t(M_INF)}
    inputs = {
        "supplyAirFlowRate": _t(m_sup), "exhaustAirFlowRate": _t(m_exh),
        "outdoorCO2": _t(C_OUT), "numberOfPeople": _t(n_people),
    }
    x0 = torch.full((1, 1), c_prev, dtype=torch.float64)
    x1, _ = room.forward(x0, inputs, params, dt, transform_mode=transform_mode)
    return float(x1[0, 0])


def _euler_invert(m_sup, m_exh, c_prev, c_now, dt):
    m_mu = max(m_exh - m_sup, 0.0)
    alpha = G_OCC * (constants.M_AIR / constants.M_CO2) * 1e6
    return (
        constants.RHO_AIR * V * (c_now - c_prev) / dt
        + (M_INF + m_sup + m_mu) * (c_prev - C_OUT)
    ) / alpha


class TestInversionRoundTrip(unittest.TestCase):
    def _round_trip(
        self, m_sup, m_exh, n_people=12.0, c_prev=900.0, dt=DT, transform_mode=None
    ):
        c_now = _forward_step(
            m_sup, m_exh, c_prev, n_people, dt=dt, transform_mode=transform_mode
        )
        got = _invert(
            m_sup, m_exh, c_prev, c_now, dt=dt, transform_mode=transform_mode
        )
        self.assertAlmostEqual(got, n_people, places=5, msg=(m_sup, m_exh, dt, got))

    def test_balanced(self):
        self._round_trip(0.10, 0.10)

    def test_surplus_supply(self):
        self._round_trip(0.10, 0.03)

    def test_exhaust_deficit(self):
        self._round_trip(0.10, 0.18)

    def test_ahu_step(self):
        # λh ≈ m_tot Δt / (ρ V) ≈ 0.4 at 10 min: Euler is several percent off.
        self._round_trip(0.10, 0.10, dt=DT_AHU)
        self._round_trip(0.10, 0.18, dt=DT_AHU)

    def test_transform_mode_matches_expm_ss(self):
        # Compiled / vmap path uses _expm_ss (_expm_ss_fused under inductor).
        self._round_trip(0.10, 0.10, dt=DT_AHU, transform_mode=True)
        self._round_trip(0.10, 0.18, dt=DT_AHU, transform_mode=True)

    def test_eager_and_transform_mode_values_agree(self):
        n_eager = _invert_raw(0.10, 0.18, 880.0, 910.0, DT_AHU, transform_mode=False)
        n_xfm = _invert_raw(0.10, 0.18, 880.0, 910.0, DT_AHU, transform_mode=True)
        torch.testing.assert_close(n_eager, n_xfm, rtol=1e-10, atol=1e-12)

    def test_inverts_the_zones_own_discretization(self):
        """The inverse uses the zone's matrices, not an equivalent rewrite.

        ``Bd``'s occupancy column, rebuilt here straight from
        ``mass_matrices`` + ``_discretize_onestep``, is exactly the divisor
        the inversion applies, so a change to the zone's matrix contract
        moves both or neither.
        """
        m_sup, m_exh, c_prev, n_people = 0.10, 0.18, 900.0, 12.0
        A, B, _, _, E, F = mass_matrices(_t(V), _t(G_OCC), _t(M_INF), 1)
        m_mu = max(m_exh - m_sup, 0.0)
        u = torch.stack([_t(m_sup), _t(m_mu), _t(C_OUT), _t(0.0), _t(C_OUT)], dim=-1)
        Ad, Bd = _discretize_onestep(A, B, E, F, u, DT_AHU, transform_mode=False)
        c_now = (
            float(Ad[..., 0, 0]) * c_prev
            + float((Bd @ u.unsqueeze(-1)).reshape(-1)[0])
            + float(Bd[..., 0, OCCUPANCY_SLOT]) * n_people
        )
        got = _invert_raw(m_sup, m_exh, c_prev, c_now, DT_AHU)
        self.assertAlmostEqual(float(got), n_people, places=6)

    def test_only_the_active_slots_of_B_eff_are_ever_nonzero(self):
        """``ACTIVE_INPUT_SLOTS`` is the whole structural support of ``B_eff``.

        The inversion drops every other column before exponentiating, so if
        the zone's matrices ever grow a term in slot 0 or 1 the inversion
        would silently ignore it.  Swept over flows, volumes and the
        supply/exhaust sign change.
        """
        for V_ in (10.0, 120.0, 900.0):
            for m_sup, m_exh in ((0.0, 0.0), (0.10, 0.03), (0.10, 0.18), (0.5, 0.9)):
                A, B, _, _, E, F = mass_matrices(_t(V_), _t(G_OCC), _t(M_INF), 1)
                m_mu = max(m_exh - m_sup, 0.0)
                u = torch.stack(
                    [_t(m_sup), _t(m_mu), _t(C_OUT), _t(7.0), _t(C_OUT)], dim=-1
                )
                _, B_eff = effective_matrices(A, B, E, F, u)
                nonzero = {
                    j
                    for j in range(B_eff.shape[-1])
                    if float(B_eff[..., 0, j].abs().max()) != 0.0
                }
                self.assertTrue(
                    nonzero.issubset(set(ACTIVE_INPUT_SLOTS)),
                    msg=(V_, m_sup, m_exh, sorted(nonzero)),
                )

    def test_reduced_block_is_bit_identical(self):
        """Dropping the zero columns must not move a single bit.

        ``B_d = phi(A_eff dt) dt B_eff`` is linear in ``B_eff`` column by
        column and the dropped columns are exact zeros, so the reduced 3x3
        discretization and the full 5x5 one agree exactly -- under
        ``torch.matrix_exp`` and under ``_expm_ss`` alike.
        """
        slots = ACTIVE_INPUT_SLICE
        for transform_mode in (False, True):
            for V_, m_sup, m_exh in (
                (120.0, 0.10, 0.18), (120.0, 0.10, 0.03),
                (600.0, 0.0, 0.0), (10.0, 0.5, 0.9),
            ):
                A, B, _, _, E, F = mass_matrices(_t(V_), _t(G_OCC), _t(M_INF), 1)
                m_mu = max(m_exh - m_sup, 0.0)
                u = torch.stack(
                    [_t(m_sup), _t(m_mu), _t(C_OUT), _t(0.0), _t(C_OUT)], dim=-1
                )
                Ad, Bd = _discretize_onestep(
                    A, B, E, F, u, DT_AHU, transform_mode=transform_mode
                )
                A_eff, B_eff = effective_matrices(A, B, E, F, u)
                Ad_r, Bd_r = _discretize_onestep(
                    A_eff, B_eff[..., slots], None, None, u[..., slots],
                    DT_AHU, transform_mode=transform_mode,
                )
                msg = (transform_mode, V_, m_sup, m_exh)
                self.assertTrue(torch.equal(Ad, Ad_r), msg=msg)
                self.assertTrue(torch.equal(Bd[..., slots], Bd_r), msg=msg)

    def test_active_slots_are_selected_by_slice_not_by_list(self):
        """A list index would break CUDA-graph capture.

        ``t[..., [2, 3]]`` is advanced indexing: it builds a CPU index
        tensor and copies it to the device, which raises "Cannot copy
        between CPU and CUDA tensors during CUDA graph capture".
        ``torch.compile`` folds the list at trace time, so the failure
        appears only on the uncompiled captured rollout -- after a fit has
        already run.  Keep the selection contiguous and slice-shaped.
        """
        self.assertIsInstance(ACTIVE_INPUT_SLICE, slice)
        self.assertEqual(
            tuple(range(*ACTIVE_INPUT_SLICE.indices(N_INPUTS))), ACTIVE_INPUT_SLOTS
        )
        src = inspect.getsource(tb.OccupancySystem.invert_zoh_occupancy)
        self.assertNotIn("list(ACTIVE_INPUT_SLOTS)", src)
        self.assertIn("ACTIVE_INPUT_SLICE", src)
        # And the slice must actually be a view (no copy, no index tensor).
        t = torch.zeros((2, 3, N_INPUTS), dtype=torch.float64)
        self.assertTrue(t[..., ACTIVE_INPUT_SLICE]._is_view())

    def test_batched_n_c_round_trip(self):
        """Several parallel components at once: the shapes the HTR run uses.

        The hand-rolled inverse took its leading dims from the data
        (``C_now * 0``); the zone takes them from the parameters.  Only the
        second broadcasts per-component parameters against a batch of
        simulations the way the captured rollout does.
        """
        n_c, n_s = 3, 2
        Vs = torch.tensor([90.0, 120.0, 260.0], dtype=torch.float64)
        m_infs = torch.tensor([0.002, 0.004, 0.006], dtype=torch.float64)
        g = torch.full((n_c,), G_OCC, dtype=torch.float64)
        people = torch.tensor(
            [[3.0, 12.0, 7.0], [0.0, 25.0, 1.0]], dtype=torch.float64
        )
        m_sup = torch.full((n_s, n_c), 0.10, dtype=torch.float64)
        m_exh = torch.full((n_s, n_c), 0.18, dtype=torch.float64)
        c_out = torch.full((n_s, n_c), C_OUT, dtype=torch.float64)
        c_prev = torch.full((n_s, n_c), 900.0, dtype=torch.float64)
        mass_params = {"V": Vs, "G_occ": g, "m_inf": m_infs}
        flows = {
            "supplyAirFlowRate": m_sup,
            "exhaustAirFlowRate": m_exh,
            "outdoorCO2": c_out,
        }

        room = tb.BuildingSpaceMassSystem(V=V, G_occ=G_OCC, m_inf=M_INF, id="room")
        room.n_c = n_c
        x1, _ = room.forward(
            c_prev.unsqueeze(-1),
            {**flows, "numberOfPeople": people},
            mass_params,
            DT_AHU,
            transform_mode=True,
        )
        got = tb.OccupancySystem.invert_zoh_occupancy(
            mass_params, flows, c_prev, x1[..., 0], DT_AHU,
            n_c=n_c, transform_mode=True,
        )
        self.assertEqual(tuple(got.shape), (n_s, n_c))
        torch.testing.assert_close(got, people, rtol=1e-9, atol=1e-7)

    @unittest.skipUnless(
        torch.cuda.is_available() and _has_triton(),
        "compile fullgraph diagnostic needs CUDA + Triton",
    )
    def test_compile_fullgraph(self):
        # differentiable_system_models: torch.compile(..., fullgraph=True)
        # is the diagnostic for hidden Python / dtype objects.
        device = torch.device("cuda")

        def fn(Vt, Gt, Mt, m_sup, m_exh, C_out, C_prev, C_now):
            return tb.OccupancySystem.invert_zoh_occupancy(
                {"V": Vt, "G_occ": Gt, "m_inf": Mt},
                {
                    "supplyAirFlowRate": m_sup,
                    "exhaustAirFlowRate": m_exh,
                    "outdoorCO2": C_out,
                },
                C_prev,
                C_now,
                DT_AHU,
                n_c=1,
                transform_mode=True,
            )

        args = tuple(
            t.to(device)
            for t in (
                _t(V), _t(G_OCC), _t(M_INF),
                _t(0.10), _t(0.18), _t(C_OUT), _t(880.0), _t(910.0),
            )
        )
        eager = fn(*args)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        got = compiled(*args)
        torch.testing.assert_close(got, eager, rtol=1e-10, atol=1e-12)

    def test_zero_people_decay(self):
        # OccupancySystem.forward then applies the smooth clamp, which
        # has a faithfulness floor near 0; the ZOH inverse itself is 0.
        c_now = _forward_step(0.10, 0.10, 900.0, 0.0, dt=DT_AHU)
        got = _invert_raw(0.10, 0.10, 900.0, c_now, DT_AHU)
        self.assertAlmostEqual(float(got), 0.0, places=5)

    def test_generation_only(self):
        # No ventilation, no infiltration: the step is exactly linear in
        # time, so the people come straight back out.
        air_mass = V * constants.RHO_AIR
        alpha = G_OCC * (constants.M_AIR / constants.M_CO2) * 1e6
        h, c_prev = 600.0, 800.0
        c_now = c_prev + (alpha / air_mass) * 8.0 * h
        got = tb.OccupancySystem.invert_zoh_occupancy(
            {"V": _t(V), "G_occ": _t(G_OCC), "m_inf": _t(0.0)},
            {
                "supplyAirFlowRate": _t(0.0),
                "exhaustAirFlowRate": _t(0.0),
                "outdoorCO2": _t(C_OUT),
            },
            _t(c_prev),
            _t(c_now),
            h,
        )
        self.assertAlmostEqual(float(got), 8.0, places=6)

    def test_surplus_supply_ignores_exhaust(self):
        # The forward model drops the exhaust when the supply covers it; so
        # must the inversion (same people for any smaller exhaust).
        c_now = _forward_step(0.10, 0.02, 900.0, 12.0)
        self.assertAlmostEqual(
            _invert(0.10, 0.02, 900.0, c_now),
            _invert(0.10, 0.08, 900.0, c_now),
            places=6,
        )

    def test_euler_disagrees_at_ahu_step(self):
        m_sup, m_exh, c_prev = 0.10, 0.18, 900.0
        c_now = _forward_step(m_sup, m_exh, c_prev, 12.0, dt=DT_AHU)
        euler = _euler_invert(m_sup, m_exh, c_prev, c_now, DT_AHU)
        self.assertGreater(abs(euler - 12.0) / 12.0, 0.05)

    def test_old_unbalanced_form_would_disagree(self):
        # Guard against regressing to (m_inf+m_exh)*C - (m_inf+m_sup)*C_out.
        m_sup, m_exh, c_prev = 0.10, 0.18, 900.0
        c_now = _forward_step(m_sup, m_exh, c_prev, 12.0)
        alpha = G_OCC * (constants.M_AIR / constants.M_CO2) * 1e6
        old = (
            constants.RHO_AIR * V * (c_now - c_prev) / DT
            + (M_INF + m_exh) * c_prev
            - (M_INF + m_sup) * C_OUT
        ) / alpha
        self.assertGreater(abs(old - 12.0) / 12.0, 0.05)


if __name__ == "__main__":
    unittest.main()
