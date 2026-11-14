"""What each output depends on (``OUTPUT_DEPENDENCIES``), checked against the functional step.

Every class that declares its outputs is checked three ways, per configuration and over the union of them:

* sound: what automatic differentiation finds (:func:`signal_dependencies.derive`, several operating regimes) is
  declared;
* without derivatives: changing anything an output does not declare leaves it unchanged
  (:func:`signal_dependencies.independence_violations`, catches a hard switch);
* tight: over the configurations tested, the declaration is exactly what was found.

A class whose equations change so that an output reads something new fails here, before a partition could cut
the model on a stale declaration.
"""

import datetime
import unittest

import torch

import twin4build as tb
import twin4build.utils.types as tps
from twin4build.model import signal_dependencies as sd

START = datetime.datetime(2024, 1, 8, tzinfo=datetime.timezone.utc)
END = START + datetime.timedelta(hours=2)


def s(*v):
    """A scalar port's value, ``(n_c,)``."""
    return torch.tensor(v, dtype=tps.float_dtype())


def vec(*v):
    """A vector port's value for one component, ``(1, n_v)``."""
    return torch.tensor([v], dtype=tps.float_dtype())


def initialized(component):
    component.initialize(start_time=[START], end_time=[END], step_size=[600])
    return component


class DeclarationCase(unittest.TestCase):
    """The three checks over a class's configurations: ``[(component, [inputs per regime]), ...]``."""

    def assertDeclarations(self, configurations, forward_kwargs=None):
        union = {}
        first = configurations[0][0]
        for component, regimes in configurations:
            deps, _ = sd.derive(component, regimes, forward_kwargs=forward_kwargs)
            self.assertEqual(sd.check(component, deps, tight=False), {}, f"{component.id}: unsound")
            for regime in regimes:
                self.assertEqual(sd.independence_violations(component, regime, forward_kwargs=forward_kwargs), {},
                                 f"{component.id}: an undeclared input, parameter or state moves an output")
            for port, dep in deps.items():
                union[port] = union[port].union(dep) if port in union else dep
        self.assertEqual(sd.check(first, union, tight=True), {}, f"{type(first).__name__}: declares more than found")
        return union


class TestTheModule(unittest.TestCase):
    def test_an_undeclared_output_depends_on_everything(self):
        room = initialized(tb.BuildingSpaceSystem(id="r"))
        table = sd.declared(room)
        everything = sd.everything(room)
        self.assertEqual(table["indoorTemperature"], everything)
        self.assertTrue(everything.state)
        self.assertIn("thermal.C_air", everything.parameters)
        self.assertEqual(table["surplusAirFlowRate"],
                         sd.Dependency(frozenset({"supplyAirFlowRate", "exhaustAirFlowRate"}), False, frozenset()))

    def test_the_state_update_is_a_pseudo_output(self):
        """What a copy carrying the state needs: undeclared, everything; derived, what the update reads."""
        room = initialized(tb.BuildingSpaceSystem(id="r"))
        self.assertEqual(sd.declared(room)[sd.STATE], sd.everything(room))
        deps, _ = sd.derive(room, TestRoom().regimes())
        self.assertIn(sd.STATE, deps)
        self.assertTrue(deps[sd.STATE].state)
        self.assertIn("supplyAirTemperature", deps[sd.STATE].inputs)
        self.assertTrue(sd.everything(room).covers(deps[sd.STATE]))
        self.assertNotIn(sd.STATE, sd.declared(tb.DamperSystem(id="d")))  # no state, no update
        self.assertIn("STATE_DEPENDENCIES", sd.proposal(deps))

    def test_a_declaration_with_an_unknown_port_or_parameter_is_an_error(self):
        class Typo(tb.DamperSystem):
            OUTPUT_DEPENDENCIES = {"airFlowRate": {"inputs": ("damperPositon",)}}

        with self.assertRaises(ValueError):
            sd.declared(Typo(id="t"))

        class BadParameter(tb.DamperSystem):
            OUTPUT_DEPENDENCIES = {"airFlowRate": {"inputs": ("damperPosition",), "parameters": ("nominal",)}}

        with self.assertRaises(ValueError):
            sd.declared(BadParameter(id="t"))

    def test_a_narrower_declaration_is_caught_both_ways(self):
        """A damper that leaves out its fan gate: the derivation finds ``fanSpeed`` (while the fan starts), and
        stopping the fan moves the flow."""

        class Narrow(tb.DamperSystem):
            OUTPUT_DEPENDENCIES = dict(tb.DamperSystem.OUTPUT_DEPENDENCIES,
                                       airFlowRate={"inputs": ("damperPosition",),
                                                    "parameters": ("nominalAirFlowRate", "a", "c")})

        damper = initialized(Narrow(id="narrow"))
        starting = {"damperPosition": s(0.5), "fanSpeed": s(0.05)}
        deps, _ = sd.derive(damper, [starting])
        self.assertIn("airFlowRate", sd.check(damper, deps, tight=False))
        # from a starting fan the bump (x3 + 1) runs it: the flow moves (from a running one it would stay at the gate's
        # limit -- the regimes must cross the switch, as for the derivation)
        self.assertEqual(sd.independence_violations(damper, starting), {"airFlowRate": ["input fanSpeed"]})

    def test_covers(self):
        small = sd.Dependency(frozenset({"a"}), False, frozenset({"p"}))
        big = sd.Dependency(frozenset({"a", "b"}), True, frozenset({"p", "q"}))
        self.assertTrue(big.covers(small))
        self.assertFalse(small.covers(big))
        self.assertEqual(small.union(big), big)


class TestDamper(DeclarationCase):
    def test_declarations(self):
        running = {"damperPosition": s(0.5), "fanSpeed": s(0.8)}
        starting = {"damperPosition": s(0.5), "fanSpeed": s(0.05)}
        tied = initialized(tb.DamperSystem(id="tied"))
        untied = initialized(tb.DamperSystem(id="untied", c=-0.5))
        union = self.assertDeclarations([(tied, [running, starting]), (untied, [running, starting])])
        self.assertNotIn("c", sd.derive(tied, [running, starting])[0]["airFlowRate"].parameters)
        self.assertIn("c", union["airFlowRate"].parameters)


class TestRoom(DeclarationCase):
    """The imbalance flows depend on the flow ports only: a world that needs a room's surplus needs its terminals,
    not its temperature (the thermal and CO2 outputs are undeclared)."""

    INPUTS = {"outdoorTemperature": s(2.0), "supplyAirFlowRate": vec(0.05, 0.04), "exhaustAirFlowRate": vec(0.06),
              "supplyAirTemperature": s(19.0), "globalIrradiation": s(100.0), "numberOfPeople": s(3.0),
              "heatGain": s(200.0), "makeUpAirTemperature": s(21.0), "plugLoad": s(150.0),
              "wallHeatGain": s(50.0), "infiltrationAirFlowRate": s(0.01), "outdoorCo2Concentration": s(420.0),
              "makeUpAirCO2": s(600.0), "supplyAirCO2": s(430.0), "exchangeCO2Gain": s(0.0)}

    def regimes(self, ports=None):
        surplus = dict(self.INPUTS)
        deficit = dict(self.INPUTS, exhaustAirFlowRate=vec(0.12))
        if ports is not None:
            surplus = {k: v for k, v in surplus.items() if k in ports}
            deficit = {k: v for k, v in deficit.items() if k in ports}
        return [surplus, deficit]

    def test_composite(self):
        room = initialized(tb.BuildingSpaceSystem(id="room"))
        union = self.assertDeclarations([(room, self.regimes())])
        self.assertFalse(union["surplusAirFlowRate"].state)
        self.assertTrue(union["indoorTemperature"].state)

    def test_thermal_part(self):
        thermal = initialized(tb.BuildingSpaceThermalSystem(id="thermal"))
        regimes = [{k: (v.sum(dim=-1) if v.dim() > 1 else v) for k, v in r.items()}
                   for r in self.regimes(set(thermal.input))]
        self.assertDeclarations([(thermal, regimes)])


class TestProjections(unittest.TestCase):
    """A room's state-free outputs computed without the room (``OUTPUT_PROJECTIONS``): the projection equals the
    room's own step for every regime, at any state."""

    def assertProjects(self, component, regimes):
        for port in ("surplusAirFlowRate", "makeUpAirFlowRate"):
            cls = sd.projection(component, port)
            self.assertIs(cls, tb.AirImbalanceSystem)
        projection = tb.AirImbalanceSystem(id="p")
        state = component.get_state() if sd.has_state(component) else None
        for regime in regimes:
            # the component's own port shapes (the thermal part's flows are scalars); the projection's are vectors
            own_shapes = {k: (v.sum(dim=-1) if v.dim() > 1 and isinstance(component.input[k], tps.Scalar) else v)
                          for k, v in regime.items()}
            _, full = component.forward(state if state is None else state * 1.7, own_shapes,
                                        component.full_params(sd.forward_params(component), 600.0), 600.0)
            _, own = projection.forward(None, {k: v for k, v in regime.items() if k in projection.input}, {}, 600.0)
            for port in ("surplusAirFlowRate", "makeUpAirFlowRate"):
                torch.testing.assert_close(own[port], full[port])
        self.assertEqual(sd.declared(projection)["surplusAirFlowRate"].state, False)

    def test_room(self):
        room = initialized(tb.BuildingSpaceSystem(id="room"))
        self.assertProjects(room, TestRoom().regimes())

    def test_thermal_part(self):
        thermal = initialized(tb.BuildingSpaceThermalSystem(id="thermal"))
        regimes = [{k: v for k, v in r.items() if k in thermal.input} for r in TestRoom().regimes()]
        self.assertProjects(thermal, regimes)

    def test_the_projection_s_declarations(self):
        p = tb.AirImbalanceSystem(id="p")
        regime = {"supplyAirFlowRate": vec(0.05, 0.04), "exhaustAirFlowRate": vec(0.06), "extractAirFlowRate": vec(0.03)}
        deficit = dict(regime, extractAirFlowRate=vec(0.2))
        deps, _ = sd.derive(p, [regime, deficit])
        self.assertEqual(sd.check(p, deps), {})
        self.assertEqual(sd.independence_violations(p, regime), {})

    def test_an_output_with_a_state_cannot_be_projected(self):
        class Wrong(tb.BuildingSpaceSystem):
            OUTPUT_PROJECTIONS = {"indoorTemperature": "AirImbalanceSystem"}

        with self.assertRaises(ValueError):
            sd.projection(initialized(Wrong(id="w")), "indoorTemperature")


class TestJunctions(DeclarationCase):
    def test_supply(self):
        junction = tb.SupplyFlowJunctionSystem(id="supply")
        self.assertDeclarations([(junction, [{"airFlowRateOut": vec(0.1, 0.2, 0.3)}])])
        _, slots = sd.derive(junction, {"airFlowRateOut": vec(0.1, 0.2, 0.3)})
        self.assertEqual(slots[("airFlowRateIn", "airFlowRateOut")], {0: frozenset({0, 1, 2})})  # every slot

    def test_return(self):
        junction = tb.ReturnFlowJunctionSystem(id="return")
        self.assertDeclarations([(junction, [{"airFlowRateIn": vec(0.1, 0.2), "airTemperatureIn": vec(21.0, 23.0)}])])


if __name__ == "__main__":
    unittest.main()
