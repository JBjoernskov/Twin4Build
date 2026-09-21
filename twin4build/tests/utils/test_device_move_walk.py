"""``Model.to`` walks the model once, not once per component.

The tensor-moving walk follows a component's wiring to the other
components, so a per-component walk visited every object once per
component: quadratic in the model size, minutes on 2000 components.
The walk takes a shared ``seen`` set and skips the semantic model's
entities, which hold no tensors.
"""

# Standard library imports
import unittest
from unittest import mock

# Third party imports
import torch

# Local application imports
import twin4build as tb
from twin4build.utils import device as device_module


def build(n):
    """A chain of ``n`` scalar products fed by one schedule."""
    model = tb.Model(id=f"device_walk_{n}")
    schedule = tb.ScheduleSystem(weekday_ruleset={"ruleset_default_value": 1.0}, id="schedule")
    previous = schedule
    port = "scheduleValue"
    for i in range(n):
        product = tb.ScalarProductSystem(scale_factor=1.0, id=f"product{i}")
        model.add_connection(previous, product, port, "input_1")
        model.add_connection(schedule, product, "scheduleValue", "input_2")
        previous, port = product, "output"
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model


def visits(model):
    """Objects visited by ``model.to`` (one per attribute iteration)."""
    counter = {"n": 0}
    original = device_module._iter_attributes

    def counting(obj):
        counter["n"] += 1
        return original(obj)

    with mock.patch.object(device_module, "_iter_attributes", counting):
        model.to(device="cpu", dtype=torch.float64)
    return counter["n"]


class TestDeviceMoveWalk(unittest.TestCase):
    def test_walk_is_linear_in_the_model_size(self):
        small, large = visits(build(5)), visits(build(40))
        # per-component walks grew the count with the square of the size
        self.assertLess(large, 3.0 * (40 / 5) * small)

    def test_shared_seen_visits_each_object_once(self):
        model = build(3)
        seen: set = set()
        components = list(model.components.values())
        device_module.move_object_tensors(components[0], "cpu", None, seen=seen)
        n_after_first = len(seen)
        for component in components[1:]:
            device_module.move_object_tensors(component, "cpu", None, seen=seen)
        # the first walk already reached every component through the wiring
        self.assertEqual(len(seen), n_after_first)

    def test_semantic_entities_are_not_entered(self):
        class Entity:
            __module__ = "twin4build.model.semantic_model.semantic_model"

            def __init__(self):
                self.payload = torch.zeros(2)

        class Holder:
            __module__ = "twin4build.systems.some_system"

            def __init__(self):
                self.entity = Entity()
                self.value = torch.zeros(2, dtype=torch.float32)

        holder = Holder()
        device_module.move_object_tensors(holder, "cpu", torch.float64)
        self.assertEqual(holder.value.dtype, torch.float64)
        self.assertEqual(holder.entity.payload.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
