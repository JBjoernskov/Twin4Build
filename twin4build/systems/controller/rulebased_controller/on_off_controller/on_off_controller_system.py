# Standard library imports
import datetime
from typing import Optional

# Third party imports
import torch

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.translator.translator import StepRule, Node, SignaturePattern
from twin4build.utils.deprecation import deprecate_args


class OnOffControllerSystem(core.System):
    """
    A rule-based on/off (bang-bang) controller.

    The controller compares a measured value against a setpoint at every step
    and outputs one of two constant values. There is no deadband or hysteresis:
    the output switches as soon as the comparison changes.

    Behavior:
        - Normal mode (``is_reverse=False``): output is ``on_value`` when
          ``actualValue > setpointValue``, otherwise ``off_value``.
        - Reverse mode (``is_reverse=True``): output is ``on_value`` when
          ``actualValue < setpointValue``, otherwise ``off_value``
          (e.g. turn heating on when the temperature is below setpoint).

    Args:
        off_value: Output value when the trigger condition is not met. Defaults to 0.
        on_value: Output value when the trigger condition is met. Defaults to 1.
        is_reverse: If True, trigger when the measured value is below the
            setpoint instead of above it. Defaults to False.

    Inputs:
        - "actualValue": Measured value of the controlled property.
        - "setpointValue": Setpoint to compare against.

    Outputs:
        - "inputSignal": Control signal, either ``on_value`` or ``off_value``.
    """

    def __init__(
        self,
        off_value=0,
        on_value=1,
        is_reverse=False,
        **kwargs,
    ):
        legacy = deprecate_args(
            ["offValue", "onValue", "isReverse"],
            ["off_value", "on_value", "is_reverse"],
            [None, None, None],
            kwargs,
        )
        off_value = legacy.get("off_value", off_value)
        on_value = legacy.get("on_value", on_value)
        is_reverse = legacy.get("is_reverse", is_reverse)

        super().__init__(**kwargs)
        self.off_value = off_value
        self.on_value = on_value
        self.is_reverse = is_reverse

        self.input = {"actualValue": tps.Scalar(), "setpointValue": tps.Scalar()}
        self.output = {"inputSignal": tps.Scalar()}
        self._config = {
            "parameters": ["off_value", "on_value", "is_reverse"],
        }

    # Deprecated camelCase aliases (removed in 2.1)
    @property
    def offValue(self):
        from twin4build.utils.deprecation import deprecate_name

        deprecate_name("offValue", "off_value")
        return self.off_value

    @offValue.setter
    def offValue(self, value):
        from twin4build.utils.deprecation import deprecate_name

        deprecate_name("offValue", "off_value")
        self.off_value = value

    @property
    def onValue(self):
        from twin4build.utils.deprecation import deprecate_name

        deprecate_name("onValue", "on_value")
        return self.on_value

    @onValue.setter
    def onValue(self, value):
        from twin4build.utils.deprecation import deprecate_name

        deprecate_name("onValue", "on_value")
        self.on_value = value

    @property
    def isReverse(self):
        from twin4build.utils.deprecation import deprecate_name

        deprecate_name("isReverse", "is_reverse")
        return self.is_reverse

    @isReverse.setter
    def isReverse(self, value):
        from twin4build.utils.deprecation import deprecate_name

        deprecate_name("isReverse", "is_reverse")
        self.is_reverse = value

    @property
    def config(self):
        return self._config

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
    ) -> None:
        """Initialize the controller's input and output ports for simulation."""
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)
        self.input["actualValue"].initialize(
            n_t=max_timesteps,
            n_s=batch_size,
        )
        self.input["setpointValue"].initialize(
            n_t=max_timesteps,
            n_s=batch_size,
        )
        self.output["inputSignal"].initialize(
            n_t=max_timesteps,
            n_s=batch_size,
        )

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """Perform one control step."""
        actual_value = self.input["actualValue"].get()
        setpoint_value = self.input["setpointValue"].get()

        if self.is_reverse:
            trigger_on = actual_value < setpoint_value
        else:
            trigger_on = actual_value > setpoint_value

        output_signal = torch.where(trigger_on, self.on_value, self.off_value)
        self.output["inputSignal"]._set(output_signal, i_t=step_index)


def saref_signature_pattern():
    """Get the SAREF signature pattern of the on-off controller component."""
    node0 = Node(cls=(core.namespace.S4BLDG.RulebasedController))
    node1 = Node(cls=(core.namespace.SAREF.Sensor))
    node2 = Node(cls=(core.namespace.SAREF.Property))
    node3 = Node(cls=(core.namespace.S4BLDG.Schedule))
    sp = SignaturePattern(id="on_off_controller_signature_pattern")
    sp.add_rule(
        StepRule(subject=node0, object=node2, predicate=core.namespace.SAREF.observes)
    )
    sp.add_rule(
        StepRule(subject=node1, object=node2, predicate=core.namespace.SAREF.observes)
    )
    sp.add_rule(
        StepRule(subject=node0, object=node3, predicate=core.namespace.SAREF.hasProfile)
    )
    sp.add_input("actualValue", node1, "measuredValue")
    sp.add_input("setpointValue", node3, "scheduleValue")
    sp.add_modeled_node(node0)
    return sp


def brick_signature_pattern():
    """Get the BRICK signature pattern of the on-off controller component."""
    node0 = Node(cls=core.namespace.BRICK.On_Off_Controller)
    node1 = Node(cls=core.namespace.BRICK.Sensor)
    node2 = Node(cls=core.namespace.BRICK.Setpoint)

    sp = SignaturePattern(id="on_off_controller_signature_pattern_brick")
    sp.add_rule(
        StepRule(subject=node1, object=node0, predicate=core.namespace.BRICK.isPointOf)
    )
    sp.add_rule(
        StepRule(subject=node2, object=node0, predicate=core.namespace.BRICK.isPointOf)
    )

    sp.add_input("actualValue", node1, "measuredValue")
    sp.add_input("setpointValue", node2, "setpoint")
    sp.add_modeled_node(node0)
    return sp


OnOffControllerSystem.add_signature_pattern(brick_signature_pattern())
OnOffControllerSystem.add_signature_pattern(saref_signature_pattern())
