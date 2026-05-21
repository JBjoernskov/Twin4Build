# Standard library imports
import datetime
from typing import Optional

# Third party imports
import torch

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.translator.translator import StepRule, AnyPathRule, Node, SignaturePattern


class OnOffControllerSystem(core.System):

    def __init__(self, offValue=0, onValue=1, isReverse=False, **kwargs):
        super().__init__(**kwargs)
        self.offValue = offValue
        self.onValue = onValue
        self.isReverse = isReverse

        self.input = {"actualValue": tps.Scalar(), "setpointValue": tps.Scalar()}
        self.output = {"inputSignal": tps.Scalar()}
        self._config = {
            "parameters": ["offValue", "onValue", "isReverse"],
        }

    @property
    def config(self):
        return self._config

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
    ) -> None:
        """
        This function initializes the FMU component by setting the start_time and fmu_filename attributes,
        and then sets the parameters for the FMU model.
        """
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
        """
        This function calls the do_step method of the FMU component, and then sets the output of the FMU model.
        """
        actual_value = self.input["actualValue"].get()
        setpoint_value = self.input["setpointValue"].get()

        # Determine trigger condition based on reverse mode
        # Reverse: trigger ON when actual < setpoint
        # Normal: trigger ON when actual > setpoint
        if self.isReverse:
            trigger_on = actual_value < setpoint_value
        else:
            trigger_on = actual_value > setpoint_value

        # Select output signal based on trigger condition
        output_signal = torch.where(trigger_on, self.onValue, self.offValue)
        self.output["inputSignal"]._set(output_signal, i_t=step_index)


def saref_signature_pattern():
    """
    Get the SAREF signature pattern of the on-off controller component.

    Returns:
        SignaturePattern: The SAREF signature pattern of the on-off controller component.
    """
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
    """
    Get the BRICK signature pattern of the on-off controller component.

    Returns:
        SignaturePattern: The BRICK signature pattern of the on-off controller component.
    """
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
