# Standard library imports
import datetime
from typing import Optional

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps


class ShadingDeviceSystem(core.System):
    """A shading device system model for controlling solar heat gain.

    This model represents window blinds, shades, or other devices that control
    solar heat gain through windows. The model acts as a pass-through for shade
    position control signals, allowing other systems to control the shading device.

    The model simply passes through the shade position from input to output, allowing for
    control of the shading device by other systems.

    The shade position is typically represented as a value between 0 and 1, where:
    - 0 represents fully closed/blocked
    - 1 represents fully open/transparent

    Args:
        **kwargs: Additional keyword arguments passed to the parent System class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input = {"shadePosition": tps.Scalar()}
        self.output = {"shadePosition": tps.Scalar()}

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
    ) -> None:
        """Initialize the shading device system.

        This method is a no-op as the shading device system does not require initialization.
        The system has no internal state to initialize and simply passes through
        the shade position from input to output.

        Args:
            start_time (datetime.datetime): Start time of the simulation period.
            end_time (datetime.datetime): End time of the simulation period.
            step_size (int): Time step size in seconds.
        """
        pass

    PARAM_NAMES = ()

    def forward(self, x, inputs, params, sample_time):
        """Pure one-step pass-through (functorch-safe, stateless)."""
        return x, {"shadePosition": inputs["shadePosition"]}

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """Perform one simulation step.

        Passes through the shade position from input to output (typically
        controlled by a schedule or control system).  Thin port-I/O wrapper
        delegating to :meth:`forward`.

        Args:
            second_time (float, optional): Current simulation time in seconds.
            date_time (date_time, optional): Current simulation date and time.
            step_size (float, optional): Time step size in seconds.
            step_index (int, optional): Current simulation step index.
        """
        inputs = {"shadePosition": self.input["shadePosition"].get()}
        _, outs = self.forward(
            None, inputs, self._forward_params(), self._scalar_sample_time(step_size)
        )
        self.output["shadePosition"]._set(outs["shadePosition"], i_t=step_index)
