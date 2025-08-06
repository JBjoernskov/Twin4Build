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
        startTime: datetime.datetime,
        endTime: datetime.datetime,
        stepSize: int,
        simulator: core.Simulator,
    ) -> None:
        """Initialize the shading device system.

        This method is a no-op as the shading device system does not require initialization.
        The system has no internal state to initialize and simply passes through
        the shade position from input to output.

        Args:
            startTime (datetime.datetime): Start time of the simulation period.
            endTime (datetime.datetime): End time of the simulation period.
            stepSize (int): Time step size in seconds.
            simulator (core.Simulator): Simulation model object.
        """
        pass

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        stepSize: int,
        stepIndex: int,
    ) -> None:
        """Perform one simulation step.

        This method passes through the shade position from input to output.
        The shade position is typically controlled by a schedule or control system.

        Args:
            secondTime (float, optional): Current simulation time in seconds.
            dateTime (datetime, optional): Current simulation date and time.
            stepSize (float, optional): Time step size in seconds.
            stepIndex (int, optional): Current simulation step index.
        """
        self.output["shadePosition"].set(self.input["shadePosition"], stepIndex)
