# Local application imports
import twin4build.utils.types as tps
from twin4build.systems.utils.sigmoid_gate import SigmoidGate


class OccupancyDetectorSystem(SigmoidGate):
    r"""Smooth binary occupancy detector (mimics a PIR sensor).

    Inherits sigmoid threshold logic from :class:`SigmoidGate`.  Overrides
    port names for backward compatibility with existing model graphs.

    Converts a continuous occupancy estimate :math:`N_{occ}` into a smooth
    0--1 signal using a sigmoid:

    .. math::

        \sigma = \frac{1}{1 + e^{-k\,(N_{occ} - T)}}

    where :math:`T` is the ``threshold`` and :math:`k` is the ``steepness``.

    Args:
        threshold: Occupancy level that triggers "occupied" (estimable).
        steepness: Controls sigmoid sharpness (higher = sharper).
        **kwargs: Forwarded to ``core.System`` (must include ``id``).
    """

    INPUT_PORT = "occupancy"
    OUTPUT_PORT = "occupancySignal"

    def __init__(
        self,
        threshold: float = 1,
        steepness: float = 10.0,
        **kwargs,
    ):
        super().__init__(threshold=threshold, steepness=steepness, **kwargs)
        self._input = {"occupancy": tps.Scalar()}
        self._output = {"occupancySignal": tps.Scalar()}
