# Local application imports
import twin4build.utils.types as tps
from twin4build.systems.utils.sigmoid_gate import SigmoidGate


class OccupancyDetectorSystem(SigmoidGate):
    r"""Smooth binary occupancy detector (mimics a PIR sensor).

    Inherits sigmoid threshold logic from :class:`SigmoidGate`.  Overrides
    port names for backward compatibility with existing model graphs.

    Converts a continuous occupancy estimate :math:`N_{occ}` into a smooth
    0--1 signal.  Despite the name, this is *not* a logistic sigmoid: it is
    :meth:`SigmoidGate._gate`, a linear ramp clamped to [0, 1] with a
    power-law tail:

    .. math::

        \sigma = S\bigl(0.5 + k\,(N_{occ} - T)\bigr)

    where :math:`T` is the ``threshold``, :math:`k` is the ``steepness``, and
    :math:`S` is the smooth clamp of :func:`~twin4build.systems.utils.\
smooth_saturation.clamp`.

    **Choosing ``steepness`` for estimation.**  The ramp is linear only over
    :math:`|N_{occ} - T| < 1/(2k)`; outside that window the gradient decays
    as a power law.  So ``steepness`` sets not just how sharp the on/off
    transition looks, but how wide the window is in which ``threshold`` (and
    anything upstream that shifts :math:`N_{occ}`) has usable gradient.  At
    :math:`k = 50` the gradient half an occupant off the threshold is ~4
    orders of magnitude below its peak, which is effectively a hard step for
    a gradient-based estimator: once the gate saturates, no solver brings it
    back.  Prefer a low ``steepness`` when the gate sits inside a calibration
    loop.

    **``threshold`` is rarely a constant.**  It is compared against an
    inferred occupancy whose *scale* usually depends on other estimated
    parameters (e.g. ``OccupancySystem`` divides by ``mass.G_occ``, so
    :math:`N_{occ} \propto 1/G_{occ}`).  Fixing the threshold while such a
    parameter is free makes the gate a degenerate direction of the
    objective; estimate the two together.

    Args:
        threshold: Occupancy level that triggers "occupied" (estimable --
            and usually should be, see above).
        steepness: Ramp gain; the differentiable window is ``1/steepness``
            wide (higher = sharper, but harder to calibrate through).
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
