# Third party imports
import torch

# Local application imports
import twin4build.core as core
from twin4build.systems.saref4syst.declarations import Input, Output, Parameter
from twin4build.systems.utils.smooth_saturation import clamp


class ThresholdGate(core.System):
    """The parameters and the threshold of a smooth gate, without ports: the
    base of :class:`SigmoidGate` and of the occupancy detector, which name
    their ports differently.  ``forward`` reads ``INPUT_PORT`` and writes
    ``OUTPUT_PORT``, blending with ``controllerSignal`` when the class
    declares it."""

    #: The ports ``forward`` reads and writes.
    INPUT_PORT = "inputSignal"
    OUTPUT_PORT = "outputSignal"

    threshold = Parameter(0.5, doc="Input level that triggers the gate")
    steepness = Parameter(100.0, doc="Sharpness of the on/off transition")
    polarity = Parameter(1.0, doc="+1: active above the threshold, -1: below")
    default_output = Parameter(0.0, lb=0.0, ub=1.0, doc="The output while the gate is closed")

    @staticmethod
    def _gate(
        x: torch.Tensor,
        threshold: torch.Tensor,
        steepness: torch.Tensor,
        polarity: torch.Tensor,
    ) -> torch.Tensor:
        """Pure sigmoid threshold math on explicit tensors (single source of
        truth for both :meth:`compute_gate` and :meth:`forward`).

        Always uses ``mode="smooth"``: a hard step would zero the gradient
        with respect to ``threshold`` everywhere, making the gate threshold
        structurally non-estimable.
        """
        u = 0.5 + polarity * (x - threshold) * steepness
        return clamp(u, lower=0.0, upper=1.0, curve_start=0.1, mode="smooth")

    def compute_gate(self, x: torch.Tensor) -> torch.Tensor:
        """The gate at ``x`` with the System's own parameters.

        polarity > 0: gate ≈ 1 when x > threshold (active above).
        polarity < 0: gate ≈ 1 when x < threshold (active below).
        """
        return self._gate(x, self.threshold.get(), self.steepness.get(), self.polarity.get())

    def _signal(self, gate, inputs, p):
        """The gate blended with ``controllerSignal`` when the class has that
        port (unwired it is 1).  A parent that calls ``forward`` without it
        (the identified controller, which blends on its own) gets the gate."""
        signal = inputs.get("controllerSignal") if "controllerSignal" in type(self)._declarations["input"] else None
        if signal is None:
            return gate
        return gate * signal + (1.0 - gate) * p["default_output"]

    def forward(self, x, inputs, p, step_size):
        """Stateless: the gate on ``INPUT_PORT``, blended with
        ``controllerSignal`` when the class has it."""
        gate = self._gate(inputs[self.INPUT_PORT], p["threshold"], p["steepness"], p["polarity"])
        return x, {self.OUTPUT_PORT: self._signal(gate, inputs, p)}


class SigmoidGate(ThresholdGate):
    r"""Smooth binary gate with configurable polarity.

    Uses :func:`clamp` (with explicit ``mode="smooth"``) for AD-safe
    differentiability.  Gates always stay in smooth mode regardless of
    the global saturation-mode toggle: a hard step would zero the
    gradient with respect to ``threshold`` everywhere, making the gate
    threshold structurally non-estimable.

    Can be used in two ways:

    1. As a standalone System in the model graph.
    2. As an embedded sub-System (via :meth:`compute_gate` or ``forward``
       called by a parent System such as ``ControllerIdentificationSystem``).

    .. math::

        \sigma = S\bigl(0.5 + p \cdot (x - T) \cdot k\bigr)

    where :math:`T` is the ``threshold``, :math:`k` is the ``steepness``,
    :math:`p` is the ``polarity``, and :math:`S` is the smooth saturation
    function clamping to [0, 1].

    With ``polarity = +1`` (default): gate ≈ 1 when ``x > threshold`` (active above).
    With ``polarity = -1``: gate ≈ 1 when ``x < threshold`` (active below).

    Input ports:
        inputSignal: The value to threshold (Scalar).

        controllerSignal: A controller output to blend with
            ``default_output`` using the gate (Scalar): the output is
            ``gate * controllerSignal + (1 - gate) * default_output``.  This
            lets the gate double as a signal-blender (controller "switch")
            without a separate wrapper System.  Unwired it is 1, so with
            ``default_output = 0`` (the default) the output is the gate.

    Output ports:
        outputSignal: The blended signal (the 0--1 gate signal while
            ``controllerSignal`` is unwired and ``default_output`` is 0).

    Args:
        threshold: Input level that triggers the gate (estimable).
        steepness: Controls sigmoid sharpness (higher = sharper on/off).
        polarity: +1 = active above threshold, -1 = active below (estimable).
        default_output: Fallback value emitted when the gate is closed.
            Default 0.0.
    """

    inputSignal = Input(doc="The value to threshold")
    controllerSignal = Input(default=1.0, doc="A controller output to blend with default_output; unwired: 1")
    outputSignal = Output(doc="The gate, or the blended controller output")


class BandGate(SigmoidGate):
    r"""Smooth band-pass gate: active (≈1) when ``threshold < x < threshold + band``.

    Parameterized by a lower edge ``threshold`` (``T_lo``) and a
    non-negative ``band`` width ``w``; the upper edge is derived as
    ``T_hi = T_lo + w``.  This structurally prevents ``T_lo > T_hi``,
    which would silently produce an always-off gate if upper/lower were
    stored independently.

    Computed as the product of two logistic sigmoids, one rising at the
    lower edge and one falling at the upper edge:

    .. math::

        \sigma = S\bigl(0.5 + k \cdot (x - T_\mathrm{lo})\bigr)
                 \cdot S\bigl(0.5 + k \cdot (T_\mathrm{lo} + w - x)\bigr)

    Yields ``σ ≈ 1`` inside the band and ``σ ≈ 0`` outside.  The concept
    of *polarity* (single-threshold direction) is absorbed into the
    band: direction is implicit in ``w > 0``.

    Designed as a drop-in replacement for :class:`SigmoidGate` inside
    ``ControllerIdentificationSystem``: inherits ``threshold``
    (= low) and ``polarity`` (unused, kept for interface compatibility)
    so that CITS parameter discovery continues to work without
    modification.  A read-only :attr:`threshold_high` property is
    exposed for summary/reporting code that wants the derived upper
    edge.

    Args:
        threshold: Lower edge ``T_lo`` (estimable; stored in inherited
            ``self.threshold`` attribute).
        band: Band width ``w = T_hi - T_lo`` (estimable, clamped to
            ``>= 0``).
        steepness: Logistic steepness ``k`` applied to both sigmoids.
    """

    REMOVED_ATTRIBUTES = {"threshold_high": "give band = threshold_high - threshold (threshold_high is read-only)"}

    threshold = Parameter(0.0, doc="Lower edge of the band")
    band = Parameter(1.0, lb=0.0, doc="Width of the band (>= 0)")

    #: The polarity is absorbed into the band (kept, unused, for the
    #: identified controller's parameter discovery).
    PARAM_NAMES = ("threshold", "band", "steepness", "default_output")

    @property
    def threshold_high(self) -> torch.Tensor:
        """Derived upper edge ``T_hi = threshold + band``: a plain tensor,
        read-only, for summary and reporting code (e.g. CITS
        ``print_detailed_summary``)."""
        return self.threshold.get() + self.band.get()

    @staticmethod
    def _band_gate(
        x: torch.Tensor,
        threshold: torch.Tensor,
        band: torch.Tensor,
        steepness: torch.Tensor,
    ) -> torch.Tensor:
        """Pure band-pass gate math on explicit tensors (single source of
        truth for both :meth:`compute_gate` and :meth:`forward`).  Always
        uses ``mode="smooth"`` so that ``threshold`` and ``band`` retain
        non-zero gradient everywhere."""
        u_lo = 0.5 + (x - threshold) * steepness
        u_hi = 0.5 + (threshold + band - x) * steepness
        s_lo = clamp(u_lo, lower=0.0, upper=1.0, curve_start=0.1, mode="smooth")
        s_hi = clamp(u_hi, lower=0.0, upper=1.0, curve_start=0.1, mode="smooth")
        return s_lo * s_hi

    def compute_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Band-pass gate: product of a lower rising sigmoid and an upper
        falling sigmoid.  ``polarity`` is ignored."""
        return self._band_gate(x, self.threshold.get(), self.band.get(), self.steepness.get())

    def forward(self, x, inputs, p, step_size):
        """Stateless: the band gate, blended with ``controllerSignal``."""
        gate = self._band_gate(inputs[self.INPUT_PORT], p["threshold"], p["band"], p["steepness"])
        return x, {self.OUTPUT_PORT: self._signal(gate, inputs, p)}
