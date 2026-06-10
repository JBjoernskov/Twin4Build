# Standard library imports
import datetime
from typing import List

# Third party imports
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.utils.smooth_saturation import clamp


class SigmoidGate(core.System, nn.Module):
    r"""Smooth binary gate with configurable polarity.

    Uses :func:`clamp` (with explicit ``mode="smooth"``) for AD-safe
    differentiability.  Gates always stay in smooth mode regardless of
    the global saturation-mode toggle: a hard step would zero the
    gradient with respect to ``threshold`` everywhere, making the gate
    threshold structurally non-estimable.

    Can be used in two ways:

    1. As a standalone System in the model graph (via ``do_step`` with I/O ports).
    2. As an embedded sub-System (via :meth:`compute_gate` called directly by a
       parent System such as ``ControllerIdentificationTorchSystem``).

    .. math::

        \sigma = S\bigl(0.5 + p \cdot (x - T) \cdot k\bigr)

    where :math:`T` is the ``threshold``, :math:`k` is the ``steepness``,
    :math:`p` is the ``polarity``, and :math:`S` is the smooth saturation
    function clamping to [0, 1].

    With ``polarity = +1`` (default): gate ≈ 1 when ``x > threshold`` (active above).
    With ``polarity = -1``: gate ≈ 1 when ``x < threshold`` (active below).

    Input ports:
        inputSignal: The value to threshold (Scalar).
        controllerSignal: Optional controller output to blend with
            ``default_output`` using the gate (Scalar). When wired, the
            output becomes
            ``gate * controllerSignal + (1 - gate) * default_output``
            instead of the raw gate signal.  This lets the gate double as
            a signal-blender (controller "switch") without needing a
            separate wrapper System.
    Output ports:
        outputSignal: Either the 0--1 gate signal (when ``controllerSignal``
            is unwired) or the blended controller output (when wired).

    Args:
        threshold: Input level that triggers the gate (estimable).
        steepness: Controls sigmoid sharpness (higher = sharper on/off).
        polarity: +1 = active above threshold, -1 = active below (estimable).
        default_output: Fallback value emitted when the gate is closed
            (only used when ``controllerSignal`` is wired).  Default 0.0.
        **kwargs: Forwarded to ``core.System`` (must include ``id``).
    """

    def __init__(
        self,
        threshold: float = 0.5,
        steepness: float = 100.0,
        polarity: float = 1.0,
        default_output: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        self.threshold = tps.Parameter(
            torch.tensor(threshold, dtype=torch.float64), requires_grad=False
        )
        self.steepness = tps.Parameter(
            torch.tensor(steepness, dtype=torch.float64), requires_grad=False
        )
        self.polarity = tps.Parameter(
            torch.tensor(polarity, dtype=torch.float64), requires_grad=False
        )
        self.default_output = tps.Parameter(
            torch.tensor(default_output, dtype=torch.float64),
            min_value=0.0,
            max_value=1.0,
            requires_grad=False,
        )

        self._input = {
            "inputSignal": tps.Scalar(),
            "controllerSignal": tps.Scalar(optional=True),
        }
        self._output = {"outputSignal": tps.Scalar()}
        self._config = {
            "parameters": ["threshold", "steepness", "polarity", "default_output"]
        }
        self.INITIALIZED = False
        self._controller_wired = False

    @property
    def config(self):
        return self._config

    def initialize(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: List[int],
    ) -> None:
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)
        for inp in self.input.values():
            inp.initialize(n_t=max_timesteps, n_s=batch_size)
        for out in self.output.values():
            out.initialize(n_t=max_timesteps, n_s=batch_size)

        self.threshold = self.threshold.expand_to_n_c(self.n_c)
        self.steepness = self.steepness.expand_to_n_c(self.n_c)
        self.polarity = self.polarity.expand_to_n_c(self.n_c)
        self.default_output = self.default_output.expand_to_n_c(self.n_c)

        # Detect whether the optional controllerSignal input is wired.
        # Mirrors the check in BuildingSpaceThermalTorchSystem.setup_variable_inputs:
        # look for a ConnectionPoint on the controllerSignal port with at
        # least one incoming connection.
        self._controller_wired = False
        for cp in self.connects_at:
            if cp.input_port == "controllerSignal" and len(
                cp.connects_system_through
            ) > 0:
                self._controller_wired = True
                break

        self.INITIALIZED = True

    def compute_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Core sigmoid threshold logic -- reusable by subclasses and parent systems.

        polarity > 0: gate ≈ 1 when x > threshold (active above).
        polarity < 0: gate ≈ 1 when x < threshold (active below).

        Always uses ``mode="smooth"``: a hard step would zero the
        gradient with respect to ``threshold`` everywhere, making the
        gate threshold structurally non-estimable.
        """
        error = self.polarity.get() * (x - self.threshold.get())
        u = 0.5 + error * self.steepness.get()
        return clamp(u, lower=0.0, upper=1.0, curve_start=0.1, mode="smooth")

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        x = self.input["inputSignal"].get()
        gate = self.compute_gate(x)
        if self._controller_wired:
            ctrl = self.input["controllerSignal"].get()
            signal = gate * ctrl + (1.0 - gate) * self.default_output.get()
        else:
            signal = gate
        self.output["outputSignal"]._set(signal, i_t=step_index)


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
    ``ControllerIdentificationTorchSystem``: inherits ``threshold``
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
        **kwargs: Forwarded to :class:`SigmoidGate` (must include ``id``).
    """

    def __init__(
        self,
        threshold: float = 0.0,
        band: float = 1.0,
        steepness: float = 100.0,
        default_output: float = 0.0,
        **kwargs,
    ):
        # Backward-compatibility: the pre-``band`` API used
        # ``threshold_high`` as a kwarg.  Translate it to the equivalent
        # ``band = threshold_high - threshold`` (clamped to 0) and emit
        # a one-shot DeprecationWarning so downstream code can migrate.
        if "threshold_high" in kwargs:
            import warnings

            legacy_hi = float(kwargs.pop("threshold_high"))
            derived_band = max(0.0, legacy_hi - float(threshold))
            warnings.warn(
                "BandGate(threshold_high=...) is deprecated; pass "
                "``band=threshold_high - threshold`` instead. "
                "``band`` is clamped non-negative, which structurally "
                "prevents the inverted-band (T_hi < T_lo) failure mode. "
                f"Translating threshold_high={legacy_hi} -> "
                f"band={derived_band} given threshold={threshold}.",
                DeprecationWarning,
                stacklevel=2,
            )
            band = derived_band

        super().__init__(
            threshold=threshold,
            steepness=steepness,
            polarity=1.0,
            default_output=default_output,
            **kwargs,
        )
        # ``band`` is the *width* of the band; constrained non-negative
        # so the upper edge ``threshold + band`` is always ``>= threshold``.
        self.band = tps.Parameter(
            torch.tensor(band, dtype=torch.float64),
            min_value=0.0,
            requires_grad=False,
        )
        self._config["parameters"] = [
            "threshold",
            "band",
            "steepness",
            "default_output",
        ]

    @property
    def threshold_high(self) -> torch.Tensor:
        """Derived upper edge ``T_hi = threshold + band``.

        Returned as a plain :class:`torch.Tensor` (not a
        :class:`tps.Parameter`) so it stays always-consistent with
        ``threshold`` and ``band`` — callers should treat it as
        read-only.  Exposed for summary/reporting code (e.g. CITS
        ``print_detailed_summary``) that wants to display the band as
        ``[T_lo, T_hi]``.
        """
        return self.threshold.get() + self.band.get()

    def initialize(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: List[int],
    ) -> None:
        super().initialize(start_time, end_time, step_size)
        self.band = self.band.expand_to_n_c(self.n_c)

    def compute_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Band-pass gate: product of a lower rising sigmoid and an upper
        falling sigmoid.  ``polarity`` is ignored.  Always uses
        ``mode="smooth"`` so that ``threshold`` and ``band`` retain
        non-zero gradient everywhere.
        """
        k = self.steepness.get()
        lo = self.threshold.get()
        w = self.band.get()
        u_lo = 0.5 + (x - lo) * k
        u_hi = 0.5 + (lo + w - x) * k
        s_lo = clamp(u_lo, lower=0.0, upper=1.0, curve_start=0.1, mode="smooth")
        s_hi = clamp(u_hi, lower=0.0, upper=1.0, curve_start=0.1, mode="smooth")
        return s_lo * s_hi
