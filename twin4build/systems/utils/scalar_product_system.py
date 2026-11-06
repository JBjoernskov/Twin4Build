# Local application imports
import twin4build.core as core
from twin4build.systems.saref4syst.declarations import Input, Output, Setting


class ScalarProductSystem(core.System):
    r"""Multiplies two scalar inputs element-wise with an optional scaling factor.

    Computes ``output = input_1 * input_2 * scale_factor`` at each time step.
    Useful for combining a physical quantity (e.g. power in W) with a weighting
    signal (e.g. electricity price in DKK/kWh) and a unit-conversion constant
    (e.g. ``step_size / 3600 / 1000`` to go from W to kWh).

    Args:
        scale_factor: Constant multiplier applied after the element-wise
            product of the two inputs. Defaults to 1.0.
        **kwargs: Additional keyword arguments passed to the parent
            :class:`core.System`.

    Inputs:
        input_1 (Scalar): First operand.
        input_2 (Scalar): Second operand.

    Outputs:
        output (Scalar): Result of ``input_1 * input_2 * scale_factor``.
    """

    scale_factor = Setting(1.0, doc="Constant multiplier after the element-wise product")
    input_1 = Input(doc="First operand")
    input_2 = Input(doc="Second operand")
    output = Output(doc="input_1 * input_2 * scale_factor")

    def forward(self, x, inputs, p, step_size):
        """Stateless: ``input_1 * input_2 * scale_factor``."""
        return x, {"output": inputs["input_1"] * inputs["input_2"] * self.scale_factor}

