# Third party imports
import torch

# Local application imports
import twin4build.core as core
from twin4build.systems.saref4syst.declarations import Input, Output, Setting


class OnOffSystem(core.System):
    r"""
    On-Off System.

    Gates a signal based on a criteria input: if the "criteriaValue" input is
    greater than or equal to ``threshold``, the "value" input is passed through
    to the output; otherwise the output is set to ``is_off_value``.

    Args:
        threshold: Threshold that the "criteriaValue" input is compared against.
        is_on_value: Currently unused. The argument is accepted but never
            stored; when the criteria is met, the "value" input is passed
            through instead.
        is_off_value: Output value when criteriaValue < threshold.
        **kwargs: Additional keyword arguments
    """

    threshold = Setting(None, doc="What criteriaValue is compared against")
    is_on_value = Setting(None, doc="Unused (accepted for compatibility)")
    is_off_value = Setting(None, doc="The output while criteriaValue < threshold")
    value = Input(doc="Passed through while criteriaValue >= threshold")
    criteriaValue = Input(doc="Compared against the threshold")
    value = Output(doc="The gated value")

    def forward(self, x, inputs, p, step_size):
        """Stateless: ``value`` where ``criteriaValue >= threshold``, else ``is_off_value``."""
        input_value = inputs["value"]
        output_value = torch.where(
            inputs["criteriaValue"] >= self.threshold,
            input_value,
            torch.full_like(input_value, self.is_off_value),
        )
        return x, {"value": output_value}

