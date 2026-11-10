"""A signal passed on as a value, without its gradient.

:func:`twin4build.model.partition.split_worlds` puts one between a provided component and each world that reads it
but is not its home (7 Nov 2026): a building's hidden interior node is simulated once, in the world that reads it only
through measurements (the units' return air), and fitted there; the rooms' worlds read its temperature as it is, so
their residuals move only their own parameters and the worlds stay separate blocks for the fit.  The partition counts
the barrier's output as exogenous: it ties no groups, and the estimator's structure walk
(``FunctionalModel.index_coupling``) lets no parameter reach through it.  The simulator does not: the barrier reads a
simulated, parameter-dependent signal, so it is evaluated every step, never recorded once.
"""

# Local application imports
from twin4build.systems.utils.pass_input_to_output import PassInputToOutput


class GradientBarrierSystem(PassInputToOutput):
    r"""Pass ``value`` on unchanged, without a gradient: :math:`y = x`, :math:`\partial y / \partial x = 0`."""

    def forward(self, x, inputs, p, step_size):
        """Stateless: the input's value, detached from the graph (functorch-safe)."""
        return x, {"value": inputs["value"].detach()}
