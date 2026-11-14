"""The library's settings: every knob in one place.

Each setting is declared once, at the end of this module, with its name, its
environment variable, its type, its default and what it does (the manual's
settings page lists them).  A setting's value is, first found:

1. the value :func:`set` (or :func:`override`) gave it in this process;
2. its environment variable;
3. its default.

The library reads a setting where it uses it, never when a module is
imported, so a value set after ``import twin4build`` holds from its next use.
A step that ``torch.compile`` or a CUDA graph has already traced may keep the
values it was traced with: give those settings their values before the run.

::

    import twin4build as tb

    tb.settings.value("zoh_expm")  # 'phi1'
    tb.settings.set(zoh_expm="block")  # for the rest of this process
    with tb.settings.override(jac_check=True):  # for this block only
        ...
    tb.settings.reset("zoh_expm")  # the environment variable's or the default
    tb.settings.snapshot()  # {name: {"value": ..., "source": ...}}
"""

from __future__ import annotations

# Standard library imports
import contextlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

# Local application imports
from twin4build.utils.logger import LOGGER

__all__ = [
    "value",
    "set",
    "reset",
    "override",
    "snapshot",
    "log_active",
]

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")
_MISSING = object()


@dataclass(frozen=True)
class _Setting:
    """A knob: its name, environment variable, type, default and what it does.

    ``kind`` is ``bool``, ``int`` or ``str``.  A ``str`` setting with
    ``choices`` takes one of them, in any case; a ``str`` setting whose default
    is ``None`` is off until it is given a value (a file name, a filter).
    """

    name: str
    env: str
    kind: type
    default: Any
    doc: str
    choices: Tuple[str, ...] = ()


_REGISTRY: Dict[str, _Setting] = {}
_SET: Dict[str, Any] = {}
_LOGGED: tuple = ()


def _register(setting: _Setting) -> None:
    """Declare a setting; only this module declares them, at its end."""
    if setting.kind not in (bool, int, str):
        raise TypeError(f"setting {setting.name!r}: kind is bool, int or str")
    if setting.name in _REGISTRY:
        raise ValueError(f"setting {setting.name!r} is already declared")
    if any(other.env == setting.env for other in _REGISTRY.values()):
        raise ValueError(f"setting {setting.name!r}: {setting.env} is taken")
    _check(setting, setting.default, f"setting {setting.name!r}")
    _REGISTRY[setting.name] = setting


def _get(name: str) -> _Setting:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"no setting {name!r}; the settings are {known}") from None


def _check(setting: _Setting, v: Any, where: str) -> Any:
    """``v`` as a value of ``setting``, or an error naming ``where`` it came from."""
    if v is None and setting.default is None:
        return None
    if not isinstance(v, setting.kind) or (setting.kind is int and isinstance(v, bool)):
        raise TypeError(f"{where} is a {setting.kind.__name__}, not {v!r}")
    if setting.choices:
        v = v.strip().lower()
        if v not in setting.choices:
            choices = ", ".join(setting.choices)
            raise ValueError(f"{where}: {v!r} is not one of {choices}")
    return v


def _parse(setting: _Setting, text: str) -> Any:
    """An environment variable's (stripped, non-empty) text as a value."""
    where = f"{setting.env}={text!r}"
    if setting.kind is bool:
        if text.lower() in _TRUE:
            return True
        if text.lower() in _FALSE:
            return False
        raise ValueError(f"{where}: give one of {', '.join(_TRUE + _FALSE)}")
    if setting.kind is int:
        try:
            return int(text)
        except ValueError:
            raise ValueError(f"{where}: give a whole number") from None
    return _check(setting, text, where)


def _resolve(name: str) -> Tuple[Any, str]:
    """``(value, source)``: from :func:`set`, else the environment, else the default."""
    setting = _get(name)
    if name in _SET:
        return _SET[name], "set"
    text = os.environ.get(setting.env, "").strip()
    if text:
        return _parse(setting, text), setting.env
    return setting.default, "default"


def value(name: str) -> Any:
    """The setting's value now: :func:`set`'s, else its environment variable's
    (an empty variable counts as unset), else its default."""
    return _resolve(name)[0]


def set(**values: Any) -> None:  # noqa: A001 -- read as tb.settings.set(...)
    """Give settings a value for the rest of this process: ``set(jac_check=True)``.

    Every name and value is checked before any is applied.  ``None`` turns off
    a setting whose default is ``None``, whatever its environment variable says.
    """
    checked = {
        name: _check(_get(name), v, f"setting {name!r}") for name, v in values.items()
    }
    _SET.update(checked)


def reset(*names: str) -> None:
    """Undo :func:`set` for ``names`` (for every setting when none is given)."""
    for name in names:
        _get(name)
    for name in names or tuple(_SET):
        _SET.pop(name, None)


@contextlib.contextmanager
def override(**values: Any):
    """:func:`set` for the length of a ``with`` block; afterwards every setting
    is as it was (set or not)."""
    before = {name: _SET.get(name, _MISSING) for name in values}
    set(**values)
    try:
        yield
    finally:
        for name, v in before.items():
            if v is _MISSING:
                _SET.pop(name, None)
            else:
                _SET[name] = v


def snapshot() -> Dict[str, Dict[str, Any]]:
    """Every setting's value and its source, to record with a result.

    ``{name: {"value": ..., "source": ...}}``; the source is ``"set"``, the
    environment variable's name, or ``"default"``.  The results of
    ``Estimator.estimate``, ``Optimizer.optimize`` and
    ``Optimizer.pareto_front`` carry it as ``settings``.
    """
    out = {}
    for name in _REGISTRY:
        v, source = _resolve(name)
        out[name] = {"value": v, "source": source}
    return out


def log_active(where: str) -> None:
    """Log the settings that are off their defaults, when they have changed
    since the last such line (``where``: the run that starts, e.g. "estimate")."""
    global _LOGGED
    active = tuple(
        (name, entry["value"], entry["source"])
        for name, entry in snapshot().items()
        if entry["value"] != _REGISTRY[name].default
    )
    if active == _LOGGED:
        return
    _LOGGED = active
    text = ", ".join(f"{name}={v!r} ({source})" for name, v, source in active)
    LOGGER.info("Settings (%s): %s", where, text or "all at their defaults")


# ---------------------------------------------------------------------------
# The library's settings (docs/source/manual/settings.rst describes each).
# ---------------------------------------------------------------------------
_register(
    _Setting(
        name="zoh_expm",
        env="T4B_ZOH_EXPM",
        kind=str,
        default="phi1",
        choices=("phi1", "block", "aug"),
        doc="How a state-space step is discretised in transform mode "
        "(compiled, captured, vmapped): 'phi1' exponentiates A T and carries "
        "phi1(A T) through the same scaling and squaring, Bd = phi1(A T) B T; "
        "'block' exponentiates the (n + m) block [[A T, B T], [0, 0]]; 'aug' "
        "reads the next state off one exponential of the (n + 1) matrix "
        "[[A T, B T u], [0, 0]].  The same zero-order-hold step to rounding.",
    )
)
_register(
    _Setting(
        name="small_matmul_max",
        env="T4B_SMALL_MATMUL_MAX",
        kind=int,
        default=16,
        doc="The largest matrix whose exponential is written with broadcast "
        "products while torch.compile traces (Inductor fuses them into few "
        "kernels); a larger one uses bmm.",
    )
)
_register(
    _Setting(
        name="fused_sparse",
        env="T4B_FUSED_SPARSE",
        kind=bool,
        default=True,
        doc="A fused block's bilinear terms (state x input, input x input) are "
        "held as their entries, not as dense tensors that are almost all zeros; "
        "off: dense.  The same matrices; the trajectories agree to rounding.",
    )
)
_register(
    _Setting(
        name="expm_checkpoint",
        env="T4B_EXPM_CHECKPOINT",
        kind=bool,
        default=False,
        doc="Recompute each matrix exponential in the backward pass instead of "
        "saving its chain (torch.utils.checkpoint): one matrix per component per "
        "step stays on the autograd tape.",
    )
)
_register(
    _Setting(
        name="rollout_checkpoint",
        env="T4B_ROLLOUT_CHECKPOINT",
        kind=bool,
        default=False,
        doc="Recompute every time step of a functional rollout in the backward "
        "pass instead of saving its intermediates: the tape holds the state and "
        "exogenous rows per step, and the backward pass runs each step again.",
    )
)
_register(
    _Setting(
        name="testing",
        env="TWIN4BUILD_TESTING",
        kind=bool,
        default=False,
        doc="Estimators and optimizers stop after one iteration and a Pareto "
        "front computes at most three points.  The notebook tests set it, so "
        "that every cell runs, quickly.",
    )
)
_register(
    _Setting(
        name="jac_check",
        env="TWIN4BUILD_JAC_CHECK",
        kind=bool,
        default=False,
        doc="Collocation compares the composed constraint Jacobian with finite "
        "differences on a few columns and logs the largest difference.",
    )
)
_register(
    _Setting(
        name="bench_timestep",
        env="TWIN4BUILD_BENCH_TIMESTEP",
        kind=bool,
        default=False,
        doc="Collocation times one step of every segment both ways, the "
        "functional map and the object-graph simulation, and logs the times "
        "and their agreement.",
    )
)
_register(
    _Setting(
        name="match_diag_file",
        env="TWIN4BUILD_MATCH_DIAG_FILE",
        kind=str,
        default=None,
        doc="The translator writes its pattern-matching trace (per-node walker "
        "decisions, merge attempts) to this file.",
    )
)
_register(
    _Setting(
        name="match_diag_pattern",
        env="TWIN4BUILD_MATCH_DIAG_PATTERN",
        kind=str,
        default=None,
        doc="With match_diag_file: trace only the patterns whose id contains "
        "this text.",
    )
)
_register(
    _Setting(
        name="cuda_graph_sync_phases",
        env="T4B_CUDA_GRAPH_SYNC_PHASES",
        kind=bool,
        default=False,
        doc="A CUDA-graph capture synchronizes the device after every phase "
        "(warmup, record, replay, parity check), so an asynchronous CUDA fault "
        "is reported in the phase that launched it.",
    )
)
_register(
    _Setting(
        name="logger_strict",
        env="TWIN4BUILD_LOGGER_STRICT",
        kind=bool,
        default=False,
        doc="The logger raises when a status change names no logged line, "
        "instead of ignoring it.",
    )
)
