"""Idempotent model-graph rewiring for PI-CITS instances.

This module implements the operation that turns a freshly-translated
:class:`~twin4build.model.simulation_model.SimulationModel` containing
multi-candidate :class:`ControllerIdentificationPITorchSystem` instances
into a single-candidate, parameter-seeded model ready for estimation.

For each PI-CITS the operation:

    1. Initialises every connected sensor/setpoint/actuator-measurement
       :class:`SensorSystem` so that timeseries values are loaded from the
       database (or spreadsheet) into ``sensor.time_series_input.values``.
    2. Reads the loaded values as plain NumPy arrays.
    3. Scores every ``(sensor_i, setpoint_j)`` candidate pair against the
       actuator-measurement timeseries via
       :func:`~loop_classifier.score_pair`.
    4. If the best pair has sufficient confidence:
        - Removes the losing sensor/setpoint connections from the model
          graph via :meth:`SimulationModel.remove_connection`.
        - Re-numbers the surviving input-port indices to ``[0, 1, ...]``.
        - Collapses ``cits.n_sensors`` and ``cits.n_setpoints`` to 1.
        - Rebuilds the CITS internal components (resetting ``alpha_0``,
          ``beta_0``, ``gamma_0``, the BandGate, etc.).
        - Writes data-driven seeds (``kp``, ``Ti``, ``output_min``,
          ``output_max``, ``default_output_0``) onto the surviving
          ``candidate_0_0``.
        - Sets ``candidate_0_0.isReverse`` from ``slope >= 0`` so the
          simulator's internal ``err`` polarity matches the regression
          (twin4build's PI flips ``err = sp - fb`` to ``fb - sp`` when
          ``isReverse=False``; see ``pid_controller_system.do_step``).

The function is idempotent: calling it twice on the same model produces
the same final state.  On the second call every PI-CITS already has a
single connected sensor/setpoint pair, so the cross-product has one
member, the winner is trivially that pair, no further pruning happens, and
the seeds are recomputed from the same data to identical values.
"""

from __future__ import annotations

# Standard library imports
import datetime
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

# Third party imports
import numpy as np
import torch

# Local application imports
import twin4build.core as core
from twin4build.systems.controller.controller_identification.controller_identification_pi_torch_system import (
    ControllerIdentificationPITorchSystem,
)
from twin4build.systems.controller.controller_identification.loop_classifier import (
    ActuatorSeeds,
    GateSeeds,
    LoopScore,
    confidence_label,
    derive_actuator_seeds,
    derive_actuator_seeds_gmm,
    derive_gate_seeds_from_on_mask,
    score_pair,
)
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.utils.print_progress import LOGGER


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RewireReport:
    """Per-CITS outcome of a rewire pass.

    Attributes:
        cits_id: Component id of the PI-CITS that was processed.
        pruned: ``True`` if the rewire removed connections and collapsed the
            CITS to single-candidate; ``False`` if the CITS was left
            untouched (typically because the best pair had low confidence
            or the CITS had no candidates wired).
        confidence: One of ``"high"``, ``"medium"``, ``"low"``, ``"failed"``.
        winner: ``(sensor_id, setpoint_id)`` of the winning pair, or
            ``None`` if no pair scored above the confidence floor.
        actuator_id: The actuator-measurement sensor id used to compute
            the score (the downstream consumer of ``inputSignal``).
        score: The full :class:`LoopScore` of the winning pair (or the
            best available pair, when below confidence_threshold).
        kp_x0, Ti_x0, output_min_x0, output_max_x0, default_output_x0,
        is_reverse: Data-driven seeds applied to ``candidate_0_0``.
            ``None`` if the CITS was left untouched.
        candidate_scores: Mapping of every scored ``(sensor_id, setpoint_id)``
            pair to its R^2; useful for diagnostics.
        reason: Failure reason when ``pruned=False``.
    """

    cits_id: str
    pruned: bool
    confidence: str
    winner: Optional[Tuple[str, str]]
    actuator_id: Optional[str]
    score: Optional[LoopScore]
    kp_x0: Optional[float]
    Ti_x0: Optional[float]
    output_min_x0: Optional[float]
    output_max_x0: Optional[float]
    default_output_x0: Optional[float]
    is_reverse: Optional[bool]
    # Bounds derived by the rewire so callers can plumb them into their
    # estimator's (x0, lb, ub) tuples.  ``None`` when the CITS was untouched.
    kp_lb: Optional[float] = None
    kp_ub: Optional[float] = None
    Ti_lb: Optional[float] = None
    Ti_ub: Optional[float] = None
    candidate_scores: Dict[Tuple[str, str], float] = field(default_factory=dict)
    reason: Optional[str] = None
    # ----- GMM-based actuator decomposition (post-rewire diagnostic) -----
    # Sample-level "PI is active" label inferred from a 2-component GMM on
    # the actuator timeseries.  Available for every CITS that had loadable
    # actuator data, regardless of whether the rewire pruned.  ``None``
    # when the actuator series was missing or the GMM bailed out.
    kind: Optional[str] = None                  # damper / reheat / always_on / ambiguous
    bimodality: Optional[float] = None          # cluster separation in pooled-std units
    on_frac: Optional[float] = None             # fraction of samples flagged active
    # ----- Gate seeds derived from on_mask vs onOffSignal candidates -----
    # ``None`` when the rewire could not run the GMM (no actuator data) or
    # the on_mask was degenerate (all True / all False).  See
    # :class:`GateSeeds` for semantics.
    gate_seeds: Optional[GateSeeds] = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _rewire_pi_loops(
    model: core.SimulationModel,
    *,
    start_time: List[datetime.datetime],
    end_time: List[datetime.datetime],
    step_size: int,
    mode: str = "train",
    confidence_high: float = 0.5,
    confidence_low: float = 0.2,
    Ti_default: float = 1800.0,
    kp_decade_pad: float = 1.0,
    Ti_decade_pad: float = 0.5,
    Ti_lb_floor: float = 60.0,
    Ti_ub_ceil: float = 7200.0,
    kp_lb_floor: float = 0.05,
    kp_ub_ceil: float = 20.0,
    n_min_active: int = 50,
    sat_lo: float = 0.03,
    sat_hi: float = 0.97,
    sp_fb_corr_max: float = 0.95,
    fb_actuator_corr_max: float = 0.95,
    fb_sp_scale_max_offset: float = 30.0,
    fb_sp_median_tracking_max: float = 1.5,
) -> Dict[str, RewireReport]:
    """Run the data-driven rewire on every PI-CITS in ``model``.

    Internal helper.  End users should call
    :meth:`SimulationModel.rewire` or :meth:`Model.rewire`, which in
    turn dispatch here.  The leading underscore + missing ``__all__``
    entry reflect that the entry point is the model method, not this
    function.

    The function is mode-aware:

      * ``mode="train"``   -- intended for Stage-1 estimation.  Each
        CITS has its ``alpha_gate_{a}`` pinned to ``1.0`` so the
        BandGate fully gates the actuator on the active regime.
      * ``mode="simulate"`` -- intended for Stage-2 closed-loop
        physics simulation.  ``alpha_gate_{a}`` is pinned to ``0.0``
        so the gate is bypassed (``gate_input = 1 - 0 + 0 * gate =
        1``) and the PI passes through.  See
        :file:`controller_identification_torch_system.py:787-791` for
        the gate-mixing formula.

    In both modes the function also pins the frozen selection weights
    (``alpha_0`` / ``beta_0`` / ``gamma_0`` / optional ``beta_b_0``)
    to one-hot vectors of the right shape (post-rebuild ``[1.0]`` for
    pruned CITS, sensor-id-keyed one-hot for the rare untouched CITS)
    and sets ``gate_0.polarity = 1.0``.  These pins replace what used
    to be a downstream ``_prepare_stage2_cits_topology`` helper in
    user code.

    Args:
        model: The :class:`SimulationModel` containing PI-CITS instances.
        start_time, end_time, step_size: Forwarded to
            :meth:`SensorSystem.initialize` to load timeseries data.
        confidence_high, confidence_low: R^2 thresholds for the confidence
            ladder.

              * ``r2 >= confidence_high``  -> ``"high"``   : prune to the
                regression's winner pair and seed kp/Ti from the joint
                fit slope and integral coefficient.
              * ``confidence_low <= r2 < confidence_high`` -> ``"medium"``
                : same path as ``"high"`` (per Q4 of the design).
              * ``r2 < confidence_low``     -> ``"low"`` (or ``"failed"``
                when ``n_active < n_min_active``) : the regression slope
                is treated as noise.  The CITS is **still pruned** to
                the best-available pair (the candidate filters already
                eliminated obvious wrong ones), but kp / Ti are filled
                in with **heuristic** defaults
                (``kp = sqrt(kp_lb_floor * kp_ub_ceil) / 10``,
                ``Ti = Ti_default``).  ``output_min`` / ``output_max``
                / ``default_output`` are still derived from the GMM on
                the actuator timeseries.  The ``RewireReport`` will
                carry ``confidence in {"low", "failed"}`` and a
                ``"heuristic_seed (...)"`` reason so callers can spot
                the difference.
        Ti_default: Fallback ``Ti`` when the joint regression fails to
            identify it (purely-P loop or noisy data).
        kp_decade_pad: Half-width of the log-decade padding on ``kp``
            bounds around the data-driven ``kp_x0``.
        Ti_decade_pad: Same, for ``Ti``.
        Ti_lb_floor, Ti_ub_ceil: Hard bounds on ``Ti`` (in seconds),
            applied after the decade-pad clipping.
        kp_lb_floor, kp_ub_ceil: Hard bounds on ``kp``.
        n_min_active, sat_lo, sat_hi: Forwarded to
            :func:`loop_classifier.score_pair`.
        sp_fb_corr_max: Maximum allowed ``|Pearson(sp, fb)|`` for a setpoint
            candidate to be considered a real schedule setpoint.  When
            a BRICK ontology tags both the *active resolved control
            reference* (which closely tracks the zone temperature) and
            the *true schedule setpoint* under the same Brick class, the
            former produces ``e = sp - fb ~ 0`` and the joint regression
            ``Δu ~ Δe + e_mid`` is regression-on-noise: any winner has
            spurious slope.  Candidates with ``|corr(sp, fb)| >=
            sp_fb_corr_max`` are excluded *before* scoring so the
            schedule-setpoint candidate has a fair shot.  Default
            ``0.95`` rejects measurement clones while keeping legitimate
            setpoints (typical occupied-mode ``corr ~ 0.5--0.8``).
        fb_actuator_corr_max: Maximum allowed ``|Pearson(fb, u)|`` for a
            feedback-sensor candidate.  BRICK ontologies frequently wire
            *every* sensor near a VAV (zone temperature, supply-air
            temp, supply-air flow, percent-air-flow, ...) onto the
            CITS's ``sensorValue`` port.  When the actuator-feedback
            sensor (e.g. ``Zone_Percent_Air_Flow``) is one of those
            candidates it is essentially identical to the actuator
            timeseries ``u`` -- the regression ``Δu ~ Δ(-fb) + ...``
            is then a near-perfect identity with ``R² ~ 1`` and
            unphysical ``kp`` / ``Ti`` (clamped at the ``kp_lb_floor`` /
            ``Ti_ub_ceil`` bounds).  Candidates with ``|corr(fb, u)| >=
            fb_actuator_corr_max`` are excluded *before* scoring so the
            true zone-temperature feedback can win.  Default ``0.95``
            rejects actuator clones while keeping legitimate feedbacks
            (typical zone-temp loops have ``|corr(fb, u)| < 0.6``).
        fb_sp_scale_max_offset: Maximum allowed scale offset (in
            setpoint units, typically °C) for a feedback candidate to
            be considered physically compatible with the setpoint.
            For each pair the offset is the *largest* of three
            deviations:

              * ``|mean(fb) - mean(sp)|`` -- catches gross unit
                mismatches (CFM-scale flow ~300 vs temperature ~20).
              * ``max(fb) - max(sp)`` -- catches percent / count
                signals whose maximum overshoots any plausible
                setpoint (e.g. damper at 100% vs sp_max ~21 °C).
              * ``min(sp) - min(fb)`` -- catches signals that drop
                far below the lowest setpoint (a percent goes to
                0 while sp_min ~15 °C).

            For a temperature PI loop, all three should be small
            (a few K) since the controller cannot drive the feedback
            far outside the setpoint envelope.  When BRICK wires a
            flow / percent / cumulative counter onto ``sensorValue``
            (e.g. ``Zone_Percent_Air_Flow`` on a damper that toggles
            between 0% at night and 100% during the day), the mean
            offset alone may slip under the threshold, but the
            ``max(fb) - max(sp)`` term catches it (~80 vs sp_max=21).
            Candidates whose offset exceeds this threshold are
            excluded *before* scoring.  Default ``30.0`` keeps
            Supply-Air-Temp (~13 °C vs 21 °C setpoint = 8 K offset)
            but rejects Percent-Air-Flow (max ~100 → 80-unit
            envelope excursion) and Supply-Air-Flow (CFM-scale).
        fb_sp_median_tracking_max: Maximum allowed *median* ``|fb -
            sp|`` (in setpoint units, K) for a feedback candidate.
            Where ``fb_sp_scale_max_offset`` rejects gross *unit* /
            envelope mismatches (flow vs temp), this filter rejects
            *like-unit* candidates whose typical operating point is
            far from the setpoint -- the textbook example is wiring
            ``Supply_Air_Temp`` (downstream of a reheat valve, sits
            10-30 K above ``Zone_Air_Temp_Setpoint``) onto a CITS that
            should be closing on ``Zone_Air_Temp``.  The valve→supply
            mechanical link is so direct that the regression
            identifies a *spurious* high-R^2 negative slope, but the
            relationship is causality (u causes Δfb), not the
            closed-loop PI law (e drives Δu).  Median ``|fb - sp|`` during
            the active mask cleanly separates the regimes empirically
            observed on the bldg1 / Mortar dataset:

              * true PI feedback (Zone_Air_Temp vs Setpoint):
                median ~ 0.2 - 0.7 K (controller keeps them close;
                comfort band sits well under 1 K most of the time).
              * downstream-of-actuator (Supply_Air_Temp vs zone
                setpoint, even on a *mildly*-reheated system where the
                AHU supply is already close to occupied setpoint):
                median ~ 2 - 4 K (the valve has to clear the AHU/zone
                offset every time it opens, never gets there).

            On heavily-reheated systems the downstream gap is much
            larger (10-30 K), but the lower bound is what determines
            the discriminator: as long as the threshold sits *above*
            the worst-case legitimate tracking error and *below* the
            best-case downstream gap, the filter does the right thing.
            Default ``1.5`` K leaves an honest tracking-loop alone
            even during disturbances but rejects feedbacks that the
            CITS does not actually close on.

    Returns:
        Mapping of PI-CITS id to its :class:`RewireReport`.
    """
    pi_cits_list = [
        c
        for c in model.components.values()
        if isinstance(c, ControllerIdentificationPITorchSystem)
    ]

    LOGGER.info(
        f"[REWIRE] Found {len(pi_cits_list)} PI-CITS components to rewire"
    )

    if not pi_cits_list:
        return {}

    # Step 0: ensure every CITS is built before we start scoring.
    #
    # Several downstream helpers (``_collect_on_off_slot_signals``,
    # ``_populate_on_off_signal_norm_bounds``,
    # ``_populate_gate_seeds_from_on_mask``) read
    # ``cits.n_on_off_signals`` to know how many gate-input slots to
    # walk and silently fall back to ``0`` when the attribute is
    # ``None``.  On the documented call path
    # (``translate → set_dbconfigs → set_transformations → rewire →
    # load``) the CITS has never been touched between the translator
    # and the rewire, so every ``n_*`` is still ``None`` and the entire
    # GMM-based gate-seed pipeline gets bypassed -- without warning.
    # That manifested as the rewire producing materially *worse* seeds
    # (no gate-mode mask intersection in the kp/Ti regression, no
    # data-driven ``gate_0.threshold`` / ``gate_0.band``, no
    # ``gamma_gate_0`` selection) on un-built CITS than on otherwise-
    # identical pre-built ones.
    #
    # ``_initialize`` does this same n_* derivation at load time
    # (controller_identification_torch_system.py:757-783); we run it
    # earlier here so the rewire's scoring sees a consistent CITS state.
    # ``_rewire_one`` later calls ``_build_components`` again post-
    # pruning to collapse ``n_sensors = n_setpoints = 1``, so this
    # early build does not add work -- it just moves the
    # "construct gates + candidates" step to before scoring.
    for cits in pi_cits_list:
        if cits._built:
            continue
        n_s = cits.get_n_v_from_connections("sensorValue")
        n_sp = cits.get_n_v_from_connections("setpointValue")
        n_oo = cits.get_n_v_from_connections("onOffSignal")
        n_act = cits._get_n_actuators_from_connections()
        if n_s is not None:
            cits.n_sensors = n_s
        if n_sp is not None:
            cits.n_setpoints = n_sp
        if n_oo is not None:
            cits.n_on_off_signals = n_oo
        if n_act is not None:
            cits.n_actuators = n_act
        if (
            cits.n_sensors is None
            or cits.n_setpoints is None
            or cits.n_on_off_signals is None
        ):
            LOGGER.warning(
                f"[REWIRE] {cits.id}: cannot pre-build (n_sensors="
                f"{cits.n_sensors}, n_setpoints={cits.n_setpoints}, "
                f"n_on_off_signals={cits.n_on_off_signals}); "
                f"gate-seed pipeline will degrade for this CITS."
            )
            continue
        cits._build_components()

    # Step 1: collect every sensor that we need to read data from.  This
    # includes the input sensors (sensorValue + setpointValue +
    # onOffSignal) and the downstream actuator-measurement sensor.
    sensors_to_init = _collect_sensors(pi_cits_list)
    LOGGER.info(
        f"[REWIRE] Initialising {len(sensors_to_init)} sensors for data load"
    )
    # SensorSystem.initialize expects ``start_time``, ``end_time`` and
    # ``step_size`` as parallel lists (one entry per simulation batch).
    # Callers typically pass ``start_time`` / ``end_time`` as lists (one
    # entry per period) but a bare ``step_size`` int.  Normalise all three
    # to lists of the same length so the sensor sees consistent shapes.
    init_start = start_time if isinstance(start_time, list) else [start_time]
    init_end = end_time if isinstance(end_time, list) else [end_time]
    n_batches = max(len(init_start), len(init_end))
    if isinstance(step_size, list):
        init_step = list(step_size)
    else:
        init_step = [step_size]
    if len(init_step) == 1 and n_batches > 1:
        init_step = init_step * n_batches
    for s in sensors_to_init:
        try:
            s.initialize(
                start_time=init_start,
                end_time=init_end,
                step_size=init_step,
            )
        except Exception as ex:  # noqa: BLE001
            LOGGER.warning(
                f"[REWIRE] Sensor '{s.id}' failed to initialise: {ex}"
            )

    # Step 2: per-CITS rewire.  Pruning calls ``cits._build_components()``
    # which rebuilds the ``on_off_signal_norm_min`` /
    # ``on_off_signal_norm_max`` buffers at the (unchanged)
    # ``n_on_off_signals`` size with default identity values, so we
    # must populate the gate-input normalisation bounds *after* the
    # rewire loop -- otherwise a pruned CITS silently reverts to
    # ``[0, 1]`` bounds while the runtime signal is still in physical
    # units (e.g. ~21 deg C), pushing the gate input far above the
    # ``[0, 1]`` band and forcing the gate to ~0 for the entire
    # trajectory.  That single ordering bug used to manifest as PI
    # loops collapsing to ``output = default_output`` (constant
    # prediction with ``RMSE == std(u)``), with ``kp`` / ``Ti`` /
    # ``output_min`` / band parameters showing zero gradient.
    h = float(step_size)
    reports: Dict[str, RewireReport] = {}
    for cits in pi_cits_list:
        try:
            report = _rewire_one(
                cits=cits,
                model=model,
                h=h,
                confidence_high=confidence_high,
                confidence_low=confidence_low,
                Ti_default=Ti_default,
                kp_decade_pad=kp_decade_pad,
                Ti_decade_pad=Ti_decade_pad,
                Ti_lb_floor=Ti_lb_floor,
                Ti_ub_ceil=Ti_ub_ceil,
                kp_lb_floor=kp_lb_floor,
                kp_ub_ceil=kp_ub_ceil,
                n_min_active=n_min_active,
                sat_lo=sat_lo,
                sat_hi=sat_hi,
                sp_fb_corr_max=sp_fb_corr_max,
                fb_actuator_corr_max=fb_actuator_corr_max,
                fb_sp_scale_max_offset=fb_sp_scale_max_offset,
                fb_sp_median_tracking_max=fb_sp_median_tracking_max,
            )
        except Exception as ex:  # noqa: BLE001
            import traceback as _tb
            LOGGER.warning(
                f"[REWIRE] CITS '{cits.id}' rewire failed: {ex}\n"
                + _tb.format_exc()
            )
            report = RewireReport(
                cits_id=cits.id,
                pruned=False,
                confidence="failed",
                winner=None,
                actuator_id=None,
                score=None,
                kp_x0=None,
                Ti_x0=None,
                output_min_x0=None,
                output_max_x0=None,
                default_output_x0=None,
                is_reverse=None,
                reason=f"exception: {ex}",
            )
        reports[cits.id] = report

    # Step 3: populate per-CITS setpoint-signal min/max buffers from the
    # loaded sensor data so the gate input is in normalised ``[0, 1]``
    # units (decouples ``gate_0.threshold`` / ``gate_0.band`` x0 from the
    # physical units of the wired setpoint sensors).  Operates on the
    # ``onOffSignal`` input port -- which the rewire never prunes -- so
    # the gate's input space is preserved across rewire (the schedule
    # remains visible to the gate even when the rewire winner picks a
    # different setpoint for the PI error term).  Deferred to after
    # pruning to be safe; ``onOffSignal`` connectivity is unchanged by
    # rewire, but ``_build_components`` recreates the buffers when the
    # CITS is rebuilt so we still need to populate them post-rebuild.
    _populate_on_off_signal_norm_bounds(pi_cits_list)

    # Step 4: data-driven gate seeds.  Run a 2-component GMM on each
    # actuator timeseries to obtain a sample-level "PI is active" mask,
    # then rank every wired ``onOffSignal`` slot by ROC AUC of its
    # normalised value vs the mask.  The winner gives ``gamma_gate``
    # (one-hot when AUC is decisive, soft when several slots tie),
    # ``gate.threshold`` and ``gate.band`` (active-range quantiles of
    # the winning slot, padded by a margin).  Must run after step 3 so
    # the per-slot normalisation bounds are already populated; the
    # gate-seed function uses the *same* normalisation the CITS forward
    # pass does so the seeds land in their physically meaningful basin.
    gate_results = _populate_gate_seeds_from_on_mask(pi_cits_list)
    for cid, (kind, bimodality, on_frac, gate_seeds) in gate_results.items():
        rep = reports.get(cid)
        if rep is None:
            continue
        rep.kind = kind
        rep.bimodality = bimodality
        rep.on_frac = on_frac
        rep.gate_seeds = gate_seeds

    # Step 5: mode-aware frozen-pin pass.  Sets one-hot selection
    # weights, gate polarity, and the gate-activity scalar
    # ``alpha_gate_{a}`` according to ``mode``.  Idempotent: a
    # subsequent call with the same ``mode`` produces the same final
    # state.
    _pin_frozen_cits_state(pi_cits_list, mode=mode)

    return reports


# Backward-compat alias for direct callers that import the function
# from the module.  New code should reach this via
# :meth:`SimulationModel.rewire` / :meth:`Model.rewire`; the alias is
# kept so diagnostic / loss-landscape example scripts under
# ``twin4build/examples`` keep importing the function from the
# module by name without modification.  The package-level export
# (``twin4build.systems.rewire_pi_loops``) is intentionally dropped.
rewire_pi_loops = _rewire_pi_loops


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_sensors(
    pi_cits_list: List[ControllerIdentificationPITorchSystem],
) -> List[SensorSystem]:
    """Return the deduplicated list of every sensor connected (in or out)
    to any PI-CITS in ``pi_cits_list``.
    """
    seen: Dict[str, SensorSystem] = {}
    for cits in pi_cits_list:
        # Inputs: sensorValue + setpointValue + onOffSignal connection
        # points.  ``onOffSignal`` is the gate-input bus -- we still
        # need its upstream sensor data initialised so
        # :func:`_populate_on_off_signal_norm_bounds` can read finite
        # min/max for the per-slot normalisation.
        for cp in cits.connects_at:
            if cp.input_port not in ("sensorValue", "setpointValue", "onOffSignal"):
                continue
            for conn in cp.connects_system_through:
                sender = conn.connects_system
                if isinstance(sender, SensorSystem):
                    seen[sender.id] = sender
        # Outputs: actuator-measurement sensors downstream of inputSignal.
        for conn in cits.connected_through:
            if conn.output_port != "inputSignal":
                continue
            for cp in conn.connects_system_at:
                receiver = cp.connection_point_of
                if isinstance(receiver, SensorSystem):
                    seen[receiver.id] = receiver
    return list(seen.values())


def _sensor_timeseries(sensor: SensorSystem) -> Optional[np.ndarray]:
    """Extract the loaded timeseries from a :class:`SensorSystem` as a 1D
    NumPy array, concatenating across batches.

    Returns the values exactly as stored in
    :attr:`SensorSystem.time_series_input.values` -- i.e. in *physical*
    units when the sensor's ``_transformation`` was set *before* the
    sensor was initialised (the typical case for setpoint / zone-temp
    sensors, whose unit conversion runs inside
    :meth:`TimeSeriesInputSystem.initialize`).

    For actuator-command sensors whose ``_transformation`` is wired
    *after* ``initialize()`` (translator examples set the percent->fraction
    transform on the downstream ``inputSignal`` consumer only at the very
    end), the stored values stay raw (``[0, 100]``).  Callers that need
    the actuator in fraction units should follow this with
    :func:`_maybe_rescale_percent`, which auto-detects that case and
    divides by 100.

    Important: do **not** re-apply ``sensor._transformation`` here.  Doing
    so would double-convert sensors whose transform already ran at load
    time (e.g. mapping a stored 21 degC through ``(x - 32) * 5/9`` again
    gives ``-6.1`` -- a silent numerical disaster that corrupts every
    downstream consumer that compares values against physical thresholds).

    Returns ``None`` if the sensor was not initialised or has no data.
    """
    tsi = getattr(sensor, "time_series_input", None)
    if tsi is None:
        return None
    values = getattr(tsi, "values", None)
    if values is None:
        return None
    arr = values.detach().cpu().numpy()
    if arr.ndim == 3:
        flat = arr[..., 0].T.reshape(-1)
    elif arr.ndim == 2:
        flat = arr.reshape(-1)
    else:
        flat = arr.reshape(-1)
    return flat.astype(np.float64)


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Pearson correlation between two equal-length arrays, with NaN /
    constant-input guards.

    Returns ``None`` when either input is constant (zero variance) or
    has fewer than two finite samples; the caller then treats the
    candidate as "cannot be ruled out as a measurement clone" and
    proceeds to the regular ``score_pair`` path.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return None
    a, b = a[mask], b[mask]
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-12 or sb < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class ContinuityStats:
    """Cheap continuity diagnostics used to distinguish a real continuous
    measurement from a quantised / scheduled override.

    Attributes:
        n_unique: number of distinct (rounded to 3 decimals) values in
            the timeseries.  A real continuous sensor typically has
            hundreds; a 1-2-degF override has < 30.
        frac_on_F_grid: fraction of finite samples that fall on (or
            very close to) an integer-Fahrenheit grid after converting
            from Celsius.  Schedules / overrides are usually authored in
            integer °F and survive the C->F->C round-trip; real sensor
            histories have this fraction near zero.
        frac_zero_diffs: fraction of consecutive identical samples.
            Step-like overrides have most diffs = 0; live measurements
            have very few.
    """

    n_unique: int
    frac_on_F_grid: float
    frac_zero_diffs: float

    @property
    def quality_score(self) -> float:
        """Single scalar combining ``n_unique`` and ``frac_on_F_grid``.

        Higher = more measurement-like.  Used to rank clone candidates:
        when two signals correlate at >= ``sp_fb_corr_max`` we keep the
        one with the higher quality score and drop / repurpose the
        other.
        """
        return float(self.n_unique) * (1.0 - float(self.frac_on_F_grid))


def _score_continuity(ts: np.ndarray) -> ContinuityStats:
    """Compute :class:`ContinuityStats` for one signal.

    Robust to NaN / Inf inputs and to empty / tiny arrays (returns the
    "looks dead" defaults so the rest of the pipeline rejects it
    naturally).
    """
    arr = np.asarray(ts, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return ContinuityStats(n_unique=0, frac_on_F_grid=0.0, frac_zero_diffs=1.0)
    n_unique = int(np.unique(np.round(finite, 3)).size)
    arr_f = finite * 9.0 / 5.0 + 32.0
    frac_F = float(np.mean(np.abs(arr_f - np.round(arr_f)) < 0.05))
    if finite.size > 1:
        d = np.diff(finite)
        frac_zero = float(np.mean(np.abs(d) < 1e-6))
    else:
        frac_zero = 1.0
    return ContinuityStats(
        n_unique=n_unique, frac_on_F_grid=frac_F, frac_zero_diffs=frac_zero
    )


def _collect_on_off_slot_signals(
    cits: ControllerIdentificationPITorchSystem,
    *,
    pad_constant: float = 0.01,
) -> Tuple[List[Optional[np.ndarray]], np.ndarray, np.ndarray, int]:
    """Walk a CITS's ``onOffSignal`` connections and return per-slot data.

    For each of the ``n_on_off_signals`` slots, look up the wired
    :class:`SensorSystem` (if any), pull its loaded timeseries, and
    derive empirical ``[lo, hi]`` normalisation bounds the same way
    :func:`_populate_on_off_signal_norm_bounds` does -- but **without**
    writing anything back onto the CITS.  This makes it safe to call
    from :func:`_rewire_one` *before* the post-rewire bounds-population
    step has run; the caller gets a self-contained snapshot of the
    schedule signals plus their normalisation frame.

    Used by :func:`_rewire_one` to build the gate-mode predicate that
    masks the kp/Ti regression to "PI is in active control mode"
    samples (intersection of GMM(u) on_mask and the BandGate's active
    region on the AUC-winning slot).  The same logic is later re-run
    by :func:`_populate_gate_seeds_from_on_mask` to produce the
    *persisted* gate seeds that the runtime sigmoid uses; both paths
    therefore share a single, consistent definition of "active mode"
    derived from the same data-driven AUC ranking.

    Args:
        cits: The PI-CITS component.  Must have its upstream sensors
            already initialised (``time_series_input.values``
            populated) so each slot's signal is recoverable.
        pad_constant: Half-width applied symmetrically when a slot's
            signal is constant (range < 1e-6) so the normalised range
            stays non-degenerate.  Default ``0.01`` is small relative
            to typical setpoint swings (~5 K).

    Returns:
        Tuple ``(slot_signals, oo_min, oo_max, n_oo)``:

        * ``slot_signals``: list of length ``n_oo`` holding the raw
          signal arrays per slot, or ``None`` if the slot is unwired
          / has no data.
        * ``oo_min``, ``oo_max``: ``np.ndarray`` of length ``n_oo``
          with empirical bounds (defaults ``0`` / ``1`` for unwired
          slots).
        * ``n_oo``: number of slots (``cits.n_on_off_signals``).  May
          be ``0`` for CITS without a gate.
    """
    n_oo = int(getattr(cits, "n_on_off_signals", 0) or 0)
    slot_signals: List[Optional[np.ndarray]] = [None] * n_oo
    oo_min = np.zeros(n_oo, dtype=np.float64)
    oo_max = np.ones(n_oo, dtype=np.float64)
    if n_oo <= 0:
        return slot_signals, oo_min, oo_max, n_oo

    for cp in cits.connects_at:
        if cp.input_port != "onOffSignal":
            continue
        for conn in cp.connects_system_through:
            idx = cp.input_port_index.get(conn)
            if idx is None or idx < 0 or idx >= n_oo:
                continue
            sender = conn.connects_system
            if not isinstance(sender, SensorSystem):
                continue
            arr = _sensor_timeseries(sender)
            if arr is None or arr.size == 0:
                continue
            finite_arr = arr[np.isfinite(arr)]
            if finite_arr.size == 0:
                continue
            slot_signals[idx] = arr
            lo = float(np.min(finite_arr))
            hi = float(np.max(finite_arr))
            if hi - lo < 1e-6:
                lo -= pad_constant
                hi += pad_constant
            oo_min[idx] = lo
            oo_max[idx] = hi
    return slot_signals, oo_min, oo_max, n_oo


def _populate_on_off_signal_norm_bounds(
    pi_cits_list: List[ControllerIdentificationPITorchSystem],
    *,
    pad_constant: float = 0.5,
) -> None:
    """Compute per-onOffSignal min/max from initialised sensor data and
    write onto each PI-CITS as ``on_off_signal_norm_min`` /
    ``on_off_signal_norm_max``.

    Maps each onOffSignal slot that flows into the gate to a unit-free
    ``[0, 1]`` range, decoupling ``gate_0.threshold`` / ``gate_0.band``
    x0 from physical units.  The CITS forward pass divides
    ``(value - on_off_signal_norm_min) / (on_off_signal_norm_max - on_off_signal_norm_min)``
    before applying the ``gamma_gate``-weighted sum.

    Slots without wired data (or with constant signals) get a degenerate
    ``[v - pad_constant, v + pad_constant]`` range so the gate input
    remains finite.  Defaults (identity transform: ``[0, 1]``) are kept
    for any slot whose connected sensor has no usable data.

    Args:
        pi_cits_list: PI-CITS instances whose upstream sensors have
            already been initialised (``time_series_input.values``
            populated).
        pad_constant: Half-width to use when a signal is constant (so
            the normalised range degenerates to a sensible non-zero
            interval).
    """
    for cits in pi_cits_list:
        n_oo = int(getattr(cits, "n_on_off_signals", 0) or 0)
        if n_oo <= 0:
            continue

        # Start from existing tensor on the CITS (defaults to [0, 1]).
        # Build fresh numpy arrays so we can edit per-slot then write
        # back.
        oo_min = (
            cits.on_off_signal_norm_min.detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            .copy()
            if hasattr(cits, "on_off_signal_norm_min")
            else np.zeros(n_oo, dtype=np.float64)
        )
        oo_max = (
            cits.on_off_signal_norm_max.detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            .copy()
            if hasattr(cits, "on_off_signal_norm_max")
            else np.ones(n_oo, dtype=np.float64)
        )
        if oo_min.size != n_oo:
            oo_min = np.zeros(n_oo, dtype=np.float64)
        if oo_max.size != n_oo:
            oo_max = np.ones(n_oo, dtype=np.float64)

        # Walk every onOffSignal connection point and resolve its slot.
        seen_slots: set[int] = set()
        for cp in cits.connects_at:
            if cp.input_port != "onOffSignal":
                continue
            for conn in cp.connects_system_through:
                idx = cp.input_port_index.get(conn)
                if idx is None or idx < 0 or idx >= n_oo:
                    continue
                sender = conn.connects_system
                if not isinstance(sender, SensorSystem):
                    continue
                arr = _sensor_timeseries(sender)
                if arr is None or arr.size == 0:
                    continue
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    continue
                lo = float(np.min(finite))
                hi = float(np.max(finite))
                if hi - lo < 1e-6:
                    # Constant signal -- pad symmetrically so
                    # normalisation produces ~0.5 everywhere instead of
                    # NaN/Inf.
                    lo = lo - pad_constant
                    hi = hi + pad_constant
                oo_min[idx] = lo
                oo_max[idx] = hi
                seen_slots.add(idx)

        cits.on_off_signal_norm_min = torch.tensor(oo_min, dtype=torch.float64)
        cits.on_off_signal_norm_max = torch.tensor(oo_max, dtype=torch.float64)

        bounds_str = ", ".join(
            f"slot{j}=[{oo_min[j]:.3f}, {oo_max[j]:.3f}]"
            for j in sorted(seen_slots)
        ) or "none"
        LOGGER.info(
            f"[REWIRE] onOffSignal norm bounds for '{cits.id}': {bounds_str}"
        )


def _populate_gate_seeds_from_on_mask(
    pi_cits_list: List[ControllerIdentificationPITorchSystem],
    *,
    sharpness_beta: float = 8.0,
) -> Dict[str, Tuple[Optional[str], Optional[float], Optional[float], Optional[GateSeeds]]]:
    """Per-CITS: run GMM(actuator) -> on_mask -> rank onOffSignal slots.

    For each PI-CITS:

    1. Resolve its downstream actuator-measurement sensor and fetch the
       loaded timeseries (auto-rescaling percent->fraction when the
       transform was set late by the example).
    2. Run :func:`derive_actuator_seeds_gmm` to obtain ``on_mask`` plus
       the GMM ``kind`` and ``bimodality`` diagnostics.
    3. Walk the CITS's ``onOffSignal`` connection points and gather a
       per-slot list of raw signal arrays (one per slot index).
    4. Read the per-slot normalisation bounds previously written by
       :func:`_populate_on_off_signal_norm_bounds` so the gate seeds
       are in the same frame the CITS forward pass uses.
    5. Call :func:`derive_gate_seeds_from_on_mask` to compute
       ``gamma_gate_x0``, ``gate_threshold_x0`` and ``gate_band_x0``.
    6. Write the seeds onto the CITS components:
       - ``cits.gamma_gate_0`` (Parameter, vector of length ``n_oo``)
       - ``cits.gate_0.threshold`` (Parameter, scalar)
       - ``cits.gate_0.band`` (Parameter, scalar)

    The function never raises -- it logs diagnostics and falls through
    on any failure mode (degenerate GMM, missing actuator data, etc.)
    so a single problematic CITS does not block seeding for the rest.

    Args:
        pi_cits_list: PI-CITS components (same list passed to
            :func:`_populate_on_off_signal_norm_bounds`).
        sharpness_beta: Softmax temperature for the gamma_gate seed.
            Higher = more peaked.  Default ``8`` makes a 0.95 vs 0.5
            AUC split essentially one-hot.

    Returns:
        Mapping ``cits_id -> (kind, bimodality, on_frac, gate_seeds)``
        for every CITS that produced a result; missing CITS get a
        ``None`` 4-tuple.  Caller plumbs these into
        :class:`RewireReport`.
    """
    results: Dict[
        str,
        Tuple[Optional[str], Optional[float], Optional[float], Optional[GateSeeds]],
    ] = {}
    for cits in pi_cits_list:
        n_oo = int(getattr(cits, "n_on_off_signals", 0) or 0)
        if n_oo <= 0:
            results[cits.id] = (None, None, None, None)
            continue

        # 1+2: actuator series -> GMM on_mask.
        actuator = _resolve_actuator_measurement(cits)
        if actuator is None:
            LOGGER.info(
                f"[REWIRE] {cits.id}: gate seeding skipped (no actuator)"
            )
            results[cits.id] = (None, None, None, None)
            continue
        u = _sensor_timeseries(actuator)
        if u is None or u.size == 0:
            LOGGER.info(
                f"[REWIRE] {cits.id}: gate seeding skipped (no actuator data)"
            )
            results[cits.id] = (None, None, None, None)
            continue
        u, _ = _maybe_rescale_percent(u)
        bimodal = derive_actuator_seeds_gmm(u)
        on_frac = float(bimodal.on_mask.sum()) / max(bimodal.on_mask.size, 1)

        # 3: per-slot onOffSignal arrays.  We need the same raw values
        # we used for the norm-bounds, indexed by the connection's slot.
        slot_signals: list = [None] * n_oo
        for cp in cits.connects_at:
            if cp.input_port != "onOffSignal":
                continue
            for conn in cp.connects_system_through:
                idx = cp.input_port_index.get(conn)
                if idx is None or idx < 0 or idx >= n_oo:
                    continue
                sender = conn.connects_system
                if not isinstance(sender, SensorSystem):
                    continue
                arr = _sensor_timeseries(sender)
                if arr is None:
                    continue
                slot_signals[idx] = arr

        # 4: per-slot norm bounds from the CITS buffers.
        oo_min = (
            cits.on_off_signal_norm_min.detach().cpu().numpy().astype(np.float64)
            if hasattr(cits, "on_off_signal_norm_min")
            else np.zeros(n_oo, dtype=np.float64)
        )
        oo_max = (
            cits.on_off_signal_norm_max.detach().cpu().numpy().astype(np.float64)
            if hasattr(cits, "on_off_signal_norm_max")
            else np.ones(n_oo, dtype=np.float64)
        )

        # 5: derive seeds.
        gate_seeds = derive_gate_seeds_from_on_mask(
            bimodal.on_mask,
            slot_signals,
            oo_min,
            oo_max,
            sharpness_beta=sharpness_beta,
        )

        # 6: write onto CITS.  ``gamma_gate_0`` is sized
        # ``n_on_off_signals``; ``gate_0.threshold`` / ``gate_0.band``
        # are scalars.  Existing bounds on the Parameter objects are
        # preserved -- the example's ``set_parameters`` call later
        # may overwrite the bounds explicitly, but the x0 we write
        # here is what the optimizer starts from.
        try:
            if hasattr(cits, "gamma_gate_0"):
                cits.gamma_gate_0.set(
                    torch.tensor(gate_seeds.gamma_gate_x0, dtype=torch.float64),
                    normalized=False,
                )
        except Exception as ex:  # noqa: BLE001
            LOGGER.warning(
                f"[REWIRE] {cits.id}: failed to write gamma_gate_0 seed: {ex}"
            )
        gate = getattr(cits, "gate_0", None)
        if gate is not None:
            for attr, val in (
                ("threshold", gate_seeds.gate_threshold_x0),
                ("band", gate_seeds.gate_band_x0),
            ):
                p = getattr(gate, attr, None)
                if p is None:
                    continue
                try:
                    p.set(
                        torch.tensor(float(val), dtype=torch.float64),
                        normalized=False,
                    )
                except Exception as ex:  # noqa: BLE001
                    LOGGER.warning(
                        f"[REWIRE] {cits.id}: failed to write gate_0.{attr} "
                        f"seed (val={val}): {ex}"
                    )

        # Diagnostic log: per-CITS one-liner so the AUC scores and
        # the chosen gate seeds are easy to grep.  Polarity sign is
        # spelled out so a flipped schedule is visible at a glance.
        auc_str = ", ".join(
            f"slot{i}={a:.2f}" for i, a in enumerate(gate_seeds.auc_per_slot)
        )
        LOGGER.info(
            f"[REWIRE] {cits.id}: kind={bimodal.kind} on_frac={on_frac:.2f} "
            f"bimodal={bimodal.bimodality:.2f}  gate-conf={gate_seeds.confidence} "
            f"winner=slot{gate_seeds.winner_slot} "
            f"polarity={'+' if gate_seeds.winner_polarity > 0 else '-'} "
            f"AUC=[{auc_str}]  "
            f"T_lo={gate_seeds.gate_threshold_x0:.3f} "
            f"band={gate_seeds.gate_band_x0:.3f}"
        )
        results[cits.id] = (
            bimodal.kind,
            float(bimodal.bimodality),
            on_frac,
            gate_seeds,
        )

    return results


def _maybe_rescale_percent(arr: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Auto-scale a ``[0, 100]`` percent series down to ``[0, 1]`` fraction.

    Used when ``_transformation`` is not set on the actuator-measurement
    sensor (the common case before the example wires up actuator transforms
    inside its CITS post-processing loop).  Detection rule: if the finite
    max exceeds 1.5 and is at most ~120, treat as percent.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr, False
    mx = float(np.nanmax(finite))
    if 1.5 < mx <= 120.0:
        return arr / 100.0, True
    return arr, False


def _resolve_actuator_measurement(
    cits: ControllerIdentificationPITorchSystem,
) -> Optional[SensorSystem]:
    """Return the first actuator-measurement :class:`SensorSystem` downstream
    of ``cits.inputSignal``.  Returns ``None`` if none are wired.
    """
    for conn in cits.connected_through:
        if conn.output_port != "inputSignal":
            continue
        for cp in conn.connects_system_at:
            recv = cp.connection_point_of
            if isinstance(recv, SensorSystem):
                return recv
    return None


def _collect_input_signals(
    cits: ControllerIdentificationPITorchSystem,
) -> Tuple[Dict[str, Tuple[SensorSystem, Any]], Dict[str, Tuple[SensorSystem, Any]]]:
    """Collect the wired sensors and setpoints for one CITS.

    Returns:
        ``(sensors, setpoints)`` where each is a dict mapping
        ``sender_id -> (sender_component, connection_object)``.  The
        connection object is needed for pruning later.
    """
    sensors: Dict[str, Tuple[SensorSystem, Any]] = {}
    setpoints: Dict[str, Tuple[SensorSystem, Any]] = {}
    for cp in cits.connects_at:
        if cp.input_port == "sensorValue":
            target = sensors
        elif cp.input_port == "setpointValue":
            target = setpoints
        else:
            continue
        for conn in cp.connects_system_through:
            sender = conn.connects_system
            if isinstance(sender, SensorSystem) and sender.id not in target:
                target[sender.id] = (sender, conn)
    return sensors, setpoints


def _disconnect_sender_from_cits(
    sender: SensorSystem,
    cits: ControllerIdentificationPITorchSystem,
    model: core.SimulationModel,
    *,
    ports: Tuple[str, ...] = ("sensorValue", "setpointValue", "onOffSignal"),
) -> List[str]:
    """Remove every connection from ``sender`` into ``cits`` on the
    listed input ports.  Idempotent: silently skips ports that are
    not currently wired.  Returns the list of ports actually removed
    (for logging).
    """
    removed: List[str] = []
    for port in ports:
        # Check membership before calling remove_connection -- the
        # SimulationModel implementation raises if the connection
        # isn't there, and we want this helper to be a no-op for
        # missing wirings.
        wired = False
        for cp in cits.connects_at:
            if cp.input_port != port:
                continue
            for conn in cp.connects_system_through:
                if conn.connects_system is sender:
                    wired = True
                    break
            if wired:
                break
        if not wired:
            continue
        try:
            model.remove_connection(
                sender_component=sender,
                receiver_component=cits,
                output_port="measuredValue",
                input_port=port,
            )
            removed.append(port)
        except (ValueError, AttributeError) as ex:
            LOGGER.warning(
                f"[REWIRE] {cits.id}: failed to remove '{sender.id}' from "
                f"'{port}': {ex}"
            )
    return removed


def _maybe_swap_broken_sensor(
    *,
    cits: ControllerIdentificationPITorchSystem,
    model: core.SimulationModel,
    sensors_dict: Dict[str, Tuple[SensorSystem, Any]],
    setpoints_dict: Dict[str, Tuple[SensorSystem, Any]],
    quality_swap_ratio: float = 5.0,
    min_unique_for_good_sp: int = 30,
    max_unique_for_broken_fb: int = 30,
    swap_corr_min: float = 0.5,
) -> List[Tuple[str, str]]:
    """Detect and physically repair the "broken sensor + real measurement
    on setpoint port" data pathology, then return the list of swapped
    pairs.

    Background
    ----------
    Some BMS data pipelines historize the CITS feedback channel as a
    heavily quantised override (a handful of round-Fahrenheit values,
    almost all consecutive samples identical, occasional spurious
    extreme values) while the *true* continuous zone-temperature
    measurement appears on the *setpoint* side because the BRICK
    ontology aliases two ``Zone_Air_Temperature`` points -- one
    BMS-flagged ``measuredValue`` and one BMS-flagged ``Setpoint`` --
    both pointing at the same underlying sensor.  Concretely (mortar
    bldg1 RM112)::

        sensor-side    Zone_Air_Temp        : n_unique=10,  frac_on_°F=0.98,
                                              range=[15.56, 29.44] (the
                                              29.44 °C is a spurious
                                              override outlier)
        setpoint-side  Zone_Air_Control_Temp: n_unique=148, frac_on_°F=0.17,
                                              range=[16.16, 23.43]

    The two are physically the same zone temperature, but quantization
    plus outlier spikes in the broken side drop their Pearson
    correlation below ``sp_fb_corr_max=0.95`` (the clone-exclusion
    threshold designed to catch literal duplicates).  Without this
    helper, the rewire's downstream regression sees ``Δfb ≈ 0``
    everywhere on the broken sensor, ``kp`` collapses to its floor,
    ``Ti`` to its ceiling, and the simulation runs open-loop on a
    stale override.

    Detection criteria (all must hold for a (fb, sp) pair)
    ------------------------------------------------------
    * ``|corr(fb, sp)| >= swap_corr_min`` -- they track the same
      physical signal (loose threshold; quantization noise allowed).
    * ``n_unique(fb) <= max_unique_for_broken_fb`` -- the sensor-side
      really is a quantised / scheduled override, not a fine
      measurement.
    * ``n_unique(sp) >= min_unique_for_good_sp`` -- the setpoint-side
      really is a continuous measurement, not a scheduled setpoint.
    * ``quality(sp) >= quality_swap_ratio * quality(fb)`` -- the
      asymmetry is large (default 5x).

    Repair
    ------
    1. Disconnect the broken sensor from every input port of the CITS
       (sensorValue, setpointValue, onOffSignal -- whichever it touches).
    2. Disconnect the good signal from setpointValue and onOffSignal.
    3. Add a fresh connection sending the good signal's
       ``measuredValue`` into the CITS's ``sensorValue`` port.

    After this, the caller must re-run :func:`_collect_input_signals`
    so the regular scoring loop sees a clean (fb, sp) pool.
    """
    if not sensors_dict or not setpoints_dict:
        return []

    fb_quality: Dict[str, ContinuityStats] = {}
    sp_quality: Dict[str, ContinuityStats] = {}
    fb_ts: Dict[str, np.ndarray] = {}
    sp_ts: Dict[str, np.ndarray] = {}
    for s_id, (s_obj, _) in sensors_dict.items():
        ts = _sensor_timeseries(s_obj)
        if ts is None or len(ts) == 0:
            continue
        fb_ts[s_id] = ts
        fb_quality[s_id] = _score_continuity(ts)
    for sp_id, (sp_obj, _) in setpoints_dict.items():
        ts = _sensor_timeseries(sp_obj)
        if ts is None or len(ts) == 0:
            continue
        sp_ts[sp_id] = ts
        sp_quality[sp_id] = _score_continuity(ts)

    if not fb_quality or not sp_quality:
        return []

    # Build the list of swap candidates: (fb_id, sp_id, fb_stats,
    # sp_stats, corr).  Diagnostic: log every (fb, sp) pair where the
    # quality asymmetry is large, even if the corr or n_unique gates
    # block the swap, so the user can see why the routine did or did
    # not fire.
    candidates: List[Tuple[str, str, ContinuityStats, ContinuityStats, float]] = []
    for s_id, s_stats in fb_quality.items():
        for sp_id, sp_stats in sp_quality.items():
            n = min(len(fb_ts[s_id]), len(sp_ts[sp_id]))
            corr = _safe_pearson(sp_ts[sp_id][:n], fb_ts[s_id][:n])
            if corr is None:
                continue
            sp_q = sp_stats.quality_score
            fb_q = s_stats.quality_score
            quality_ratio = sp_q / max(fb_q, 1e-6)

            # Only consider pairs with substantial quality asymmetry.
            if quality_ratio < quality_swap_ratio:
                continue

            # All four tightening conditions must hold for the swap to
            # actually fire.  Log near-miss pairs for diagnostics so a
            # future broken-sensor case is easy to spot.
            reasons_skipped: List[str] = []
            if abs(corr) < swap_corr_min:
                reasons_skipped.append(
                    f"|corr|={abs(corr):.3f} < swap_corr_min={swap_corr_min}"
                )
            if s_stats.n_unique > max_unique_for_broken_fb:
                reasons_skipped.append(
                    f"fb.n_unique={s_stats.n_unique} > "
                    f"max_unique_for_broken_fb={max_unique_for_broken_fb} "
                    f"(sensor-side does not look quantised enough)"
                )
            if sp_stats.n_unique < min_unique_for_good_sp:
                reasons_skipped.append(
                    f"sp.n_unique={sp_stats.n_unique} < "
                    f"min_unique_for_good_sp={min_unique_for_good_sp} "
                    f"(setpoint-side too coarse to be a real measurement)"
                )

            if reasons_skipped:
                LOGGER.info(
                    f"[REWIRE] {cits.id}: NOT swapping (fb='{s_id}', "
                    f"sp='{sp_id}'): quality_ratio={quality_ratio:.1f} "
                    f"(>= {quality_swap_ratio}), corr={corr:+.3f}, "
                    f"fb.n_unique={s_stats.n_unique}, "
                    f"sp.n_unique={sp_stats.n_unique}; "
                    f"blocked by: {' AND '.join(reasons_skipped)}."
                )
                continue
            candidates.append((s_id, sp_id, s_stats, sp_stats, corr))

    if not candidates:
        return []

    # When several setpoints could be swapped into the same broken
    # sensor (or vice versa) keep only the best (sp, fb) pair: highest
    # sp quality and lowest fb quality wins.  This avoids partial
    # repairs that would leave dangling fb/sp entries.
    candidates.sort(
        key=lambda c: (-c[3].quality_score, c[2].quality_score)
    )
    used_fb: set = set()
    used_sp: set = set()
    final_swaps: List[Tuple[str, str]] = []
    for s_id, sp_id, s_stats, sp_stats, corr in candidates:
        if s_id in used_fb or sp_id in used_sp:
            continue
        used_fb.add(s_id)
        used_sp.add(sp_id)
        final_swaps.append((s_id, sp_id))
        LOGGER.info(
            f"[REWIRE] {cits.id}: broken-sensor pathology detected "
            f"(corr={corr:+.3f} >= swap_corr_min={swap_corr_min}); "
            f"sensorValue side '{s_id}' looks broken "
            f"(n_unique={s_stats.n_unique}, "
            f"frac_on_°F_grid={s_stats.frac_on_F_grid:.2f}, "
            f"quality={s_stats.quality_score:.1f}); "
            f"setpointValue side '{sp_id}' looks like the real measurement "
            f"(n_unique={sp_stats.n_unique}, "
            f"frac_on_°F_grid={sp_stats.frac_on_F_grid:.2f}, "
            f"quality={sp_stats.quality_score:.1f}); "
            f"physically swapping ports."
        )

    # Apply the physical changes.
    #
    # ``model.remove_connection`` does NOT clean up the cached
    # ``input_port_index`` mapping on the surviving ConnectionPoints;
    # that is the existing pruning step's responsibility (see
    # :func:`_reindex_connection_point`).  We must replicate that here
    # because the next ``model.add_connection`` call goes through
    # :meth:`SimulationModel._resolve_port_index`, which asserts that an
    # ``input_port_index`` is supplied whenever the receiving port is a
    # Vector and the sender port is a Scalar (which is exactly our case
    # -- ``cits.input['sensorValue']`` is a Vector of size
    # ``n_sensors``, and ``SensorSystem.output['measuredValue']`` is a
    # Scalar).  Without this dance the call raises an AssertionError
    # AFTER the partial Connection / ConnectionPoint objects have been
    # appended to the graph, leaving the model in an inconsistent state
    # that crashes the next ``sim_model.load()`` with a KeyError on the
    # missing port-index entry.
    successfully_swapped: List[Tuple[str, str]] = []
    for s_id, sp_id in final_swaps:
        s_obj, _ = sensors_dict[s_id]
        sp_obj, _ = setpoints_dict[sp_id]
        # The broken sensor: disconnect from EVERY input port.
        removed_bad = _disconnect_sender_from_cits(s_obj, cits, model)
        # The good signal: disconnect from setpointValue and any
        # onOffSignal connection it may have, but DO NOT touch
        # sensorValue (in case this sensor was already wired there too,
        # though that would be unusual).
        removed_good = _disconnect_sender_from_cits(
            sp_obj, cits, model, ports=("setpointValue", "onOffSignal")
        )
        # Re-number the surviving connection-point indices on every
        # touched port so the new add_connection lands at a clean
        # index = current count (the next free slot in the Vector).
        for cp in list(cits.connects_at):
            if cp.input_port in ("sensorValue", "setpointValue", "onOffSignal"):
                _reindex_connection_point(cp)

        # Pick the input_port_index for the fresh sensorValue
        # connection.  After the disconnect + reindex above, the
        # remaining sensorValue connections occupy slots 0..N-1 where N
        # is len(sv_cp.connects_system_through); the new connection
        # claims slot N (within the original Vector size, since the
        # broken sensor freed exactly one slot).
        sv_cp = next(
            (cp for cp in cits.connects_at if cp.input_port == "sensorValue"),
            None,
        )
        target_index = len(sv_cp.connects_system_through) if sv_cp else 0
        try:
            model.add_connection(
                sender_component=sp_obj,
                receiver_component=cits,
                output_port="measuredValue",
                input_port="sensorValue",
                input_port_index=target_index,
            )
        except (ValueError, AttributeError, AssertionError) as ex:
            LOGGER.error(
                f"[REWIRE] {cits.id}: failed to add '{sp_id}' to "
                f"sensorValue (index={target_index}): {ex}; "
                f"attempting to roll back the swap so the model is not "
                f"left in a half-rewired state."
            )
            # Best-effort rollback: put the connections back where they
            # were.  Using ``input_port_index=target_index`` for the
            # broken sensor restores it to the slot it just vacated.
            try:
                model.add_connection(
                    sender_component=s_obj,
                    receiver_component=cits,
                    output_port="measuredValue",
                    input_port="sensorValue",
                    input_port_index=target_index,
                )
                for port in removed_good:
                    model.add_connection(
                        sender_component=sp_obj,
                        receiver_component=cits,
                        output_port="measuredValue",
                        input_port=port,
                    )
                for cp in list(cits.connects_at):
                    if cp.input_port in (
                        "sensorValue", "setpointValue", "onOffSignal",
                    ):
                        _reindex_connection_point(cp)
            except Exception as roll_ex:  # noqa: BLE001
                LOGGER.error(
                    f"[REWIRE] {cits.id}: rollback after failed swap "
                    f"raised again ({roll_ex}); model state may be "
                    f"corrupt for this CITS."
                )
            continue
        # Sync the stored ``n_*`` attributes with the post-swap
        # connection counts.  The downstream pruning + rebuild in
        # :func:`_rewire_one` resets ``n_sensors`` and ``n_setpoints``
        # to 1 itself, but leaves ``n_on_off_signals`` alone -- so when
        # we shrink the onOffSignal port (Zone_Air_Control_Temp goes
        # away), we MUST update the stored count here, or the rebuild
        # will allocate a Vector with an extra unused slot and
        # ``_populate_on_off_signal_norm_bounds`` writes a stale [0, 1]
        # default into it.  The same logic applies to ``n_setpoints``
        # (we removed the good signal from setpointValue too): updating
        # it here keeps the state consistent in case the rewire later
        # bails out before the pruning step runs.
        for port_name, attr_name in (
            ("sensorValue", "n_sensors"),
            ("setpointValue", "n_setpoints"),
            ("onOffSignal", "n_on_off_signals"),
        ):
            cp = next(
                (c for c in cits.connects_at if c.input_port == port_name),
                None,
            )
            new_count = len(cp.connects_system_through) if cp is not None else 0
            setattr(cits, attr_name, new_count)

        LOGGER.info(
            f"[REWIRE] {cits.id}: swap done -- "
            f"dropped '{s_id}' from {removed_bad}; "
            f"dropped '{sp_id}' from {removed_good}; "
            f"added '{sp_id}' -> sensorValue (input_port_index={target_index}); "
            f"n_sensors={cits.n_sensors}, n_setpoints={cits.n_setpoints}, "
            f"n_on_off_signals={cits.n_on_off_signals}."
        )
        successfully_swapped.append((s_id, sp_id))

    return successfully_swapped


def _reindex_connection_point(cp: Any) -> None:
    """Rebuild ``cp.input_port_index`` so the surviving connections are
    re-numbered to a contiguous ``0, 1, ..., N-1``.

    Called after :meth:`SimulationModel.remove_connection` removes losing
    connections; that method does not clean stale entries from the dict.
    """
    survivors = list(cp.connects_system_through)
    cp.input_port_index.clear()
    for i, conn in enumerate(survivors):
        cp.set_input_port_index(conn, i)


def _apply_seeds(
    cits: ControllerIdentificationPITorchSystem,
    *,
    score: LoopScore,
    actuator_seeds: ActuatorSeeds,
    Ti_default: float,
    kp_decade_pad: float,
    Ti_decade_pad: float,
    Ti_lb_floor: float,
    Ti_ub_ceil: float,
    kp_lb_floor: float,
    kp_ub_ceil: float,
    h: float,
) -> Tuple[float, float, float, float, float, float, bool]:
    """Write seeds onto ``cits.candidate_0_0``.

    Returns ``(kp_x0, kp_lb, kp_ub, Ti_x0, Ti_lb, Ti_ub, is_reverse)``.
    """
    cand = cits.candidate_0_0
    # twin4build's PI law internally flips the error sign for
    # ``isReverse=False`` (see pid_controller_system.py: ``err = sp - fb;
    # if not isReverse: err = -err``).  Concretely:
    #   * ``isReverse=True``  -> simulator uses ``err = sp - fb``
    #     (heating-style: u rises when fb < sp).
    #   * ``isReverse=False`` -> simulator uses ``err = fb - sp``
    #     (cooling-style: u rises when fb > sp).
    # Our regression fits ``Δu = slope * Δe`` with ``e = sp - fb``
    # (loop_classifier.score_pair).  For the simulator's
    # ``du_sim = kp * Δerr`` to match the regression's ``slope * Δe``
    # we need ``kp * sign(internal_err) = slope``.  Since ``kp`` is set
    # to ``|slope| >= 0``, the sign convention works out as:
    #   slope >= 0  -> isReverse=True   (sim sees +Δe, du = +|slope|·Δe)
    #   slope <  0  -> isReverse=False  (sim sees -Δe, du = -|slope|·Δe)
    # i.e. the boolean is the sign-of-slope, NOT its negation.  An
    # earlier version had this inverted, which silently flipped every
    # heating loop into cooling mode at simulation time.
    is_reverse = bool(score.slope >= 0.0)

    # --- kp ----------------------------------------------------------------
    kp_x0 = max(kp_lb_floor, min(kp_ub_ceil, float(score.kp)))
    decade = float(kp_decade_pad)
    kp_lb = max(kp_lb_floor, kp_x0 / (10.0 ** decade))
    kp_ub = min(kp_ub_ceil, kp_x0 * (10.0 ** decade))
    # Hard guarantee: lb <= x0 <= ub.  When x0 sits at one of the global
    # floors/ceilings the decade-pad math can collapse the interval; in
    # that case widen the opposite side instead of pushing x0 around.
    if kp_lb > kp_x0:
        kp_lb = kp_x0
    if kp_ub < kp_x0:
        kp_ub = kp_x0
    if kp_lb >= kp_ub:
        kp_lb = max(kp_lb_floor, kp_x0 * 0.1)
        kp_ub = min(kp_ub_ceil, kp_x0 * 10.0)
        if kp_lb >= kp_ub:  # absolute degeneracy
            kp_lb, kp_ub = kp_x0 * 0.5, kp_x0 * 2.0
    _set_param(cand, "kp", kp_x0, kp_lb, kp_ub)

    # --- Ti ----------------------------------------------------------------
    Ti_raw = score.Ti if score.Ti is not None else Ti_default
    Ti_x0 = float(np.clip(Ti_raw, Ti_lb_floor, Ti_ub_ceil))
    decade_t = float(Ti_decade_pad)
    Ti_lb = max(Ti_lb_floor, Ti_x0 / (10.0 ** decade_t))
    Ti_ub = min(Ti_ub_ceil, Ti_x0 * (10.0 ** decade_t))
    # ``h`` (sample step) is a *soft* preference: don't allow Ti < h when
    # the seed itself supports it, but never let the floor push above x0.
    Ti_lb = max(Ti_lb, min(h, Ti_x0))
    if Ti_lb > Ti_x0:
        Ti_lb = Ti_x0
    if Ti_ub < Ti_x0:
        Ti_ub = Ti_x0
    if Ti_lb >= Ti_ub:
        Ti_lb = max(Ti_lb_floor, Ti_x0 * 0.5)
        Ti_ub = min(Ti_ub_ceil, Ti_x0 * 2.0)
        if Ti_lb >= Ti_ub:
            Ti_lb, Ti_ub = Ti_x0 * 0.5, Ti_x0 * 2.0
    _set_param(cand, "Ti", Ti_x0, Ti_lb, Ti_ub)

    # --- output_min / output_max ------------------------------------------
    omin = float(np.clip(actuator_seeds.output_min_x0, 0.0, 1.0))
    omax = float(np.clip(actuator_seeds.output_max_x0, 0.0, 1.0))
    if omax <= omin:  # degenerate: collapse to safe defaults
        omin, omax = 0.0, 1.0
    _set_param(cand, "output_min", omin, 0.0, max(omin + 1e-3, omax))
    _set_param(cand, "output_max", omax, min(omin, omax - 1e-3), 1.0)

    # --- default_output (per-actuator scalar on the CITS) ------------------
    default_out = float(np.clip(actuator_seeds.default_output_x0, 0.0, 1.0))
    if hasattr(cits, "default_output_0"):
        # bounds left at the constructor's [0, 1]; just update value.
        cits.default_output_0.set(
            torch.tensor(default_out, dtype=torch.float64), normalized=False
        )

    # --- isReverse ---------------------------------------------------------
    if hasattr(cand, "isReverse"):
        cand.isReverse = is_reverse

    return kp_x0, kp_lb, kp_ub, Ti_x0, Ti_lb, Ti_ub, is_reverse


def _set_param(component: Any, attr: str, x0: float, lb: float, ub: float) -> None:
    """Write ``(x0, lb, ub)`` onto a ``tps.Parameter`` attribute, preserving
    its current scaling mode.

    The :class:`tps.Parameter` API uses min/max-value setters and a
    ``set(value, normalized=False)`` call to set the physical value.  This
    helper bundles the three operations and performs basic sanity checks.
    """
    p = getattr(component, attr, None)
    if p is None:
        return
    # Order matters: set bounds first, then write the physical value, so
    # the renormalization inside Parameter.set sees the new bounds.
    try:
        p.min_value = torch.tensor(float(lb), dtype=torch.float64)
        p.max_value = torch.tensor(float(ub), dtype=torch.float64)
        p.set(
            torch.tensor(float(x0), dtype=torch.float64),
            normalized=False,
        )
    except Exception as ex:  # noqa: BLE001
        LOGGER.warning(
            f"[REWIRE] Failed to set {component.__class__.__name__}.{attr} "
            f"(x0={x0}, lb={lb}, ub={ub}): {ex}"
        )


def _rewire_one(
    *,
    cits: ControllerIdentificationPITorchSystem,
    model: core.SimulationModel,
    h: float,
    confidence_high: float,
    confidence_low: float,
    Ti_default: float,
    kp_decade_pad: float,
    Ti_decade_pad: float,
    Ti_lb_floor: float,
    Ti_ub_ceil: float,
    kp_lb_floor: float,
    kp_ub_ceil: float,
    n_min_active: int,
    sat_lo: float,
    sat_hi: float,
    sp_fb_corr_max: float = 0.95,
    fb_actuator_corr_max: float = 0.95,
    fb_sp_scale_max_offset: float = 30.0,
    fb_sp_median_tracking_max: float = 1.5,
) -> RewireReport:
    """Rewire one PI-CITS.  See :func:`rewire_pi_loops` for arg semantics."""
    sensors_dict, setpoints_dict = _collect_input_signals(cits)
    actuator_sensor = _resolve_actuator_measurement(cits)

    # Bail out if any side is missing -- nothing to score against.
    if actuator_sensor is None:
        return _untouched_report(cits.id, reason="no_actuator_measurement")
    if not sensors_dict:
        return _untouched_report(cits.id, reason="no_feedback_sensors_wired")
    if not setpoints_dict:
        return _untouched_report(cits.id, reason="no_setpoints_wired")

    actuator_ts = _sensor_timeseries(actuator_sensor)
    if actuator_ts is None or len(actuator_ts) == 0:
        return _untouched_report(
            cits.id,
            reason="actuator_timeseries_missing",
            actuator_id=actuator_sensor.id,
        )

    # Pre-pass: detect and physically repair the "broken sensor +
    # real measurement on setpoint port" pathology.  When the BMS
    # wires a quantised override into ``sensorValue`` while the
    # genuinely continuous zone-temperature measurement sits on
    # ``setpointValue`` (with |corr| ~ 1 and ~5x more unique
    # values), we swap their port assignments before any scoring is
    # done.  This must happen up-front because
    # :func:`_collect_input_signals` keys signals by port and the
    # downstream scoring loop trusts port semantics (sensorValue =
    # feedback, setpointValue = schedule); without this repair the
    # broken sensor would dominate the regression.
    swapped = _maybe_swap_broken_sensor(
        cits=cits,
        model=model,
        sensors_dict=sensors_dict,
        setpoints_dict=setpoints_dict,
    )
    if swapped:
        # Re-collect after the physical changes.  The pruning step
        # later in this function operates on these updated dicts.
        sensors_dict, setpoints_dict = _collect_input_signals(cits)
        if not sensors_dict:
            return _untouched_report(
                cits.id,
                reason="no_feedback_sensors_after_swap",
                actuator_id=actuator_sensor.id,
            )
        if not setpoints_dict:
            return _untouched_report(
                cits.id,
                reason="no_setpoints_after_swap",
                actuator_id=actuator_sensor.id,
            )

    # If _sensor_timeseries already applied a transformation, this is a no-op.
    # Otherwise, auto-detect a percent-scaled actuator (typical when the
    # example wires actuator transforms only after rewire) and rescale to
    # [0, 1] fraction so saturation masks and PI gains are in fractional units.
    actuator_ts, _rescaled_pct = _maybe_rescale_percent(actuator_ts)

    # Run a 2-component GMM on the actuator timeseries to obtain an
    # "on_mask" -- a boolean per-sample label of whether the PI loop
    # is actively modulating (vs parked at ``default_output``).
    # This is the same GMM that ``_populate_gate_seeds_from_on_mask``
    # computes downstream; we re-run it here (cheap: 1D 2-component
    # EM, ~ ms per CITS) so the pair-scoring regression can restrict
    # itself to active samples.  Without this restriction, off samples
    # contribute ``Δu ~ 0`` regardless of ``Δe``, biasing the slope
    # estimate toward zero (kp pinned at ``kp_lb_floor``) and the
    # integrator coefficient toward ``e_mid``-correlated noise (Ti
    # pinned at ``Ti_ub_ceil``).  The active-only fit recovers
    # ``kp`` ~ ``Δu / Δe`` in fraction-per-K which is the right
    # physical scale for the PI controller's ``do_step`` (which uses
    # raw ``err = sp - fb`` in degrees C, no internal normalisation).
    try:
        bimodal_for_mask = derive_actuator_seeds_gmm(actuator_ts)
        active_mask: Optional[np.ndarray] = bimodal_for_mask.on_mask
    except Exception as ex:  # noqa: BLE001
        LOGGER.warning(
            f"[REWIRE] {cits.id}: GMM on_mask derivation failed ({ex}); "
            f"falling back to saturation-only filter for kp/Ti regression."
        )
        active_mask = None

    # Build the *gate-mode* mask by ranking every wired ``onOffSignal``
    # slot against ``on_mask_u`` via ROC AUC and selecting the
    # BandGate's active region on the winning slot.
    #
    # Rationale: a temperature PI driven by a BMS day/night schedule
    # has two physically distinct regimes -- *occupied* (PI actively
    # tracking a comfort setpoint) and *setback / unoccupied* (PI
    # parked, schedule overrides controller).  The GMM on the
    # actuator distribution catches "off" parking but is fooled by
    # leak-through samples (actuator dribbling at 5% during setback)
    # because the setpoint doesn't appear in its construction.  The
    # AUC ranking, in contrast, identifies the most informative
    # *external* schedule indicator (any onOffSignal slot -- could be
    # a binary occupancy bit, a multimodal setpoint, an outdoor-air-
    # reset signal, ...) without assuming any particular schedule
    # shape.  Intersecting both masks removes high-leverage regression
    # poison from the kp/Ti fit:
    #
    #   on_mask_u=True  AND  gate=False  -> actuator dribbles in
    #                                       setback (regression poison
    #                                       towards Δu ≈ 0 at huge Δe).
    #   on_mask_u=False AND  gate=True   -> actuator parked off during
    #                                       occupied (also Δu ≈ 0,
    #                                       drags slope estimate down).
    #
    # The same AUC ranking is re-run later by
    # :func:`_populate_gate_seeds_from_on_mask` to produce the
    # persisted runtime gate seeds, so identification and simulation
    # share a single definition of "active mode".  When the AUC
    # ranking finds no informative slot (``confidence="low"``) or the
    # intersected mask would drop below ``n_min_active``, we fall
    # back to the actuator-only mask with a ``[REWIRE]`` log line so
    # the degradation is auditable.
    combined_mask: Optional[np.ndarray] = active_mask
    if active_mask is not None:
        slot_signals, oo_min, oo_max, n_oo = _collect_on_off_slot_signals(cits)
        if n_oo > 0:
            try:
                gate_seeds_for_mask = derive_gate_seeds_from_on_mask(
                    active_mask, slot_signals, oo_min, oo_max,
                )
            except Exception as ex:  # noqa: BLE001
                LOGGER.warning(
                    f"[REWIRE] {cits.id}: gate-mask derivation failed "
                    f"({ex}); using on_mask_u only for kp/Ti regression."
                )
                gate_seeds_for_mask = None

            if (
                gate_seeds_for_mask is not None
                and gate_seeds_for_mask.confidence != "low"
            ):
                w = int(gate_seeds_for_mask.winner_slot)
                thr = float(gate_seeds_for_mask.gate_threshold_x0)
                band = float(gate_seeds_for_mask.gate_band_x0)
                w_sig = slot_signals[w] if 0 <= w < n_oo else None
                if w_sig is not None:
                    sig_arr = np.asarray(w_sig, dtype=np.float64).ravel()
                    n_w = min(sig_arr.size, active_mask.size)
                    span = float(oo_max[w] - oo_min[w])
                    if abs(span) > 1e-12 and n_w > 0:
                        s_norm = (sig_arr[:n_w] - float(oo_min[w])) / span
                        gate_mode_partial = (s_norm >= thr) & (
                            s_norm <= thr + band
                        )
                        gate_mode_mask = np.zeros(
                            active_mask.size, dtype=bool
                        )
                        gate_mode_mask[:n_w] = gate_mode_partial
                        proposed = active_mask & gate_mode_mask
                        n_on = int(active_mask.sum())
                        n_gate = int(gate_mode_mask.sum())
                        n_combined = int(proposed.sum())
                        winner_auc = float(
                            gate_seeds_for_mask.auc_per_slot[w]
                        )
                        polarity = (
                            "+"
                            if gate_seeds_for_mask.winner_polarity > 0
                            else "-"
                        )
                        if n_combined >= n_min_active:
                            combined_mask = proposed
                            LOGGER.info(
                                f"[REWIRE] {cits.id}: kp/Ti regression "
                                f"mask = on_mask_u({n_on}) AND gate-mode("
                                f"slot{w}, polarity={polarity}, "
                                f"AUC={winner_auc:.2f}, "
                                f"conf={gate_seeds_for_mask.confidence}, "
                                f"band=[{thr:.3f}, {thr + band:.3f}]) "
                                f"-> {n_combined} samples "
                                f"(intersection drops "
                                f"{n_on - n_combined} regression-"
                                f"poisoning samples)."
                            )
                        else:
                            LOGGER.warning(
                                f"[REWIRE] {cits.id}: gate-mode mask "
                                f"would drop combined sample count to "
                                f"{n_combined} < n_min_active="
                                f"{n_min_active} (on_mask_u={n_on}, "
                                f"gate={n_gate}); falling back to "
                                f"on_mask_u only."
                            )
            elif gate_seeds_for_mask is not None:
                # ``confidence="low"`` here means the AUC ladder
                # rejected the winner for *regression-mask*
                # intersection (we don't trust a 0.55-0.65 AUC slot
                # to label every sample as active/idle), but the
                # quantile-based gate threshold/band the same call
                # produced is still kept (see
                # ``derive_gate_seeds_from_on_mask`` -- borderline
                # slots cluster the on_mask=True samples around a
                # tight active value and the seeds reflect that,
                # which is much better than a neutral
                # ``[T_lo=0.5, band=1.0]`` "always on" gate at
                # simulation time).
                LOGGER.info(
                    f"[REWIRE] {cits.id}: onOffSignal slot only "
                    f"weakly predicts on_mask_u "
                    f"(reason={gate_seeds_for_mask.reason}); "
                    f"using on_mask_u alone for kp/Ti regression "
                    f"but keeping the slot's quantile-derived "
                    f"gate seeds for the simulator."
                )

    # Score every (sensor, setpoint) pair.
    #
    # Guard against "measurement-clone setpoints" -- BRICK ontologies
    # occasionally tag the *active resolved control reference* (which
    # closely tracks the zone temperature itself) and the true *schedule
    # setpoint* under the same Brick class.  When the setpoint is in
    # fact a near-copy of the feedback sensor, ``e = sp - fb`` is
    # essentially zero and any non-trivial slope from
    # :func:`score_pair` is regression-on-noise.  Skipping those pairs
    # before scoring keeps the winner a real (sensor, setpoint) loop.
    #
    # Also guard against "actuator-clone feedbacks" -- BRICK frequently
    # wires every sensor near a VAV (zone temp, supply-air temp,
    # supply-air flow, percent-air-flow, ...) onto ``sensorValue``.
    # When the actuator-feedback sensor (e.g. ``Zone_Percent_Air_Flow``)
    # is one of those candidates it is identical to ``actuator_ts``,
    # the regression ``Δu ~ Δe = Δ(sp) - Δu`` collapses to a near-
    # perfect ``Δu = -Δu + ...`` identity with ``R² ~ 1`` but
    # unphysical ``kp`` / ``Ti`` that get clamped at the bounds.
    # Sensors with ``|corr(s, u)| >= fb_actuator_corr_max`` are
    # excluded *before* scoring so the true zone-temperature feedback
    # can win.
    actuator_clone_sensors: Dict[str, float] = {}
    for s_id, (s_obj, _conn) in sensors_dict.items():
        s_ts = _sensor_timeseries(s_obj)
        if s_ts is None or len(s_ts) == 0:
            continue
        n_uc = min(len(s_ts), len(actuator_ts))
        if n_uc < 2:
            continue
        corr_ua = _safe_pearson(actuator_ts[:n_uc], s_ts[:n_uc])
        if corr_ua is not None and abs(corr_ua) >= fb_actuator_corr_max:
            actuator_clone_sensors[s_id] = corr_ua
    # Defensive: if the filter would remove every candidate, keep them
    # all (the user is better served by a degenerate winner that the
    # ``_apply_seeds`` clipping then bound-pins than by a silently
    # untouched CITS with no kp/Ti seeds at all).
    if actuator_clone_sensors and len(actuator_clone_sensors) >= len(sensors_dict):
        LOGGER.warning(
            f"[REWIRE] {cits.id}: every sensor candidate "
            f"({len(actuator_clone_sensors)}) correlates with the actuator "
            f"timeseries at >= fb_actuator_corr_max={fb_actuator_corr_max}; "
            f"keeping them all so a winner can be picked.  Inspect the "
            f"feedback wiring -- the swap pre-pass may have failed."
        )
        actuator_clone_sensors = {}
    elif actuator_clone_sensors:
        for s_id, corr_ua in actuator_clone_sensors.items():
            LOGGER.info(
                f"[REWIRE] {cits.id}: excluding feedback candidate "
                f"'{s_id}' (corr with actuator timeseries = "
                f"{corr_ua:+.3f}, >= fb_actuator_corr_max="
                f"{fb_actuator_corr_max}); looks like an actuator clone "
                f"rather than a zone-temperature feedback."
            )

    scores: Dict[Tuple[str, str], LoopScore] = {}
    excluded_pairs: Dict[Tuple[str, str], float] = {}
    # Scale-incompatible (fb, sp) pairs: ``|mean(fb) - mean(sp)| >
    # fb_sp_scale_max_offset``.  This catches the residual case where
    # an actuator-like signal (e.g. ``Zone_Percent_Air_Flow`` parked
    # at ~100%) survives the actuator-clone filter because it does
    # not happen to correlate strongly with *this* CITS's actuator
    # (e.g. on a reheat-valve CITS the damper position is driven by
    # a *separate* PI), but is still on a wildly different physical
    # scale than the temperature setpoint.  Without this guard, a
    # constant ~100% feedback combined with a swinging temperature
    # setpoint produces a high R^2 because ``Δu ~ Δsp - 0`` collapses
    # to ``Δu ~ Δsp``, which fits the actuator's setpoint-tracking
    # behaviour even though the slope mixes incompatible units.
    scale_incompatible_pairs: Dict[Tuple[str, str], Tuple[float, float]] = {}
    # ``downstream_pairs[(s, sp)] = (median |fb - sp|, sp_mean)`` collects
    # candidates that survive the scale filter but still drift far from
    # their setpoint (typical of ``Supply_Air_Temp`` wired as feedback for
    # a reheat-valve loop -- it sits 10-30 K above the zone setpoint
    # because the valve directly heats it, which is causality, not
    # closed-loop tracking).  See the in-loop comment block for the full
    # rationale.
    downstream_pairs: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for s_id, (s_obj, _conn) in sensors_dict.items():
        if s_id in actuator_clone_sensors:
            continue
        s_ts = _sensor_timeseries(s_obj)
        if s_ts is None or len(s_ts) == 0:
            continue
        for sp_id, (sp_obj, _conn) in setpoints_dict.items():
            sp_ts = _sensor_timeseries(sp_obj)
            if sp_ts is None or len(sp_ts) == 0:
                continue
            n = min(len(s_ts), len(sp_ts))
            corr = _safe_pearson(sp_ts[:n], s_ts[:n])
            if corr is not None and abs(corr) >= sp_fb_corr_max:
                excluded_pairs[(s_id, sp_id)] = corr
                continue
            try:
                fb_arr = s_ts[:n]
                sp_arr = sp_ts[:n]
                fb_mean = float(np.nanmean(fb_arr))
                sp_mean = float(np.nanmean(sp_arr))
                fb_max = float(np.nanmax(fb_arr))
                fb_min = float(np.nanmin(fb_arr))
                sp_max = float(np.nanmax(sp_arr))
                sp_min = float(np.nanmin(sp_arr))
                # Scale-mismatch is the *largest* of three deviations:
                # (a) gross mean offset (flow CFM ~300 vs temp ~20),
                # (b) fb-max far above sp-max (percent ~100 vs temp ~21),
                # (c) fb-min far below sp-min (a real feedback never
                #     drops significantly below the lowest setpoint;
                #     a percent signal happily goes to 0 while sp_min
                #     ~15 °C).
                offset = max(
                    abs(fb_mean - sp_mean),
                    fb_max - sp_max,
                    sp_min - fb_min,
                )
            except Exception:  # noqa: BLE001
                offset = float("nan")
                sp_mean = float("nan")
            if np.isfinite(offset) and offset > fb_sp_scale_max_offset:
                scale_incompatible_pairs[(s_id, sp_id)] = (offset, sp_mean)
                continue

            # ----------- Median-tracking-error (downstream-of-actuator) -----------
            # A *true* PI loop's feedback tracks its setpoint within
            # ~1 K most of the time -- the controller's whole job is to
            # keep them close, comfort bands rarely exceed +/-1 K.
            # ``Supply_Air_Temp`` wired as ``sensorValue`` for a
            # reheat-valve loop, by contrast, is the actuator's
            # *downstream effect*: even on a mildly-reheated system
            # the valve has to clear the AHU/zone offset every time it
            # opens, so ``Supply_Air_Temp`` sits ~2-4 K above
            # ``Zone_Air_Temp_Setpoint`` whenever the valve is active
            # (much more on a heavily-reheated system).  The
            # regression then identifies a spurious negative slope
            # (``Δsupply > 0`` when ``Δu > 0``, i.e.
            # ``Δ(sp - supply) < 0``) which gets a *high* R^2 because
            # the valve->supply mechanical link is direct, but it
            # does not represent the closed-loop PI law.  Median
            # ``|fb - sp|`` over the active-mode mask cleanly
            # separates the two regimes -- empirically observed on
            # bldg1: true zone-temp feedbacks have median 0.2-0.7 K
            # while supply-air-temp feedbacks have median 1.9-3.3 K.
            # A 1.5 K threshold safely sits between the two and also
            # works on heavier-reheat systems where the gap is
            # 10-30 K.
            try:
                tracking = np.abs(fb_arr - sp_arr)
                tracking = tracking[np.isfinite(tracking)]
                if combined_mask is not None and tracking.size:
                    m = combined_mask[: tracking.size]
                    if m.any():
                        tracking = tracking[m[: tracking.size]]
                med_track = (
                    float(np.median(tracking)) if tracking.size else float("nan")
                )
            except Exception:  # noqa: BLE001
                med_track = float("nan")
            if np.isfinite(med_track) and med_track > fb_sp_median_tracking_max:
                downstream_pairs[(s_id, sp_id)] = (med_track, sp_mean)
                continue

            sc = score_pair(
                u=actuator_ts,
                sp=sp_ts,
                fb=s_ts,
                h=h,
                n_min=n_min_active,
                sat_lo=sat_lo,
                sat_hi=sat_hi,
                on_mask=combined_mask,
            )
            scores[(s_id, sp_id)] = sc

    if excluded_pairs:
        for (s_id, sp_id), corr in excluded_pairs.items():
            LOGGER.info(
                f"[REWIRE] {cits.id}: excluding setpoint candidate "
                f"'{sp_id}' (corr with sensor '{s_id}' = {corr:+.3f}, "
                f">= sp_fb_corr_max={sp_fb_corr_max}); looks like a "
                f"measurement clone rather than a schedule setpoint."
            )

    # Defensive fallback: if the scale filter would kill every pair we
    # have left, undo it -- a degenerate winner that downstream
    # ``_apply_seeds`` then bound-pins is still better than a silently
    # untouched CITS.  (This branch shouldn't fire in practice on a
    # well-modelled building -- if it does, the temperature-feedback
    # candidate likely had its mean computed on a stale / empty
    # timeseries; the warning prompts manual inspection.)
    if scale_incompatible_pairs and not scores:
        LOGGER.warning(
            f"[REWIRE] {cits.id}: every (sensor, setpoint) pair "
            f"({len(scale_incompatible_pairs)}) failed the scale-"
            f"compatibility filter (|mean(fb) - mean(sp)| > "
            f"fb_sp_scale_max_offset={fb_sp_scale_max_offset}); "
            f"keeping them all so a winner can be picked.  Inspect "
            f"the wiring and units of the feedback signals."
        )
        for (s_id, sp_id), (offset, sp_mean) in scale_incompatible_pairs.items():
            s_obj, _conn = sensors_dict[s_id]
            sp_obj, _conn = setpoints_dict[sp_id]
            s_ts = _sensor_timeseries(s_obj)
            sp_ts = _sensor_timeseries(sp_obj)
            if s_ts is None or sp_ts is None:
                continue
            sc = score_pair(
                u=actuator_ts,
                sp=sp_ts,
                fb=s_ts,
                h=h,
                n_min=n_min_active,
                sat_lo=sat_lo,
                sat_hi=sat_hi,
                on_mask=combined_mask,
            )
            scores[(s_id, sp_id)] = sc
        scale_incompatible_pairs = {}
    elif scale_incompatible_pairs:
        for (s_id, sp_id), (offset, sp_mean) in scale_incompatible_pairs.items():
            LOGGER.info(
                f"[REWIRE] {cits.id}: excluding pair (sensor='{s_id}', "
                f"setpoint='{sp_id}') -- max(|Δmean|, fb_max-sp_max, "
                f"sp_min-fb_min) = {offset:.2f} (sp_mean={sp_mean:.2f}) "
                f"exceeds fb_sp_scale_max_offset={fb_sp_scale_max_offset}"
                f"; sensor is on a different physical scale than the "
                f"setpoint (e.g. flow / percent vs temperature)."
            )

    # Same defensive fallback for the median-tracking-error filter: if
    # rejecting downstream-of-actuator pairs leaves *no* scoreable
    # candidate (e.g. on a building where only Supply_Air_Temp is
    # historized as the "feedback"), undo the rejection so we still
    # produce a winner.  In practice this branch only fires when the
    # CITS has no zone-temperature feedback wired at all.
    if downstream_pairs and not scores:
        LOGGER.warning(
            f"[REWIRE] {cits.id}: every surviving (sensor, setpoint) "
            f"pair ({len(downstream_pairs)}) failed the median-tracking "
            f"filter (median |fb - sp| > "
            f"fb_sp_median_tracking_max={fb_sp_median_tracking_max} K); "
            f"keeping them all so a winner can be picked.  This usually "
            f"means no zone-temperature feedback was wired -- inspect "
            f"the candidate ``sensorValue`` connections."
        )
        for (s_id, sp_id), (med_track, sp_mean) in downstream_pairs.items():
            s_obj, _conn = sensors_dict[s_id]
            sp_obj, _conn = setpoints_dict[sp_id]
            s_ts = _sensor_timeseries(s_obj)
            sp_ts = _sensor_timeseries(sp_obj)
            if s_ts is None or sp_ts is None:
                continue
            sc = score_pair(
                u=actuator_ts,
                sp=sp_ts,
                fb=s_ts,
                h=h,
                n_min=n_min_active,
                sat_lo=sat_lo,
                sat_hi=sat_hi,
                on_mask=combined_mask,
            )
            scores[(s_id, sp_id)] = sc
        downstream_pairs = {}
    elif downstream_pairs:
        for (s_id, sp_id), (med_track, sp_mean) in downstream_pairs.items():
            LOGGER.info(
                f"[REWIRE] {cits.id}: excluding pair (sensor='{s_id}', "
                f"setpoint='{sp_id}') -- median |fb - sp| = "
                f"{med_track:.2f} K (sp_mean={sp_mean:.2f}) exceeds "
                f"fb_sp_median_tracking_max="
                f"{fb_sp_median_tracking_max} K; the feedback never "
                f"tracks the setpoint, so it is the actuator's "
                f"downstream effect (e.g. Supply_Air_Temp heated by a "
                f"reheat valve), not a closed-loop PI feedback."
            )

    if not scores:
        return _untouched_report(
            cits.id,
            reason="no_scoreable_pairs",
            actuator_id=actuator_sensor.id,
        )

    # Pick the winner by R^2.
    winner_pair, winner_score = max(scores.items(), key=lambda kv: kv[1].r2)
    candidate_scores = {pair: s.r2 for pair, s in scores.items()}
    confidence = confidence_label(
        winner_score.r2,
        winner_score.n_active,
        r2_high=confidence_high,
        r2_low=confidence_low,
        n_min=n_min_active,
    )

    # ------------------------------------------------------------------
    # Low-confidence path: regression slope is essentially noise, but
    # leaving the CITS fully untouched would mean ``candidate_0_0`` is
    # never seeded -- the simulator then runs with NaN/zero kp/Ti and
    # produces a near-constant actuator signal (RMSE ~ 0.8).  Instead,
    # we keep the regression's *winner pair* (the candidate filters
    # already eliminated obvious wrong ones: clones, scale-mismatches,
    # actuator-clones), prune to 1x1 so the soft-attention betas don't
    # smear the output across nonsense candidates, and apply a
    # **heuristic** kp / Ti seed.  output_min / output_max /
    # default_output remain genuinely data-driven via
    # :func:`derive_actuator_seeds` on the actuator timeseries.
    #
    # Heuristic kp is one decade below the geometric center of the kp
    # bounds -- a deliberately conservative value well inside the
    # floor/ceil envelope so the optimizer has slack on both sides
    # without forcing a wild gain at startup.  Heuristic Ti is
    # ``Ti_default`` (the same fallback used by ``score_pair`` when
    # the integral term cannot be identified).  ``is_reverse`` is
    # taken from the regression sign, which is robust even at low R^2
    # because the +1/-1 decision only requires the sign of the slope.
    # ------------------------------------------------------------------
    if confidence in ("low", "failed"):
        kp_heuristic = float(np.clip(
            np.sqrt(kp_lb_floor * kp_ub_ceil) / 10.0,
            kp_lb_floor,
            kp_ub_ceil,
        ))
        Ti_heuristic = float(np.clip(Ti_default, Ti_lb_floor, Ti_ub_ceil))
        # Preserve regression's slope sign (so isReverse stays correct)
        # but override |slope|, Ti, and tag r2/reason for diagnostics.
        slope_sign = np.sign(winner_score.slope) if winner_score.slope != 0.0 else 1.0
        winner_score = replace(
            winner_score,
            slope=float(slope_sign) * kp_heuristic,
            kp=kp_heuristic,
            Ti=Ti_heuristic,
            reason=f"heuristic_seed (regression r2={winner_score.r2:.3f} "
                   f"below confidence_low={confidence_low})",
        )
        LOGGER.info(
            f"[REWIRE] {cits.id}: regression confidence={confidence} "
            f"(r2={candidate_scores.get(winner_pair, float('nan')):.3f}); "
            f"applying heuristic kp={kp_heuristic:.3f}, Ti={Ti_heuristic:.1f}s "
            f"and pruning to winner pair (sensor='{winner_pair[0][-40:]}', "
            f"setpoint='{winner_pair[1][-30:]}'). output_*x0 still derived "
            f"from actuator GMM."
        )

    # High/medium (and now low/failed): prune losers and apply seeds.
    winner_sensor_id, winner_setpoint_id = winner_pair

    # Remove non-winning sensor connections.
    for s_id, (s_obj, _conn) in list(sensors_dict.items()):
        if s_id == winner_sensor_id:
            continue
        try:
            model.remove_connection(
                sender_component=s_obj,
                receiver_component=cits,
                output_port="measuredValue",
                input_port="sensorValue",
            )
        except (ValueError, AttributeError) as ex:
            LOGGER.warning(
                f"[REWIRE] {cits.id}: could not drop sensor '{s_id}': {ex}"
            )

    # Remove non-winning setpoint connections.
    for sp_id, (sp_obj, _conn) in list(setpoints_dict.items()):
        if sp_id == winner_setpoint_id:
            continue
        try:
            model.remove_connection(
                sender_component=sp_obj,
                receiver_component=cits,
                output_port="measuredValue",
                input_port="setpointValue",
            )
        except (ValueError, AttributeError) as ex:
            LOGGER.warning(
                f"[REWIRE] {cits.id}: could not drop setpoint '{sp_id}': {ex}"
            )

    # Re-number surviving input_port_index entries to start at 0.
    for cp in cits.connects_at:
        if cp.input_port in ("sensorValue", "setpointValue"):
            _reindex_connection_point(cp)

    # Collapse n_sensors / n_setpoints to 1 and rebuild candidate components.
    # ``n_on_off_signals`` is NOT pruned by the rewire (the gate input bus
    # is structurally distinct from the PI-error setpoint bus), so we
    # derive it from the wired connections here.  This mirrors what
    # :meth:`ControllerIdentificationTorchSystem.initialize` does the
    # first time a real simulation runs -- but the rewire path is the
    # earliest user of ``_build_components`` after translation, before
    # any ``initialize()`` has happened, so the attribute would
    # otherwise still be ``None`` and ``torch.full((None,), ...)``
    # would blow up.
    cits.n_sensors = 1
    cits.n_setpoints = 1
    if cits.n_on_off_signals is None:
        n_oo = cits.get_n_v_from_connections("onOffSignal")
        cits.n_on_off_signals = n_oo if n_oo is not None else 1
    if cits.n_actuators is None:
        n_act = cits._get_n_actuators_from_connections()
        cits.n_actuators = n_act if n_act is not None else 1
    cits._built = False
    cits._build_components()

    # Apply data-driven seeds.
    actuator_seeds = derive_actuator_seeds(actuator_ts)
    kp_x0, kp_lb, kp_ub, Ti_x0, Ti_lb, Ti_ub, is_reverse = _apply_seeds(
        cits=cits,
        score=winner_score,
        actuator_seeds=actuator_seeds,
        Ti_default=Ti_default,
        kp_decade_pad=kp_decade_pad,
        Ti_decade_pad=Ti_decade_pad,
        Ti_lb_floor=Ti_lb_floor,
        Ti_ub_ceil=Ti_ub_ceil,
        kp_lb_floor=kp_lb_floor,
        kp_ub_ceil=kp_ub_ceil,
        h=h,
    )

    return RewireReport(
        cits_id=cits.id,
        pruned=True,
        confidence=confidence,
        winner=winner_pair,
        actuator_id=actuator_sensor.id,
        score=winner_score,
        kp_x0=kp_x0,
        Ti_x0=Ti_x0,
        output_min_x0=actuator_seeds.output_min_x0,
        output_max_x0=actuator_seeds.output_max_x0,
        default_output_x0=actuator_seeds.default_output_x0,
        is_reverse=is_reverse,
        kp_lb=kp_lb,
        kp_ub=kp_ub,
        Ti_lb=Ti_lb,
        Ti_ub=Ti_ub,
        candidate_scores=candidate_scores,
        # ``winner_score.reason`` carries the heuristic-seed marker on
        # the low/failed path; it stays ``None`` for high/medium because
        # ``score_pair`` only sets it when the regression itself failed.
        reason=winner_score.reason,
    )


def _untouched_report(
    cits_id: str,
    *,
    reason: str,
    actuator_id: Optional[str] = None,
) -> RewireReport:
    """Build a "no-op" report for a CITS that the rewire skipped."""
    return RewireReport(
        cits_id=cits_id,
        pruned=False,
        confidence="failed",
        winner=None,
        actuator_id=actuator_id,
        score=None,
        kp_x0=None,
        Ti_x0=None,
        output_min_x0=None,
        output_max_x0=None,
        default_output_x0=None,
        is_reverse=None,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Frozen-pin pass: sets one-hot selection weights, gate polarity, and the
# gate-activity scalar ``alpha_gate_{a}`` according to ``mode``.  Lives
# inside the rewire module because it is conceptually the final step of
# the rewire: the seeded parameters and the topology pins together
# constitute "the CITS is now data-aware".
# ---------------------------------------------------------------------------


def _pin_frozen_cits_state(
    cits_list: List["ControllerIdentificationPITorchSystem"],
    *,
    mode: str,
) -> None:
    """Pin one-hot weights, gate polarity, and gate-activity per CITS.

    Args:
        cits_list: Every PI-CITS that the rewire processed.
        mode: ``"train"`` -> ``alpha_gate_{a} = 1.0`` (gate active);
            ``"simulate"`` -> ``alpha_gate_{a} = 0.0`` (gate bypassed,
            PI passthrough).  Any other value raises ``ValueError``.

    The function never resizes parameters; it just writes one-hot or
    scalar values onto the post-rebuild tensors.  For CITS that the
    rewire *did* prune, ``alpha_0`` / ``beta_0`` / ``gamma_0`` are
    already length-1 and the one-hot collapses to ``[1.0]``.  For
    untouched CITS (low-confidence path: still at the full
    ``n_sensors`` / ``n_setpoints``) we apply a simple ``zone temp``
    / ``temp setpoint`` substring heuristic on the wired sensor ids
    to pick the surviving slot; if neither match, the first slot wins
    for ``beta_0`` and the last for ``gamma_0`` (mirrors the
    BRICK-enumeration convention of primary-setpoint-last).
    """
    if mode not in ("train", "simulate"):
        raise ValueError(
            f"_pin_frozen_cits_state: mode must be 'train' or 'simulate', "
            f"got {mode!r}."
        )
    alpha_gate_value = 1.0 if mode == "train" else 0.0

    def _param_size(param) -> int:
        return param.data.shape[0] if param.data.ndim > 0 else 1

    def _set_one_hot(param, idx: Optional[int], default: int = 0) -> None:
        n = _param_size(param)
        if n <= 1:
            val = [1.0]
        else:
            pos = idx if idx is not None else default
            val = [0.0] * n
            if 0 <= pos < n:
                val[pos] = 1.0
        param.set(
            torch.tensor(val, dtype=torch.float64), normalized=False
        )

    def _set_scalar(param, value: float) -> None:
        param.set(
            torch.tensor(float(value), dtype=torch.float64),
            normalized=False,
        )

    def _find_idx(
        connection_points, port_name: str, *keywords: str
    ) -> Optional[int]:
        """Return the input-port index whose wired source id contains
        any of ``keywords`` (case-insensitive)."""
        for cp in connection_points:
            if cp.input_port != port_name:
                continue
            for conn in cp.connects_system_through:
                src = conn.connects_system
                name = src.id.lower()
                if any(kw in name for kw in keywords):
                    return cp.input_port_index.get(conn)
        return None

    for cits in cits_list:
        for a in range(cits.n_actuators):
            alpha = getattr(cits, f"alpha_{a}", None)
            beta = getattr(cits, f"beta_{a}", None)
            gamma = getattr(cits, f"gamma_{a}", None)
            beta_b = getattr(cits, f"beta_b_{a}", None)
            alpha_gate = getattr(cits, f"alpha_gate_{a}", None)
            gate = getattr(cits, f"gate_{a}", None)

            # Find zone-air-temp sensor slot (fallback to control-temp).
            zt_idx = _find_idx(
                cits.connects_at, "sensorValue", "zone_air_temp"
            )
            if zt_idx is None:
                zt_idx = _find_idx(
                    cits.connects_at,
                    "sensorValue",
                    "zone_air_control_temp",
                )
            tsp_idx = _find_idx(
                cits.connects_at,
                "setpointValue",
                "temp_setpoint",
                "air_temp_setpoint",
            )

            if alpha is not None:
                _set_one_hot(alpha, 0)
            if beta is not None:
                _set_one_hot(beta, zt_idx, default=0)
            if gamma is not None:
                gamma_default = max(0, _param_size(gamma) - 1)
                _set_one_hot(gamma, tsp_idx, default=gamma_default)
            if beta_b is not None:
                _set_one_hot(beta_b, zt_idx, default=0)
            if alpha_gate is not None:
                _set_scalar(alpha_gate, alpha_gate_value)
            if gate is not None and hasattr(gate, "polarity"):
                pol = getattr(gate, "polarity")
                if hasattr(pol, "set"):
                    _set_scalar(pol, 1.0)
