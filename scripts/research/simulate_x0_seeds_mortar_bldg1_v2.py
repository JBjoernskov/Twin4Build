"""V2 of simulate_x0_seeds_mortar_bldg1.py -- isolates *one* change at a
time vs the v1 script so we can identify which difference vs
``translator_example_mortar_bldg1.py`` is decisive for the rewire output.

Applied changes so far (apply one per iteration):
  * **#1 -- BRICK-class-keyed sensor transformations.**  v1 uses substring
    matching on sensor ids (``"command"``, ``"valve"``, ``"damper"``,
    ``"temp"``, ``"setpoint"``, ``"control_temp"``) which misses
    ``Supply_Air_Flow_Sensor`` and ``Percent_Air_Flow_Sensor``.  v2 calls
    ``model.set_dbconfigs(DB_CONFIG)`` + ``model.set_transformations(
    TRANSFORMATIONS)`` with the same BRICK-class-keyed dict as
    ``translator_example_mortar_bldg1.py``.  Side effect: v2 always
    re-translates (the serialized-RDF load path cannot run
    ``set_transformations`` because that helper requires the translator's
    ``sim2sem_map``).  *Result: no material change to the RMSE table.*
  * **#2 -- dropped the manual BandGate swap.**  ``BandGate`` is the
    default ``_gate_class`` on
    ``ControllerIdentificationSystem``, so the CITS constructor
    already created one.  v1 explicitly replaced it with a fresh
    ``BandGate(threshold=0.5, band=2.0, steepness=5.0)``; v2 leaves the
    constructor's ``BandGate`` (defaults: ``threshold=0.0``, ``band=1.0``,
    ``steepness=100.0``) in place.  *Result: no material change.*
  * **#3 -- rewire-before-load (canonical order).**  v1's
    ``_build_stage1_model`` eager-loaded the model and then ran the
    rewire after, which invalidates the execution order computed by
    ``load`` (the rewire mutates the topology -- prunes connections,
    rebuilds CITS candidates).  The script papered over that with a
    conditional ``sim_model.load()`` after the rewire.  v2 drops the
    eager load entirely; the rewire runs first and ``main()`` calls
    ``sim_model.load()`` once, post-rewire, against the final pruned
    topology.  Matches ``translator_example_mortar_bldg1.py``'s
    ``rewire(...) → load()`` order.

  * **#4 -- dropped ``_apply_frozen_weights``.**  The rewire helper
    already pins ``alpha_0`` / ``beta_0`` / ``gamma_0`` / ``beta_b_0`` /
    ``gate_0.polarity`` / ``alpha_gate_{a}`` / ``gamma_gate_0`` /
    ``gate_0.threshold`` / ``gate_0.band`` from data-driven seeds.  The
    post-rewire write is redundant and, for non-high-confidence gate
    seeds, overwrites the rewire's GMM-derived ``gamma_gate_0`` with a
    name-matched one-hot.  *Result: small RMSE change on the single
    "borderline" gate-seed CITS (AVRM107A: 0.091 -> 0.097).  No
    ``isReverse`` change.*
  * **#5 -- dropped the CITS pre-build loop in
    ``_prepare_stage1_model``.**  This is the **decisive** change.
    v1's loop set ``cits.n_on_off_signals`` and called
    ``_build_components()`` before the rewire ran.  When that pre-build
    is present, ``_collect_on_off_slot_signals`` walks the on/off slot
    signals, the GMM produces high-confidence gate seeds, and the
    ``gate-mode mask`` intersection filters the kp/Ti regression to
    "PI is in active control mode" samples -> cleaner slope estimates
    -> all 16 CITS come out ``is_reverse=True``.  Without the pre-build
    (canonical setup), ``cits.n_on_off_signals is None`` ->
    ``_collect_on_off_slot_signals`` reads ``n_oo = 0`` -> the slot
    walk is skipped, no gate seeds, no gate-mode mask -> the regression
    operates on the unfiltered ``on_mask_u`` -> some slopes come out
    negative -> some ``is_reverse=False``.  Confirms that the canonical
    script's "some False" outcome is an artifact of the rewire
    silently bypassing its gate-seed pipeline on un-built CITS, NOT a
    physical signal in the data.

Original behaviour:

Simulate the mortar_bldg1 stage-1 model using *only* the data-driven x0
seeds (no estimator).

Mirrors the model-setup section of ``translator_example_mortar_bldg1.py``
but stops before the estimator runs:

  1. Translate (or load) the stage-1 SimulationModel.
  2. Swap SigmoidGate for BandGate, configure SensorSystem transforms.
  3. Run ``rewire_pi_loops`` so each PI-CITS's ``candidate_0_0`` carries
     data-driven ``kp``, ``Ti``, ``output_min``, ``output_max``,
     ``default_output_0`` x0 values.
  4. Apply the same one-hot frozen weights the estimator sets
     (alpha/beta/gamma + gate polarity).
  5. ``simulator.simulate(...)`` over the rewire window.
  6. Per-CITS comparison of predicted (CITS output) vs measured (actuator
     command sensor) timeseries, with RMSE.

The output is:

* ``simulate_x0_seeds_mortar_bldg1.png`` -- 4-column grid, one subplot per
  CITS, sorted by kind (reheat first, then damper) and then by RMSE.
* A summary table printed to stdout giving the seed values + RMSE per
  CITS, plus aggregated RMSE per kind.

Useful as a sanity check that ``derive_actuator_seeds`` -> ``score_pair``
-> ``_apply_pi_seeds`` produces seeds that already explain a meaningful
fraction of the observed actuator dynamics *before* the estimator polishes
them.
"""
# Standard library imports
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Third party imports
import matplotlib.pyplot as plt
import numpy as np

# Local application imports
import twin4build as tb
import twin4build.core as core
from twin4build.systems.controller.controller_identification.controller_identification_pi_system import (
    ControllerIdentificationPISystem,
)
from twin4build.systems.controller.controller_identification.pi_loop_rewire import (
    rewire_pi_loops,
)
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.utils.logger import LOGGER

# ---------------------------------------------------------------------------
# Config -- kept in lock-step with translator_example_mortar_bldg1.py
# ---------------------------------------------------------------------------
LOGGER.logfile = "simulate_x0_seeds_mortar_bldg1_v2.log"

DB_CONFIG = {
    "table_name": "mortar_bldg1",
    "db_host": "localhost",
    "db_port": 5432,
    "db_name": "postgres",
    "db_user": "postgres",
    "db_password": "postgres",
}
TZ = ZoneInfo("America/Los_Angeles")
STEP_SIZE = 600  # 10-minute intervals
START_TIME = [datetime(2017, 1, d, tzinfo=TZ) for d in range(16, 30)]
END_TIME = [datetime(2017, 1, d, tzinfo=TZ) for d in range(17, 31)]

GATE_STEEPNESS = 5.0
# (T_lo, band_width) in *normalised* on/off-signal units.  The CITS
# forward pass divides each ``onOffSignal`` slot by its per-slot
# ``[min, max]`` range (populated by ``rewire_pi_loops``) before
# feeding the gate.  In this dataset the schedule is essentially
# binary (e.g. 60/70 degF), so normalised values saturate at 0 and 1;
# ``T_lo=0.5, band=2.0`` puts the lower edge midway between setback
# and occupied and keeps the upper edge above 1, so occupied stays
# inside the band.  A tighter band like ``[0.1, 0.9]`` collapses the
# gate to zero on both extremes and the actuator parks at
# ``default_output`` (kind-specific 0 or 1) for the whole trajectory.
BAND_INIT = (0.5, 2.0)

STAGE1_MODEL_ID = "bldg1_controls"
STAGE1_TTL = r"C:\Users\jabj\Documents\python\Datasets\mortar\mortargraphs\bldg1.ttl"

# v2 change #1: LOAD_SERIALIZED_MODEL is ignored -- v2 always re-translates
# so the returned Model carries ``sim2sem_map`` for ``set_transformations``.
LOAD_SERIALIZED_MODEL = False  # noqa: F841  (kept for diff visibility vs v1)

transformation_temp = lambda x: (x - 32) * 5 / 9  # degF -> degC  # noqa: E731
transformation_pct = lambda x: x / 100.0          # pct -> fraction  # noqa: E731

# v2 change #1: use BRICK-class-keyed transformations (matches
# translator_example_mortar_bldg1.py) instead of substring-based name
# matching.  Adds Supply_Air_Flow_Sensor (x*0.000578) and
# Percent_Air_Flow_Sensor (x/100) which the v1 substring loop missed.
TRANSFORMATIONS = {
    core.namespace.BRICK.Temperature_Sensor:        lambda x: (x - 32) * 5 / 9,
    core.namespace.BRICK.Temperature_Setpoint:      lambda x: (x - 32) * 5 / 9,
    core.namespace.BRICK.Supply_Air_Flow_Sensor:    lambda x: x * 0.000578,
    core.namespace.BRICK.Command:                   lambda x: x / 100.0,
    core.namespace.BRICK.Damper_Position_Setpoint:  lambda x: x / 100.0,
    core.namespace.BRICK.Valve_Command:             lambda x: x / 100.0,
    core.namespace.BRICK.Percent_Air_Flow_Sensor:   lambda x: x / 100.0,
}


# ---------------------------------------------------------------------------
# Stage-1 model setup (mirror translator_example_mortar_bldg1)
# ---------------------------------------------------------------------------
def _prepare_stage1_model(sim_model):
    """No-op stub (kept for diff visibility against v1).

    v2 change #1: dbconfig + transformations moved to
    ``_build_stage1_model`` (BRICK-class-keyed via ``set_transformations``).

    v2 change #2: dropped the manual BandGate swap with
    ``BAND_INIT=(0.5, 2.0)``.  ``BandGate`` is already the default
    ``_gate_class``.

    v2 change #5: dropped the CITS pre-build loop -- this is the
    **decisive** difference.  When CITS are pre-built before the rewire
    runs, ``cits.n_on_off_signals`` is non-None and
    ``_collect_on_off_slot_signals`` walks the on/off slot signals,
    feeding the GMM that produces high-confidence gate seeds and
    activates the ``gate-mode mask`` intersection in the kp/Ti
    regression.  Without pre-build, ``n_on_off_signals is None`` ->
    ``int(None or 0) = 0`` -> the slot walk is skipped, no gate seeds,
    no gate-mode mask, regression runs on noisy ``on_mask_u`` -> some
    slopes come out negative -> some ``is_reverse=False``.  Matches the
    canonical ``translator_example_mortar_bldg1.py`` setup, which never
    pre-builds CITS between translate and rewire.
    """
    return  # intentional no-op; left in place so the v1↔v2 diff still
    # shows where the dropped pre-build used to be (the rewire now sees
    # ``n_on_off_signals=None`` on every CITS).


def _build_stage1_model():
    # v2 change #1: force the fresh-translate path so the returned object is
    # a translator-built ``Model`` (carries ``sim2sem_map``).  The serialized
    # ``SimulationModel.load(rdf_file=...)`` path cannot run
    # ``set_transformations`` because that helper requires the translator's
    # semantic map; bypassing it would silently fall back to the v1
    # substring matcher and defeat the experiment.
    LOGGER.info(f"[sim-x0] Translating stage-1 model from {STAGE1_TTL}")
    sm = tb.SemanticModel(rdf_file=STAGE1_TTL, id=STAGE1_MODEL_ID, verbose=1500)
    translator = tb.Translator()
    sim_model = translator.translate(
        sm,
        systems_=[
            tb.ControllerIdentificationPISystem,
            tb.SensorSystem,
        ],
        id=STAGE1_MODEL_ID,
        verbose=1000,
    )
    sim_model.validate()
    # v2 change #1: BRICK-class-keyed dbconfig + transformations.
    sim_model.set_dbconfigs(DB_CONFIG)
    sim_model.set_transformations(TRANSFORMATIONS)
    _prepare_stage1_model(sim_model)
    # v2 change #3: load AFTER the rewire (canonical order).  The rewire
    # mutates the topology (prunes connections, repins CITS frozen
    # state), so an earlier load's execution order is invalidated --
    # main() now calls ``sim_model.load()`` once, post-rewire.
    return sim_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _short_id(cits_id):
    """``[VAVRM100]_8a71e89b`` -> ``VAVRM100``."""
    if "[" in cits_id and "]" in cits_id:
        return cits_id.split("[", 1)[1].split("]", 1)[0]
    return cits_id


def _resolve_actuator_sensor(cits):
    """Return the first SensorSystem downstream of cits.inputSignal."""
    for conn in cits.connected_through:
        if conn.output_port != "inputSignal":
            continue
        for cp in conn.connects_system_at:
            recv = cp.connection_point_of
            if isinstance(recv, SensorSystem):
                return recv
    return None


def _collect_input_sources(cits):
    """For a *post-rewire* CITS (n_sensors=n_setpoints=1), return the source
    :class:`SensorSystem` instances wired into each input port.

    Returns a dict ``{port: [SensorSystem, ...]}`` for ``"sensorValue"``,
    ``"setpointValue"``, ``"onOffSignal"`` (in that order).  Lists are
    typically length 1 after pruning, but kept as lists so multi-candidate
    (low-confidence, untouched) CITS still work.
    """
    sources: dict[str, list] = {
        "sensorValue": [],
        "setpointValue": [],
        "onOffSignal": [],
    }
    for cp in cits.connects_at:
        port = cp.input_port
        if port not in sources:
            continue
        for conn in cp.connects_system_through:
            sender = conn.connects_system
            if isinstance(sender, SensorSystem):
                sources[port].append(sender)
    return sources


def _read_sensor_values(sensor):
    """Return the loaded measured timeseries from a SensorSystem as a 1D
    np.ndarray in batch-major order (matches ``simulator.date_time_steps``).

    Falls back to an empty array when the sensor has no time_series_input.
    """
    tsi = getattr(sensor, "time_series_input", None)
    if tsi is None or getattr(tsi, "values", None) is None:
        return np.array([])
    return _series_time_first(tsi.values)


def _classify_kind(actuator_sensor):
    """`reheat` if the downstream sensor name doesn't mention damper, else `damper`."""
    if actuator_sensor is None:
        return "??"
    return "damper" if "damper" in actuator_sensor.id.lower() else "reheat"


def _apply_frozen_weights(cits_components, sim_model, rewire_reports):
    """Set the same one-hot alpha/beta/gamma + polarity weights the
    translator example sets before the estimator runs, plus the
    data-driven gate seeds (``gamma_gate_0``, ``gate_0.threshold``,
    ``gate_0.band``) when available.

    After ``rewire_pi_loops`` collapses each CITS to single-candidate, all
    of the one-hot weight vectors degenerate to ``[1.0]`` -- but we still
    need to write them so the CITS forward pass uses the right weights.

    The gate parameters (``gamma_gate_0`` / ``gate_0.threshold`` /
    ``gate_0.band``) are pulled from the rewire's ``gate_seeds`` for
    high-confidence CITS, so the simulate matches what the estimator
    sees after the gate-input ranking step.  Low-confidence CITS fall
    back to the name-based one-hot heuristic (and ``BAND_INIT`` is
    already on the BandGate from ``_prepare_stage1_model``).
    """
    fixed_vals, fixed_comps, fixed_attrs = [], [], []
    for cid, ctrl in cits_components.items():
        n_s, n_sp, n_c = ctrl.n_sensors, ctrl.n_setpoints, ctrl.n_candidates
        n_oo = ctrl.n_on_off_signals

        # Find which sensor is zone_air_temp / which setpoint is
        # temp_setpoint so multi-candidate CITS (low-confidence,
        # untouched by rewire) still use the right indices.  Pruned
        # CITS have n_s=n_sp=n_c=1.  ``gamma_gate`` indexes the
        # ``onOffSignal`` port now (gate-input bus, distinct from
        # ``setpointValue``); walk that port separately.
        #
        # Note on the sensorValue match: after ``_maybe_swap_broken_sensor``
        # in ``pi_loop_rewire``, a CITS whose original ``Zone_Air_Temp`` was
        # historized as a quantised override gets rewired so
        # ``Zone_Air_Control_Temp`` lands on ``sensorValue`` instead.  The
        # naming heuristic must therefore accept both spellings; the
        # substring ``zone_air_temp`` does NOT match
        # ``zone_air_control_temp`` because of the ``_control_`` infix.
        zt_i, tsp_i, oo_i = 0, n_sp - 1, n_oo - 1
        for cp in ctrl.connects_at:
            for conn in cp.connects_system_through:
                idx = cp.input_port_index.get(conn)
                name = conn.connects_system.id.lower()
                if cp.input_port == "sensorValue" and (
                    "zone_air_temp" in name or "zone_air_control_temp" in name
                ):
                    zt_i = idx
                elif cp.input_port == "setpointValue" and (
                    "temp_setpoint" in name or "air_temp_setpoint" in name
                ):
                    tsp_i = idx
                elif cp.input_port == "onOffSignal" and (
                    "temp_setpoint" in name or "air_temp_setpoint" in name
                ):
                    oo_i = idx

        one_hot_s = [1.0 if i == zt_i else 0.0 for i in range(n_s)]
        one_hot_sp = [1.0 if i == tsp_i else 0.0 for i in range(n_sp)]
        one_hot_oo = [1.0 if i == oo_i else 0.0 for i in range(n_oo)]
        one_hot_c = [1.0] + [0.0] * (n_c - 1)

        # Pull data-driven gate seeds when the rewire report has them.
        rep = rewire_reports.get(cid) if rewire_reports else None
        gate_seeds = getattr(rep, "gate_seeds", None) if rep is not None else None
        gate_high_conf = (
            gate_seeds is not None and gate_seeds.confidence == "high"
        )
        if gate_high_conf:
            gamma_gate_val = gate_seeds.gamma_gate_x0.tolist()
        else:
            gamma_gate_val = one_hot_oo

        knobs = [
            ("alpha_0", one_hot_c),
            ("beta_0", one_hot_s),
            ("gamma_0", one_hot_sp),
            ("alpha_gate_0", 1.0),
            ("gamma_gate_0", gamma_gate_val),
            ("gate_0.polarity", 1.0),
        ]
        if hasattr(ctrl, "beta_b_0"):
            knobs.insert(3, ("beta_b_0", one_hot_s))
        # When the gate-seed AUC ladder is decisive, also pin
        # ``gate_0.threshold`` and ``gate_0.band`` to the seeded
        # active-range quantiles (matches the translator example's
        # frozen-knob set so simulate previews the same configuration
        # the estimator sees at iter 1).
        if gate_high_conf:
            knobs.append(("gate_0.threshold", float(gate_seeds.gate_threshold_x0)))
            knobs.append(("gate_0.band", float(gate_seeds.gate_band_x0)))

        for attr, val in knobs:
            fixed_vals.append(val)
            fixed_comps.append(ctrl)
            fixed_attrs.append(attr)

    sim_model.set_parameters(
        values=fixed_vals,
        components=fixed_comps,
        parameter_names=fixed_attrs,
    )


def _wire_actuator_transformations(cits_components):
    """Make sure the actuator-command sensors apply percent->fraction so the
    measured timeseries is comparable to the simulated CITS output.
    """
    actuator_map = {}
    for cid, ctrl in cits_components.items():
        for conn in ctrl.connected_through:
            if conn.output_port != "inputSignal":
                continue
            for cp in conn.connects_system_at:
                s = cp.connection_point_of
                if isinstance(s, SensorSystem):
                    s._transformation = transformation_pct
                    actuator_map.setdefault(cid, []).append(s)
    return actuator_map


def _series_time_first(arr):
    """Flatten a time-first ``(n_t, n_s[, 1])`` tensor / array to a 1D ndarray
    in batch-major order so it lines up with ``simulator.date_time_steps``
    (which is shape ``(n_s, n_t)``).

    Both ``history(i_c=0)`` (-> ``(n_t, n_s)``) and
    ``time_series_input.values[..., 0]`` (-> ``(n_t, n_s)``) use this layout.
    """
    a = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
    if a.ndim == 3:
        a = a[..., 0]
    # (n_t, n_s) -> (n_s, n_t) -> (n_s * n_t,)  (batch-major)
    return a.T.reshape(-1).astype(np.float64)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_grid(per_cits, time_axis, out_path):
    """Plot predicted vs measured u(t) for each CITS in a 4-column grid.

    ``per_cits`` is a list of dicts with keys:
        cits_id, kind, pred, meas, rmse, seeds (dict).
    Sorted by kind (reheat first, damper second) then by RMSE descending.
    """
    if not per_cits:
        LOGGER.warning("[sim-x0] No CITS to plot")
        return

    # Sort: reheat first, then damper; within each kind, worst RMSE first
    kind_rank = {"reheat": 0, "damper": 1, "??": 2}
    per_cits = sorted(per_cits, key=lambda r: (kind_rank.get(r["kind"], 9), -r["rmse"]))

    n = len(per_cits)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 2.8 * nrows), squeeze=False
    )
    axes = axes.flatten()

    t = np.asarray(time_axis).reshape(-1)
    if t.size == 0:
        t = np.arange(per_cits[0]["meas"].size)

    for ax, r in zip(axes, per_cits):
        meas = r["meas"]
        pred = r["pred"]
        # Truncate to common length (defensive: history vs time_series sometimes differ)
        m = min(meas.size, pred.size, t.size)
        ax.plot(t[:m], meas[:m], color="tab:blue", lw=0.9, label="measured")
        ax.plot(t[:m], pred[:m], color="tab:red", lw=0.9, alpha=0.85, label="x0-sim")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(
            f"{_short_id(r['cits_id'])}  [{r['kind']}]  RMSE={r['rmse']:.3f}",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)

        s = r["seeds"]
        info = (
            f"kp={s.get('kp', float('nan')):.3g}  Ti={s.get('Ti', float('nan')):.3g}\n"
            f"min={s.get('omin', float('nan')):.2f}  max={s.get('omax', float('nan')):.2f}\n"
            f"dflt={s.get('dflt', float('nan')):.2f}  rev={int(s.get('rev', 0))}"
        )
        ax.text(
            0.02, 0.98, info, transform=ax.transAxes,
            fontsize=7, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
        )
        ax.legend(fontsize=7, loc="upper right", framealpha=0.85)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        "Predicted (x0 seeds) vs measured actuator command -- mortar_bldg1",
        fontsize=12, y=1.005,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    LOGGER.info(f"[sim-x0] Wrote figure to {out_path}")
    plt.close(fig)


def _plot_inputs_grid(per_cits, time_axis, out_path):
    """Diagnostic plot: per-CITS controller inputs alongside the actuator
    response.

    Each CITS gets a vertical stack of two panels:

      * **Top**: the *post-rewire* setpoint (orange) and feedback sensor
        (blue) the CITS actually sees, plus the on/off / schedule signal
        on a secondary y-axis (gray).  All traces are in the SensorSystem's
        physical units (°C / fraction) thanks to ``_transformation``.

      * **Bottom**: measured actuator command (blue) vs the simulated
        CITS output (red).  Same trace as the main grid plot, just
        repeated under each input panel for visual context.

    Use this when a CITS has high RMSE and you need to see which input
    is driving (or *failing* to drive) the controller's output.
    """
    if not per_cits:
        LOGGER.warning("[sim-x0] No CITS to plot for inputs")
        return

    kind_rank = {"reheat": 0, "damper": 1, "??": 2}
    per_cits = sorted(per_cits, key=lambda r: (kind_rank.get(r["kind"], 9), -r["rmse"]))

    n = len(per_cits)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    # 2 sub-rows per CITS row (top: inputs, bottom: actuator).
    fig = plt.figure(figsize=(5.0 * ncols, 4.4 * nrows))
    gs = fig.add_gridspec(
        nrows * 2, ncols,
        height_ratios=[1.1, 0.8] * nrows,
        hspace=0.55, wspace=0.32,
    )

    t = np.asarray(time_axis).reshape(-1)
    if t.size == 0:
        t = np.arange(per_cits[0]["meas"].size)

    for idx, r in enumerate(per_cits):
        row = idx // ncols
        col = idx % ncols
        ax_top = fig.add_subplot(gs[row * 2, col])
        ax_bot = fig.add_subplot(gs[row * 2 + 1, col], sharex=ax_top)

        # ---- top panel: sp / fb / oof ----------------------------------
        sp = r.get("sp", np.array([]))
        fb = r.get("fb", np.array([]))
        oof_list = r.get("oof", [])
        m_top = min(
            t.size,
            sp.size if sp.size else t.size,
            fb.size if fb.size else t.size,
        )
        if fb.size:
            ax_top.plot(t[:m_top], fb[:m_top], color="tab:blue", lw=0.9, label="fb")
        if sp.size:
            ax_top.plot(t[:m_top], sp[:m_top], color="tab:orange", lw=0.9, label="sp")
        ax_top.set_ylabel("temp [°C]", fontsize=7)
        ax_top.tick_params(labelsize=7)
        ax_top.grid(True, alpha=0.25)
        ax_top.set_title(
            f"{_short_id(r['cits_id'])}  [{r['kind']}]  RMSE={r['rmse']:.3f}",
            fontsize=9,
        )

        if oof_list:
            ax_oo = ax_top.twinx()
            colors = ["#7f7f7f", "#bcbd22"]
            for i, (oof_id, oof) in enumerate(oof_list):
                m_oo = min(t.size, oof.size)
                ax_oo.plot(
                    t[:m_oo], oof[:m_oo],
                    color=colors[i % len(colors)],
                    lw=0.7, ls="--", alpha=0.7,
                    label=f"oof[{i}]",
                )
            ax_oo.set_ylabel("on/off [°C]", fontsize=7, color="#555")
            ax_oo.tick_params(labelsize=7, colors="#555")

        # Compose a compact legend that includes both axes.
        h1, l1 = ax_top.get_legend_handles_labels()
        if oof_list:
            h2, l2 = ax_oo.get_legend_handles_labels()
            ax_top.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper right",
                          framealpha=0.85, ncol=2)
        else:
            ax_top.legend(fontsize=6, loc="upper right", framealpha=0.85)

        # ---- bottom panel: actuator measured vs simulated --------------
        meas = r["meas"]
        pred = r["pred"]
        m = min(meas.size, pred.size, t.size)
        ax_bot.plot(t[:m], meas[:m], color="tab:blue", lw=0.9, label="u_meas")
        ax_bot.plot(t[:m], pred[:m], color="tab:red", lw=0.9, alpha=0.85, label="u_pred")
        ax_bot.set_ylim(-0.05, 1.05)
        ax_bot.set_ylabel("u [-]", fontsize=7)
        ax_bot.tick_params(labelsize=7)
        ax_bot.grid(True, alpha=0.25)

        s = r["seeds"]
        info = (
            f"kp={s.get('kp', float('nan')):.3g}  Ti={s.get('Ti', float('nan')):.3g}  "
            f"rev={int(bool(s.get('rev', 0)))}\n"
            f"min={s.get('omin', float('nan')):.2f}  "
            f"max={s.get('omax', float('nan')):.2f}  "
            f"dflt={s.get('dflt', float('nan')):.2f}"
        )
        ax_bot.text(
            0.02, 0.98, info, transform=ax_bot.transAxes,
            fontsize=6.5, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
        )
        ax_bot.legend(fontsize=6, loc="upper right", framealpha=0.85)

        # Show source signal ids in a footer below the bottom panel.
        src = r.get("input_ids", {})
        src_txt = (
            f"sp: {src.get('sp', '?')[-32:]}\n"
            f"fb: {src.get('fb', '?')[-32:]}"
        )
        if src.get("oof"):
            src_txt += f"\noof: {src['oof'][0][-32:]}"
        ax_bot.text(
            0.02, -0.55, src_txt, transform=ax_bot.transAxes,
            fontsize=6, color="#555", va="top", ha="left",
            family="monospace",
        )

    fig.suptitle(
        "Per-CITS inputs (sp, fb, on/off) vs actuator response -- mortar_bldg1",
        fontsize=12, y=1.005,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    LOGGER.info(f"[sim-x0] Wrote inputs figure to {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sim_model = _build_stage1_model()

    # 1) Run the rewire so each CITS's candidate_0_0 carries data-driven seeds.
    LOGGER.info("[sim-x0] Running rewire_pi_loops to seed candidate_0_0 ...")
    rewire_reports = rewire_pi_loops(
        sim_model,
        start_time=START_TIME,
        end_time=END_TIME,
        step_size=STEP_SIZE,
    )

    # Per-CITS rewire summary: surface every CITS's confidence / winner /
    # reason so untouched (pruned=False) loops self-report their failure
    # mode instead of silently showing NaN seeds in the table later.
    print("\n[sim-x0] Per-CITS rewire summary:")
    print(f"  {'cits':<14}{'pruned':<8}{'conf':<10}{'kp_x0':>10}{'Ti_x0':>10}  reason / winner")
    print("  " + "-" * 90)
    for cid, rep in sorted(
        rewire_reports.items(),
        key=lambda kv: (
            0 if kv[1].pruned else 1,
            kv[1].confidence or "",
            _short_id(kv[0]),
        ),
    ):
        kp_str = f"{rep.kp_x0:.4g}" if rep.kp_x0 is not None else "  -  "
        ti_str = f"{rep.Ti_x0:.4g}" if rep.Ti_x0 is not None else "  -  "
        reason = getattr(rep, "reason", None) or ""
        is_heuristic = reason.startswith("heuristic_seed")
        if rep.pruned and rep.winner is not None:
            sensor_id, sp_id = rep.winner
            tag = "[HEUR] " if is_heuristic else ""
            tail = (
                f"{tag}won: sensor='{sensor_id[-40:]}', "
                f"sp='{sp_id[-30:]}'"
            )
            if is_heuristic:
                tail += f"  ({reason})"
        else:
            tail = f"reason: {reason or None}"
        print(
            f"  {_short_id(cid):<14}"
            f"{str(rep.pruned):<8}"
            f"{(rep.confidence or '-'):<10}"
            f"{kp_str:>10}"
            f"{ti_str:>10}  {tail}"
        )
    print()

    # v2 change #3: rewire-before-load.  ``_build_stage1_model`` no longer
    # eager-loads; load happens once here (post-rewire) and recomputes the
    # execution order against the final pruned topology.
    sim_model.load()

    cits_components = {
        cid: c
        for cid, c in sim_model.components.items()
        if isinstance(c, ControllerIdentificationPISystem)
    }

    # v2 change #4: dropped the explicit ``_apply_frozen_weights`` call.
    # ``rewire_pi_loops`` (above) now pins ``alpha_0`` / ``beta_0`` /
    # ``gamma_0`` / ``beta_b_0`` / ``gate_0.polarity`` /
    # ``alpha_gate_{a}`` / ``gamma_gate_0`` / ``gate_0.threshold`` /
    # ``gate_0.band`` from data-driven seeds itself, so the post-rewire
    # write is redundant and -- for non-high-confidence gate seeds --
    # actually overwrites the rewire's GMM-derived ``gamma_gate_0`` with
    # a name-matched one-hot.  Matches what
    # ``translator_example_mortar_bldg1.py`` does (no manual frozen-weight
    # write between rewire and simulate).

    # 3) Make sure downstream actuator sensors apply percent->fraction.
    actuator_map = _wire_actuator_transformations(cits_components)

    # 4) Simulate.
    simulator = tb.Simulator(sim_model)
    LOGGER.info(
        f"[sim-x0] Simulating {len(cits_components)} CITS over "
        f"{len(START_TIME)} period(s) at {STEP_SIZE}s step ..."
    )
    simulator.simulate(
        start_time=list(START_TIME),
        end_time=list(END_TIME),
        step_size=STEP_SIZE,
        show_progress_bar=True,
    )

    # 5) Collect predicted vs measured per CITS.
    per_cits = []
    for cid, ctrl in cits_components.items():
        actuators = actuator_map.get(cid, [])
        if not actuators:
            LOGGER.warning(f"[sim-x0] {_short_id(cid)}: no actuator sensor wired")
            continue
        # Use the first downstream actuator (every PI-CITS has exactly one).
        actuator = actuators[0]
        kind = _classify_kind(actuator)

        # Predicted: simulated CITS output that flowed into actuator.measuredValue.
        try:
            pred = _series_time_first(actuator.input["measuredValue"].history(i_c=0))
        except Exception as ex:  # noqa: BLE001
            LOGGER.warning(f"[sim-x0] {_short_id(cid)}: cannot read predicted history: {ex}")
            continue

        # Measured: loaded timeseries on the actuator sensor itself.
        tsi = getattr(actuator, "time_series_input", None)
        if tsi is None or getattr(tsi, "values", None) is None:
            LOGGER.warning(f"[sim-x0] {_short_id(cid)}: actuator has no time_series_input")
            continue
        meas = _series_time_first(tsi.values)

        m = min(pred.size, meas.size)
        if m == 0:
            continue
        pred, meas = pred[:m], meas[:m]
        mask = np.isfinite(pred) & np.isfinite(meas)
        if not np.any(mask):
            continue
        rmse = float(np.sqrt(np.mean((pred[mask] - meas[mask]) ** 2)))

        rep = rewire_reports.get(cid)
        seeds = {}
        if rep is not None and rep.pruned:
            seeds = {
                "kp": rep.kp_x0, "Ti": rep.Ti_x0,
                "omin": rep.output_min_x0, "omax": rep.output_max_x0,
                "dflt": rep.default_output_x0, "rev": rep.is_reverse,
            }

        # Collect controller inputs (sp / fb / on-off) for the inputs plot.
        # After rewire each port has at most one source SensorSystem; the
        # multi-source case (low-confidence / untouched CITS) is rare now
        # that we apply heuristic seeds, but we still handle it by taking
        # the first source so the plot stays single-trace per CITS.
        sources = _collect_input_sources(ctrl)
        sp_arr = np.array([])
        fb_arr = np.array([])
        oof_list: list = []
        sp_id = "?"
        fb_id = "?"
        oof_ids: list = []
        if sources["setpointValue"]:
            sp_src = sources["setpointValue"][0]
            sp_arr = _read_sensor_values(sp_src)
            sp_id = sp_src.id
        if sources["sensorValue"]:
            fb_src = sources["sensorValue"][0]
            fb_arr = _read_sensor_values(fb_src)
            fb_id = fb_src.id
        for oof_src in sources["onOffSignal"]:
            oof_arr = _read_sensor_values(oof_src)
            if oof_arr.size:
                oof_list.append((oof_src.id, oof_arr))
                oof_ids.append(oof_src.id)

        per_cits.append({
            "cits_id": cid,
            "actuator_id": actuator.id,
            "kind": kind,
            "pred": pred, "meas": meas, "rmse": rmse, "seeds": seeds,
            "sp": sp_arr,
            "fb": fb_arr,
            "oof": oof_list,
            "input_ids": {"sp": sp_id, "fb": fb_id, "oof": oof_ids},
        })

    # 6) Plot grid + summary table.
    out_dir = Path(__file__).parent
    fig_path = out_dir / "simulate_x0_seeds_mortar_bldg1_v2.png"
    inputs_fig_path = out_dir / "simulate_x0_seeds_mortar_bldg1_v2_inputs.png"
    time_axis = np.asarray(simulator.date_time_steps).reshape(-1) \
        if simulator.date_time_steps is not None else np.array([])
    _plot_grid(per_cits, time_axis, fig_path)
    _plot_inputs_grid(per_cits, time_axis, inputs_fig_path)

    # Print summary
    print(
        f"\n{'cits':<14} {'kind':<8} {'kp':>7} {'Ti':>7} "
        f"{'omin':>6} {'omax':>6} {'dflt':>6} {'rev':>4} {'rmse':>7}"
    )
    print("-" * 76)
    by_kind_rmses = {"reheat": [], "damper": [], "??": []}
    for r in sorted(per_cits, key=lambda x: (x["kind"], -x["rmse"])):
        s = r["seeds"]
        kp = s.get("kp", float("nan"))
        Ti = s.get("Ti", float("nan"))
        om = s.get("omin", float("nan"))
        oM = s.get("omax", float("nan"))
        df = s.get("dflt", float("nan"))
        rv = int(bool(s.get("rev", 0)))
        print(
            f"{_short_id(r['cits_id']):<14} {r['kind']:<8} "
            f"{kp:>7.3g} {Ti:>7.3g} {om:>6.2f} {oM:>6.2f} {df:>6.2f} "
            f"{rv:>4d} {r['rmse']:>7.3f}"
        )
        by_kind_rmses[r["kind"]].append(r["rmse"])

    print("-" * 76)
    print(f"{'aggregate':<14} {'kind':<8} {'count':>7} {'mean':>7} {'median':>7} "
          f"{'min':>7} {'max':>7}")
    for k, rmses in by_kind_rmses.items():
        if not rmses:
            continue
        arr = np.asarray(rmses)
        print(
            f"{'':<14} {k:<8} {arr.size:>7d} {arr.mean():>7.3f} "
            f"{np.median(arr):>7.3f} {arr.min():>7.3f} {arr.max():>7.3f}"
        )
    if per_cits:
        all_arr = np.concatenate([r["pred"] - r["meas"] for r in per_cits])
        all_arr = all_arr[np.isfinite(all_arr)]
        global_rmse = float(np.sqrt(np.mean(all_arr ** 2))) if all_arr.size else float("nan")
        print(f"\nGlobal x0-RMSE across all CITS samples: {global_rmse:.4f}")


if __name__ == "__main__":
    sys.exit(main())
