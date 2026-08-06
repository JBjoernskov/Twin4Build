"""Three-stage controller + physics identification on Mortar bldg1.

Stage 1 translates the BRICK graph with PI-CITS + Sensor systems and
calls :meth:`Estimator.estimate` to identify each loop's PI gains.

Stage 2 re-translates with the full physics stack, fills missing
physics-side inputs (occupancy / heatGain / outdoor CO2 / supply water
T) and wires the AHU SAT setpoint from the BMS measurement (see
:func:`_build_measured_sat_overrides`), then transfers the Stage-1 PI
parameters onto the closed-loop graph.

Stage 3 estimates the building-space RC parameters on the closed-loop
model.  PI gains are frozen by simply not listing them in the parameter
vector -- they keep the values loaded from Stage 1.

A final calibrated replay-simulate runs on a single-window slice.

Set ``CTRL_PICKLE_FILE`` / ``PHYSICS_PICKLE_FILE`` to a saved estimation
pickle to skip the corresponding estimator call and only re-apply the
stored parameters.  Flip ``PLOT_CTRL_MEASUREMENTS`` /
``PLOT_PHYSICS_MEASUREMENTS`` to draw per-sensor predicted-vs-measured
diagnostics for the corresponding stage (see
:func:`_plot_estimation_measurements`).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import twin4build as tb
import twin4build.core as core
from twin4build.systems.utils.smooth_saturation import saturation_mode
from twin4build.utils.plot.plot import Colors
from twin4build.utils.logger import LOGGER


# ---------------------------------------------------------------------------
# Dataset + window config
# ---------------------------------------------------------------------------
BLDG1_TTL = r"C:\Users\jabj\Documents\python\Datasets\mortar\mortargraphs\bldg1.ttl"
DB_CONFIG = {
    "table_name": "mortar_bldg1",
    "db_host": "localhost",
    "db_port": 5432,
    "db_name": "postgres",
    "db_user": "postgres",
    "db_password": "postgres",
}
TZ = ZoneInfo("America/Los_Angeles")
# 2-day batches with a 1-day stride: each window covers ``(d, d+2)``
# and the next window starts at ``d+1``, so the second day of every
# batch (= the one that survives the ``N_WARMUP`` slice below) lines
# up with day ``d+1``.  Together the 13 batches fully tile Jan 17-30
# in the *scored* portion while giving each batch a full 24-hour run
# to forget its initial conditions -- 1-day windows (the previous
# setup) were not long enough for the zone-temperature thermal mass
# to settle before the residual started being computed, which is
# what caused the long-horizon drift to look so bad in the
# single-window playback plot.
START_TIME = [datetime(2017, 1, d, tzinfo=TZ) for d in range(16, 28, 2)]
END_TIME = [datetime(2017, 1, d, tzinfo=TZ) for d in range(19, 31, 2)]
STEP_SIZE = 600  # 10-minute intervals
# Per-batch warmup in timesteps; 144 * 600s = 86400s = 1 day, so the
# first day of every 2-day batch is discarded from the Stage-3
# residual.  Only the *physics* estimator / plot use this -- the
# Stage-1 controller fit has no thermal mass to settle, so it stays
# at the :class:`Estimator` default (``n_warmup=60`` ~ 10 h, just
# enough to forget the PI integrator's initial value).
N_WARMUP = 144

# Absolute paths to previously saved estimation pickles -- one per
# stage that runs an :class:`Estimator`.  Set either to ``None`` to
# re-run that stage from scratch.  Stage 2 transfers the Stage-1
# parameters onto the closed-loop model regardless of whether they
# came from a fresh estimate or were just loaded from
# ``CTRL_PICKLE_FILE``.
CTRL_PICKLE_FILE: Optional[str] = r"C:\Users\jabj\Documents\python\Twin4Build\twin4build\examples\generated_files\models\bldg1_controls\model_parameters\estimation_results\20260515_150317_scipy_SLSQP_ad.pickle"
PHYSICS_PICKLE_FILE: Optional[str] = None#r"C:\Users\jabj\Documents\python\Twin4Build\twin4build\examples\generated_files\models\bldg1_physics\model_parameters\estimation_results\20260520_073744_scipy_SLSQP_ad.pickle"

# Per-stage diagnostic plots.  When ``True``, we re-simulate the
# matching model over the multi-batch ``START_TIME`` / ``END_TIME``
# window under ``saturation_mode("hard")`` (the saturation mode the
# estimator's final phase locks in) and draw predicted-vs-measured
# curves for every sensor that ``measurements="auto"`` selected -- see
# :meth:`Estimator._auto_measurements`.  Enabling
# ``PLOT_CTRL_MEASUREMENTS`` forces ``ctrl_model`` to be built even
# when ``CTRL_PICKLE_FILE`` is set, because the pickle alone has no
# model to simulate on.
PLOT_CTRL_MEASUREMENTS: bool = False
PLOT_PHYSICS_MEASUREMENTS: bool = True

# Per-Brick-class unit conversions, dispatched by class hierarchy in
# :meth:`SimulationModel.set_transformations`.
TRANSFORMATIONS = {
    core.namespace.BRICK.Temperature_Sensor:        lambda x: (x - 32) * 5 / 9,
    core.namespace.BRICK.Temperature_Setpoint:      lambda x: (x - 32) * 5 / 9,
    core.namespace.BRICK.Supply_Air_Flow_Sensor:    lambda x: x * 0.000578,
    core.namespace.BRICK.Command:                   lambda x: x / 100.0,
    core.namespace.BRICK.Damper_Position_Setpoint:  lambda x: x / 100.0,
    core.namespace.BRICK.Valve_Command:             lambda x: x / 100.0,
    core.namespace.BRICK.Percent_Air_Flow_Sensor:   lambda x: x / 100.0,
}

# Physics-side constants the BRICK graph does not carry, fanned out
# to every unwired matching input by ``fill_missing_inputs``.  The AHU
# SAT setpoint is *not* a scalar here -- see
# :func:`_build_measured_sat_overrides`, which sources it from the BMS
# Supply_Air_Temp measurement via a leaf sensor (no feedback loop).
PHYSICS_DEFAULTS = {
    "numberOfPeople":         0.0,
    "heatGain":               0.0,
    "outdoorCO2":             400.0,
    "supplyWaterTemperature": 60.0,
}


def _build_measured_sat_overrides(
    model: core.SimulationModel,
) -> Dict[Any, Any]:
    """``fill_missing_inputs`` overrides that source each unwired AHU
    ``supplyAirTemperatureSetpoint`` from the BMS ``Supply_Air_Temp``
    historian.

    The measurement sensor itself is a virtual pass-through of
    ``AHU.supplyAirTemperature`` -- using it directly would close a
    positive feedback loop.  Instead we clone its uuid / dbconfig /
    transformation into a NEW leaf :class:`SensorSystem` (no incoming
    simulator edges) and use that as the setpoint provider.

    Call AFTER ``set_dbconfigs`` + ``set_transformations`` (so the
    historian binding is in place to clone) and BEFORE
    ``fill_missing_inputs`` / ``rewire`` / ``load``.
    """
    def _has_incoming(comp, port_name: str) -> bool:
        for cp in getattr(comp, "connects_at", None) or []:
            if cp.input_port == port_name and (cp.connects_system_through or []):
                return True
        return False

    def _measurement_sensor_for(ahu) -> Optional[tb.SensorSystem]:
        for conn in getattr(ahu, "connected_through", None) or []:
            if conn.output_port != "supplyAirTemperature":
                continue
            for cp in conn.connects_system_at or []:
                downstream = cp.connection_point_of
                if (
                    isinstance(downstream, tb.SensorSystem)
                    and cp.input_port == "measuredValue"
                ):
                    return downstream
        return None

    overrides: Dict[Any, Any] = {}
    LOGGER.task("Stage 2 -- AHU SAT setpoint wiring from BMS measurement")
    for ahu in model.components.values():
        if not isinstance(ahu, tb.AirHandlingUnitSystem):
            continue
        if _has_incoming(ahu, "supplyAirTemperatureSetpoint"):
            LOGGER.info("%s already had SAT setpoint wired, untouched", ahu.id)
            continue
        meas = _measurement_sensor_for(ahu)
        if meas is None:
            LOGGER.warn(
                "%s has neither a setpoint URI nor a Supply_Air_Temp "
                "measurement sensor; SAT setpoint stays at port default. "
                "Add an explicit PHYSICS_DEFAULTS entry if this AHU "
                "should run a fixed deck.",
                ahu.id,
            )
            continue
        if not (meas.uuid or meas.filename or meas.df is not None):
            LOGGER.warn(
                "%s has a measurement sensor but no historian source "
                "(uuid / filename / df all None); SAT setpoint stays at "
                "port default.  Did ``set_dbconfigs`` run first?",
                ahu.id,
            )
            continue
        leaf = tb.SensorSystem(
            id=f"{meas.id}__as_setpoint",
            filename=meas.filename,
            df=meas.df,
            uuid=meas.uuid,
            dbconfig=meas.dbconfig,
            date_column=meas.datecolumn,
            value_column=meas.valuecolumn,
            use_spreadsheet=meas.use_spreadsheet,
            use_database=meas.use_database,
            use_df=meas.use_df,
            transformation=meas.transformation,
        )
        overrides[(ahu, "supplyAirTemperatureSetpoint")] = leaf
        LOGGER.info(
            "%s.supplyAirTemperatureSetpoint <- leaf SensorSystem cloned "
            "from %s (uuid + dbconfig forwarded)",
            ahu.id, meas.id,
        )
    return overrides


_STAGE3_PHYSICS_TYPES: Tuple[type, ...] = (
    tb.BuildingSpaceSystem,
    tb.FanCoilUnitSystem,
    tb.AirHandlingUnitSystem,
)


def _build_physics_parameter_list(
    model: core.SimulationModel,
) -> List[Tuple]:
    """Estimable physics parameters for Stage 3.

    Delegates to each physics component's
    :meth:`twin4build.core.System.get_estimable_parameters` contract:
    every BuildingSpace (RC + mass), every FanCoilUnit (UA / thermal
    mass / valve sizing), and every AirHandlingUnit (damper + fan
    nominals, heat-recovery effectiveness) contributes its own bounded
    tuples.  PI gains are NOT listed because the controller types are
    excluded from :data:`_STAGE3_PHYSICS_TYPES`: the Estimator only
    moves what is in the returned list, so the Stage-1 gains stay put.
    """
    params: List[Tuple] = []
    for comp in sorted(model.components.values(), key=lambda c: c.id):
        if not isinstance(comp, _STAGE3_PHYSICS_TYPES):
            continue
        params.extend(comp.get_estimable_parameters())
    return params


def _auto_discovered_measurements(
    model: core.SimulationModel,
) -> List[Tuple[Any, float]]:
    """Return the same ``(sensor, sd)`` list ``measurements="auto"``
    builds inside :meth:`Estimator.estimate`.

    Constructing a throw-away :class:`Estimator` just to call its
    discovery method keeps a single source of truth -- if the
    auto-resolution rules change (e.g. a new "wired data source"
    sentinel), the plot helper automatically follows.  The only side
    effect is a write to the throw-away estimator's
    ``_auto_measurement_ids`` cache, which is discarded with the
    instance.
    """
    return tb.Estimator(tb.Simulator(model))._auto_measurements()


def _plot_estimation_measurements(
    simulator: core.Simulator,
    measurements: List[Tuple[Any, float]],
    title: str,
    *,
    n_warmup: int = 60,
) -> None:
    """Predicted-vs-measured diagnostic for every sensor in
    ``measurements``, drawn one figure per sensor with
    :func:`tb.plot.plot`.

    The helper auto-detects the batch geometry from
    ``simulator.date_time_steps`` (shape ``(n_batches, n_t)``) so the
    caller controls "multi-batch vs single-window" by what it passed
    to ``simulator.simulate``:

    * **Multi-batch** (``start_time`` / ``end_time`` as lists, same
      structure :meth:`Estimator.estimate` uses): each batch starts
      fresh from BMS-supplied initial conditions, so state-space
      drift only has one batch-window's worth of time to accumulate.
      We strip ``n_warmup`` leading timesteps per batch -- mirroring
      the estimator's ``_obj`` which slices ``y_model_period[n_warmup:]``
      / ``y_actual_period[n_warmup:]`` -- and concatenate batches
      chronologically (day 0 first, day 1, ...) for both the plot and
      the per-sensor RMSE.  This is the apples-to-apples comparison
      to the RMSE the estimator was minimizing.

    * **Single-window** (``start_time`` / ``end_time`` as scalars):
      one continuous free-running prediction.  Useful for assessing
      how the model behaves *as a predictor* but not directly
      comparable to the estimator's RMSE -- slow biases compound over
      the full horizon and the temperatures will look much worse
      than they did during training.  ``n_warmup`` is still applied
      to the single batch.

    Per-sensor ``sd`` / MAE / RMSE go into each figure title, and an
    aggregate row plus a sorted table go to ``LOGGER.info`` so the
    estimator's training output and this diagnostic can be diffed
    line-by-line.
    """
    if not measurements:
        LOGGER.warning("%s: no measurements to plot", title)
        return

    # ``simulator.date_time_steps`` is always 2-D after
    # ``simulate``: shape ``(n_batches, n_t_per_batch)`` -- with
    # ``n_batches=1`` for a single-window run.  We slice the warmup
    # off each row and concatenate chronologically; the resulting
    # ``time`` array length is what tb.plot.plot's batch-size check
    # validates against ``data.shape[1]`` after the
    # ``isinstance(time, pd.Index)`` wrap below.
    dts = simulator.date_time_steps
    if dts.ndim == 1:
        dts = dts.reshape(1, -1)
    n_batches, n_t_per_batch = dts.shape
    if n_warmup >= n_t_per_batch:
        LOGGER.warning(
            "%s: n_warmup=%d >= n_t_per_batch=%d -- nothing left to plot, "
            "falling back to n_warmup=0", title, n_warmup, n_t_per_batch,
        )
        n_warmup = 0
    time_flat: List[Any] = []
    for b in range(n_batches):
        time_flat.extend(dts[b, n_warmup:].tolist())
    # ``tb.plot.plot``'s ``np.issubdtype(...datetime64)`` check rejects
    # ``object``-dtype ndarrays of tz-aware Python datetimes, so wrap
    # in a ``pd.DatetimeIndex`` -- the ``isinstance(time, pd.Index)``
    # branch then normalizes to ``[time]`` and matplotlib picks up
    # proper date-axis formatting for free.
    time = pd.DatetimeIndex(time_flat)

    # Per-sensor ``(id, sd, mae, rmse, n)`` collected during plotting
    # and logged as a sorted summary table.  ``pooled_rmse`` and
    # ``weighted_rmse`` mirror the estimator's residual structure:
    # the pooled value is the RMS over the *concatenated* residual
    # vector (so a sensor with more retained timesteps weighs more
    # than a short one), and the weighted variant further folds in
    # ``1/sd**2`` -- the closest scalar summary of the loss
    # :meth:`Estimator.estimate` was actually minimizing.
    stats: List[Tuple[str, float, float, float, int]] = []
    sse_pool = 0.0
    sse_weighted = 0.0
    n_pool = 0

    for sensor, sd in sorted(measurements, key=lambda x: x[0].id):
        try:
            pred_full = (
                sensor.input["measuredValue"]
                .history(i_c=0)
                .detach()
                .cpu()
                .numpy()
            )
            actual_full = (
                sensor.time_series_input.values[:, :, 0]
                .detach()
                .cpu()
                .numpy()
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "%s -- %s: cannot extract series (%s)",
                title, sensor.id, exc,
            )
            continue

        # Both arrays come back shape ``(n_t_per_batch, n_batches)``
        # for multi-batch sims and ``(n_t_per_batch, 1)`` for a single
        # window.  Drop the per-batch warmup, then concatenate
        # chronologically (day 0 timesteps first, then day 1, ...) by
        # transposing to ``(n_batches, n_t_keep)`` and ``flatten()``
        # in row-major order.
        if pred_full.ndim == 1:
            pred_full = pred_full[:, None]
        if actual_full.ndim == 1:
            actual_full = actual_full[:, None]
        pred_concat = pred_full[n_warmup:, :].T.flatten()
        actual_concat = actual_full[n_warmup:, :].T.flatten()

        residual = pred_concat - actual_concat
        sq = residual ** 2
        n_valid = int(np.sum(~np.isnan(residual)))
        mae = float(np.nanmean(np.abs(residual)))
        rmse = float(np.sqrt(np.nanmean(sq)))
        stats.append((sensor.id, float(sd), mae, rmse, n_valid))
        sse_pool += float(np.nansum(sq))
        sse_weighted += float(np.nansum(sq)) / (float(sd) ** 2)
        n_pool += n_valid

        tb.plot.plot(
            time=time,
            entries=[
                tb.plot.Entry(
                    data=actual_concat.reshape(1, -1),
                    label="Measured",
                    color=Colors.black,
                    fmt="-",
                ),
                tb.plot.Entry(
                    data=pred_concat.reshape(1, -1),
                    label="Predicted",
                    color=Colors.blue,
                    fmt="-",
                ),
            ],
            ylabel_1axis=sensor.id,
            title=(
                f"{title} -- {sensor.id} "
                f"(sd={sd:.4g}, MAE={mae:.4g}, RMSE={rmse:.4g})"
            ),
        )

    if not stats:
        LOGGER.warning("%s: no per-sensor residuals could be computed", title)
        return

    sensor_id_width = max(len(s[0]) for s in stats)
    LOGGER.task("%s -- per-sensor residuals (sorted by RMSE desc)", title)
    LOGGER.info(
        "%-*s  %10s  %10s  %10s  %8s",
        sensor_id_width, "sensor", "sd", "MAE", "RMSE", "n",
    )
    for sid, sd, mae, rmse, n_valid in sorted(stats, key=lambda x: -x[3]):
        LOGGER.info(
            "%-*s  %10.4g  %10.4g  %10.4g  %8d",
            sensor_id_width, sid, sd, mae, rmse, n_valid,
        )

    rmse_arr = np.array([s[3] for s in stats], dtype=float)
    pooled_rmse = float(np.sqrt(sse_pool / max(n_pool, 1)))
    weighted_rmse = float(np.sqrt(sse_weighted / max(n_pool, 1)))
    LOGGER.info(
        "%s -- aggregate: n_sensors=%d, pooled RMSE=%.4g, weighted RMSE=%.4g, "
        "mean RMSE=%.4g, median RMSE=%.4g, max RMSE=%.4g",
        title,
        len(stats),
        pooled_rmse,
        weighted_rmse,
        float(rmse_arr.mean()),
        float(np.median(rmse_arr)),
        float(rmse_arr.max()),
    )


if __name__ == "__main__":
    sm = tb.SemanticModel(rdf_file=BLDG1_TTL, id="bldg1", verbose=1500)

    # -----------------------------------------------------------------
    # Stage 1 -- per-zone PI controller identification
    # -----------------------------------------------------------------
    # ``ctrl_model`` is built whenever we either need to fit it
    # (``CTRL_PICKLE_FILE is None``) or need a model to simulate on for
    # the Stage-1 diagnostic plot (``PLOT_CTRL_MEASUREMENTS``).  When a
    # pickle is provided AND no plot is requested, we skip the
    # controller translation entirely and just load the pickle straight
    # onto ``full_model`` below.
    ctrl_result = None
    ctrl_model = None
    need_ctrl_model = (CTRL_PICKLE_FILE is None) or PLOT_CTRL_MEASUREMENTS
    if need_ctrl_model:
        LOGGER.task("Stage 1 -- building controller-identification model")
        ctrl_model = tb.Translator().translate(
            sm,
            systems_=[
                tb.ControllerIdentificationPISystem,
                tb.SensorSystem,
            ],
            id="bldg1_controls",
        )
        ctrl_model.set_dbconfigs(DB_CONFIG)
        ctrl_model.set_transformations(TRANSFORMATIONS)
        ctrl_model.rewire(
            start_time=START_TIME,
            end_time=END_TIME,
            step_size=STEP_SIZE,
            mode="train",
        )
        ctrl_model.load(draw_semantic_model=False, draw_simulation_model=False)
        if CTRL_PICKLE_FILE is None:
            # PI loops have no thermal mass to settle, so we leave
            # ``n_warmup`` at the :class:`Estimator` default (60
            # timesteps = 10 h, just enough to forget the integrator's
            # initial value).  The 1-day ``N_WARMUP`` only kicks in
            # on the Stage-3 physics estimate / plot.
            LOGGER.task("Stage 1 -- controller identification (estimate)")
            ctrl_result = tb.Estimator(tb.Simulator(ctrl_model)).estimate(
                start_time=START_TIME,
                end_time=END_TIME,
                step_size=STEP_SIZE,
                parameters="auto",
                measurements="auto",
                method=("scipy", "SLSQP", "ad"),
                options={"x_scale": "jac"},
                schedule=[
                    {"saturation_mode": "smooth"},
                    {"saturation_mode": "hard"},
                ],
            )
        else:
            LOGGER.task(
                "Stage 1 -- loading estimation pickle %s onto ctrl_model "
                "(needed for plot)", CTRL_PICKLE_FILE,
            )
            ctrl_model.load_estimation_result(
                filename=CTRL_PICKLE_FILE, verbose=1
            )
        if PLOT_CTRL_MEASUREMENTS:
            # Single-window playback so the plot shows one continuous
            # trace per sensor.  ``saturation_mode("hard")`` matches
            # the estimator's final phase.  Per-sensor RMSEs reported
            # here are *not* directly comparable to the estimator's
            # multi-batch RMSE -- slow biases in the closed-loop
            # physics compound over the full 14-day horizon -- but
            # the figures show the model's free-running behavior,
            # which is what you actually want to inspect when judging
            # whether the calibrated parameters produce plausible
            # trajectories.
            LOGGER.task("Stage 1 -- simulating ctrl_model for plot")
            ctrl_sim = tb.Simulator(ctrl_model)
            with saturation_mode("hard"):
                ctrl_sim.simulate(
                    start_time=START_TIME[0],
                    end_time=END_TIME[-1],
                    step_size=STEP_SIZE,
                )
            # Discover AFTER simulate so ``time_series_input.values``
            # is populated -- otherwise every sd falls back to
            # ``AUTO_SD_FLOOR`` rather than ``0.1 * data_std``.
            ctrl_measurements = _auto_discovered_measurements(ctrl_model)
            _plot_estimation_measurements(
                ctrl_sim,
                ctrl_measurements,
                title="Stage 1 -- controller predictions vs BMS",
            )
    else:
        LOGGER.task("Stage 1 -- loading estimation pickle %s", CTRL_PICKLE_FILE)

    # -----------------------------------------------------------------
    # Stage 2 -- full physics + controllers, Stage-1 parameter transfer
    # -----------------------------------------------------------------
    LOGGER.task("Stage 2 -- full physics model + Stage-1 parameter transfer")
    full_model = tb.Translator().translate(
        sm,
        systems_=[
            tb.BuildingSpaceSystem,
            tb.AirHandlingUnitSystem,
            tb.OutdoorEnvironmentSystem,
            tb.FanCoilUnitSystem,
            tb.SensorSystem,
            tb.ControllerIdentificationPISystem,
        ],
        id="bldg1_physics",
    )
    full_model.set_dbconfigs(DB_CONFIG)
    full_model.set_transformations(TRANSFORMATIONS)
    full_model.fill_missing_inputs({
        **PHYSICS_DEFAULTS,
        **_build_measured_sat_overrides(full_model),
    })
    full_model.rewire(
        start_time=START_TIME,
        end_time=END_TIME,
        step_size=STEP_SIZE,
        mode="simulate",
    )
    full_model.load(draw_semantic_model=False, draw_simulation_model=False)
    if ctrl_result is not None:
        full_model.load_estimation_result(result=ctrl_result, verbose=1)
    else:
        full_model.load_estimation_result(filename=CTRL_PICKLE_FILE, verbose=1)

    # -----------------------------------------------------------------
    # Stage 3 -- estimate physics parameters on the closed loop
    # -----------------------------------------------------------------
    # BuildingSpace RC + mass, FCU UA / valve sizing, AHU damper /
    # fan / heat-recovery nominals -- all discovered via each
    # component's ``get_estimable_parameters()`` contract (see
    # :func:`_build_physics_parameter_list`).  PI gains stay frozen at
    # the Stage-1 values because the CITS type is excluded from
    # :data:`_STAGE3_PHYSICS_TYPES`; the Estimator only moves what is in
    # the parameter list.  ``measurements="auto"`` picks every sensor
    # with a non-sensor upstream and a wired data source, which
    # includes the zone-temperature sensors driven by
    # ``BuildingSpace.indoorTemperature`` -- the fit target.
    if PHYSICS_PICKLE_FILE is None:
        LOGGER.task("Stage 3 -- physics parameter estimation")
        physics_result = tb.Estimator(tb.Simulator(full_model)).estimate(
            start_time=START_TIME,
            end_time=END_TIME,
            step_size=STEP_SIZE,
            parameters=_build_physics_parameter_list(full_model),
            measurements="auto",
            n_warmup=N_WARMUP,
            method=("scipy", "SLSQP", "ad"),
            options={"x_scale": "jac"},
        )
        full_model.load_estimation_result(result=physics_result, verbose=1)
    else:
        LOGGER.task(
            "Stage 3 -- loading physics estimation pickle %s",
            PHYSICS_PICKLE_FILE,
        )
        full_model.load_estimation_result(
            filename=PHYSICS_PICKLE_FILE, verbose=1
        )

    if PLOT_PHYSICS_MEASUREMENTS:
        # Single-window playback so the plot shows one continuous
        # trace per sensor rather than a stack of disjoint per-batch
        # windows.  ``saturation_mode("hard")`` matches the estimator's
        # final phase, which is also the mode the Stage-1 ctrl plot
        # uses -- keeps the two diagnostics directly comparable.
        LOGGER.task("Stage 3 -- simulating full_model for plot")
        physics_sim = tb.Simulator(full_model)
        with saturation_mode("hard"):
            physics_sim.simulate(
                start_time=START_TIME[0],
                end_time=END_TIME[-1],
                step_size=STEP_SIZE,
            )
        # Discover AFTER simulate so ``time_series_input.values`` is
        # populated -- otherwise every sd falls back to
        # ``AUTO_SD_FLOOR`` rather than ``0.1 * data_std``.
        physics_measurements = _auto_discovered_measurements(full_model)
        _plot_estimation_measurements(
            physics_sim,
            physics_measurements,
            title="Stage 3 -- physics predictions vs BMS",
            n_warmup=N_WARMUP,
        )

    # -----------------------------------------------------------------
    # Final replay-simulate on the fully calibrated model
    # -----------------------------------------------------------------
    # Single-window playback (rewire above used the multi-batch list to
    # set the n_s dimension on every Scalar; a single-window simulate
    # collapses to the first batch row).
    # with saturation_mode("hard"):
    #     tb.Simulator(full_model).simulate(
    #         start_time=datetime(2017, 1, 16, tzinfo=TZ),
    #         end_time=datetime(2017, 1, 27, tzinfo=TZ),
    #         step_size=STEP_SIZE,
    #     )

    if PLOT_CTRL_MEASUREMENTS or PLOT_PHYSICS_MEASUREMENTS:
        plt.show()
