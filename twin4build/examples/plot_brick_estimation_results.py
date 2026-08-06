"""
Plot time-series predictions vs actuals from the whole-building CITS estimation.

Loads the serialized simulation model from RDF, applies estimated parameters
via load_estimation_result, runs a final simulation, then produces per-room
plots of predicted vs measured.

IMPORTANT -- match identify's gate-swap timing
----------------------------------------------
The BandGate swap below MUST happen AFTER CITS._build_components() has run.
_build_components() unconditionally (re)creates gate_a as a fresh
SigmoidGate, so any BandGate set before that call is silently overwritten.
This script calls _build_components() explicitly in the "Build CITS
internals" loop BEFORE the gate swap, which is the correct order.  The
identify script must also follow the same order; if it runs the swap
before _build_components (as an earlier version did), the estimator
trains against a SigmoidGate while the plot script simulates with a
BandGate, and the two RMSE numbers will disagree -- mainly on reheat,
whose BandGate band ``[19, 20.5] C`` is much narrower than the damper's
``[19, 23] C``.
"""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import re
import numpy as np
import torch
import matplotlib.pyplot as plt

import twin4build as tb
from twin4build.systems.controller.controller_identification.controller_identification_system import (
    ControllerIdentificationSystem,
)
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.systems.utils.sigmoid_gate import BandGate
from twin4build.utils.rgetattr import rgetattr

# ---- Architecture constants (must match identify_mortar_bldg1_from_brick.py) ----
DAMPER_T_LO, DAMPER_T_HI = 19.0, 23.0   # heating + cooling occupied
REHEAT_T_LO, REHEAT_T_HI = 19.0, 20.5   # heating occupied only
GATE_STEEPNESS = 5.0

# ── Configuration ────────────────────────────────────────────────────────────

# Two pickle producers live alongside this plotter:
#   - identify_mortar_bldg1_from_brick.py   -> writes under   models/bldg1/
#   - translator_example_mortar_bldg1.py    -> writes under   models/bldg1_controls/
# CITS IDs differ between the two (friendly sensor names vs. translator-
# generated hash IDs), so a pickle from one will NEVER `load_estimation_result`
# cleanly into the other's RDF.  Pick the model id whose pickle you want
# to plot; both folders have their own RDF + pickle pair.
MODEL_ID = "bldg1_controls"
# Anchor all paths to the repo root so the script works regardless of the
# directory it is launched from (project root, twin4build/examples/, etc.).
# This file lives at <repo>/twin4build/examples/plot_brick_estimation_results.py
# so parents[2] == <repo>.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = (
    _REPO_ROOT / "twin4build" / "examples" / "generated_files" / "models"
    / MODEL_ID / "simulation_model"
)

RDF_FILE = str(_MODEL_DIR / "semantic_model" / "instance_graph.ttl")

# By default, pick the most recently written pickle in the estimation_results
# folder so the plot always matches the last identify-script run.  To plot a
# specific historical run, replace this with an explicit file path, e.g.:
#   PICKLE_FILE = str(_MODEL_DIR / "..." / "20260420_074617_scipy_SLSQP_ad.pickle")
_PICKLE_DIR = _MODEL_DIR / "model_parameters" / "estimation_results"
_pickle_candidates = sorted(
    _PICKLE_DIR.glob("*.pickle"), key=lambda p: p.stat().st_mtime
)
if not _pickle_candidates:
    raise FileNotFoundError(
        f"No .pickle files found in {_PICKLE_DIR}. Run the identify script first."
    )
PICKLE_FILE = str(_pickle_candidates[-1])
PICKLE_FILE = r"C:\Users\jabj\Documents\python\Twin4Build\twin4build\examples\generated_files\models\bldg1_controls\simulation_model\model_parameters\estimation_results\20260513_161404_scipy_SLSQP_ad.pickle"
print(f"Using pickle: {Path(PICKLE_FILE).name}")

# Set APPLY_PICKLE = False to visualise the INITIAL simulation (only x0 values
# applied, pickle NOT loaded).  Useful as a sanity check: it shows what the
# model predicts at the starting point SLSQP saw, so we can tell whether the
# optimiser was given a reasonable basin to descend from or not.
APPLY_PICKLE = True

# Optional debug override: after the pickle has been loaded, force every CITS'
# ``candidate_0_0.output_min`` parameter to this value before the final
# simulation runs.  Set to ``None`` to disable (use the trained value).
# Set to ``0.0`` to remove the PID lower clamp entirely -- useful for
# answering "would the fit be better if output_min were not estimated?".
OVERRIDE_OUTPUT_MIN: float | None = 0




DB_CONFIG = {
    "table_name": "mortar_bldg1",
    "db_host": "localhost",
    "db_port": 5432,
    "db_name": "postgres",
    "db_user": "postgres",
    "db_password": "postgres",
}

TZ = ZoneInfo("America/Los_Angeles")
STEP_SIZE = 600

START_TIME = [
    datetime(2017, 1, 16, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 17, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 18, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 19, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 20, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 21, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 22, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 23, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 24, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 25, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 26, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 27, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 28, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 29, 0, 0, tzinfo=TZ),
]
END_TIME = [
    datetime(2017, 1, 17, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 18, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 19, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 20, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 21, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 22, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 23, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 24, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 25, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 26, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 27, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 28, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 29, 0, 0, tzinfo=TZ),
    datetime(2017, 1, 30, 0, 0, tzinfo=TZ),
]

# ── 1) Load model from serialized RDF ───────────────────────────────────────

print("=" * 70)
print("Loading model from serialized RDF...")
print("=" * 70)

sim_model = tb.SimulationModel(id=MODEL_ID)

def _configure_sensors(model):
    """fcn callback: set dbconfig and transformations on all SensorSystems."""
    transformation_temp = lambda x: (x - 32) * 5 / 9
    transformation_pct = lambda x: x / 100.0
    for comp in model.components.values():
        if isinstance(comp, SensorSystem):
            comp.dbconfig = DB_CONFIG
            name = comp.id.lower()
            if any(kw in name for kw in ("command", "valve", "damper")):
                comp._transformation = transformation_pct
            elif any(kw in name for kw in ("temp", "setpoint", "control_temp")):
                comp._transformation = transformation_temp

sim_model.load(rdf_file=RDF_FILE, fcn=_configure_sensors)

# ── 2) Build CITS internals (lazy build from connections) ────────────────────

transformation_pct = lambda x: x / 100.0

cits_components = {
    cid: c
    for cid, c in sim_model.components.items()
    if isinstance(c, ControllerIdentificationSystem)
}
print(f"\nFound {len(cits_components)} CITS components.")

for cid, c in cits_components.items():
    if not c._built:
        n_s = c.get_n_v_from_connections("sensorValue")
        n_sp = c.get_n_v_from_connections("setpointValue")
        # ``n_on_off_signals`` is not yet serialised in the older CITS JSON
        # configs, so the load step above leaves it as ``None`` (validation
        # logs "Missing parameters in file config: parameters.n_on_off_signals").
        # Resolve it the same way as the translator example's
        # ``_prepare_stage1_model``: from the number of ``onOffSignal``
        # connection slots.  Without this, ``_build_components`` calls
        # ``torch.full((None,), ...)`` and raises a TypeError.
        n_oo = c.get_n_v_from_connections("onOffSignal")
        n_act = c._get_n_actuators_from_connections()
        if n_s is not None:
            c.n_sensors = n_s
        if n_sp is not None:
            c.n_setpoints = n_sp
        if n_oo is not None:
            c.n_on_off_signals = n_oo
        if n_act is not None:
            c.n_actuators = n_act
        c._build_components()
        print(
            f"  Built {cid}: sensors={c.n_sensors}, setpoints={c.n_setpoints}, "
            f"on_off_signals={c.n_on_off_signals}, actuators={c.n_actuators}"
        )

# ---- Swap every CITS's default SigmoidGate for a BandGate ---------------------
# The identify script estimates both BandGate edges (T_lo = `.threshold`,
# T_hi = `.threshold_high`) per CITS, and both values are serialised in the
# pickle -- so the hardcoded (T_lo, T_hi) below is only a PRIOR that exists
# long enough to install a BandGate on each CITS; `load_estimation_result`
# further down overwrites both values with the per-zone learned band before
# any simulation happens.  We therefore only need sensible generic defaults
# here (not per-CITS-accurate ones) for the model to be buildable.

def _actuator_type(cits_id: str, ctrl=None) -> str:
    """Damper-vs-reheat detection.

    The translator-generated CITS IDs are opaque hash strings like
    ``[VRM115][l_Temp]...3fa8cd72208fb283`` with no "damper" substring,
    so we look at the *downstream measurement sensor* IDs first
    (those keep friendly names like ``..._Zone_Air_Damper_Command``).
    Fallback to the CITS id for legacy identify-script pickles where
    the CITS id itself encodes the actuator kind.
    """
    if ctrl is not None:
        for conn in ctrl.connected_through:
            if conn.output_port != "inputSignal":
                continue
            for cp in conn.connects_system_at:
                meas_id = cp.connection_point_of.id.lower()
                if "damper" in meas_id:
                    return "damper"
                if "reheat" in meas_id or "valve" in meas_id:
                    return "reheat"
    return "damper" if "damper" in cits_id.lower() else "reheat"

print("\nInstalling BandGate on every CITS (priors; will be overwritten by pickle)...")
for cid, ctrl in cits_components.items():
    atype = _actuator_type(cid, ctrl)
    if atype == "damper":
        t_lo, t_hi = DAMPER_T_LO, DAMPER_T_HI
    else:
        t_lo, t_hi = REHEAT_T_LO, REHEAT_T_HI
    ctrl.gate_0 = BandGate(
        threshold=t_lo,
        threshold_high=t_hi,
        steepness=GATE_STEEPNESS,
        id=f"{ctrl.id}_gate_0",
    )
    print(f"  {cid} ({atype}): BandGate prior [{t_lo}, {t_hi}] C")

def discover_cits_topology(controller):
    sensors, setpoints, actuators = [], [], []
    for cp in controller.connects_at:
        for conn in cp.connects_system_through:
            src = conn.connects_system
            if cp.input_port == "sensorValue":
                sensors.append((cp.input_port_index.get(conn), src))
            elif cp.input_port == "setpointValue":
                setpoints.append((cp.input_port_index.get(conn), src))
    for conn in controller.connected_through:
        if conn.output_port == "inputSignal":
            for cp in conn.connects_system_at:
                idx = cp.output_port_index.get(conn)
                actuators.append((idx, cp.connection_point_of))
    sensors.sort(key=lambda x: (x[0] is None, x[0]))
    setpoints.sort(key=lambda x: (x[0] is None, x[0]))
    actuators.sort(key=lambda x: (x[0] is None, x[0]))
    return sensors, setpoints, actuators

cits_info = {}
for cits_id, controller in sorted(cits_components.items()):
    sensor_list, setpoint_list, actuator_list = discover_cits_topology(controller)
    atype = _actuator_type(cits_id, controller)

    # Fix blank-node actuator sensor transformations
    for _, s in actuator_list:
        s._transformation = transformation_pct

    zone_temp_sensor = None
    temp_setpoint_sensor = None
    for _, s in sensor_list:
        name = s.id.lower()
        if "zone_air_temp" in name and "supply" not in name and "control" not in name \
                and "setpoint" not in name:
            zone_temp_sensor = s
    for _, s in setpoint_list:
        name = s.id.lower()
        if "temp_setpoint" in name or "air_temp_setpoint" in name:
            temp_setpoint_sensor = s

    cits_info[cits_id] = {
        "controller": controller,
        "atype": atype,
        "actuator_list": actuator_list,
        "sensor_list": sensor_list,
        "setpoint_list": setpoint_list,
        "zone_temp_sensor": zone_temp_sensor,
        "temp_setpoint_sensor": temp_setpoint_sensor,
    }

# ---- Sensor diagnostic: print names only (data ranges printed after sim) --
# ``time_series_input.values`` is populated lazily by ``Simulator.simulate``,
# so before the sim runs every sensor reports "<no data>".  Print just the
# topology (names + selected zone_temp/temp_setpoint markers) here, and
# show ranges later in the post-sim diagnostic block.
print("\nSensor topology per CITS (data ranges printed after simulation):")
for cits_id, info in sorted(cits_info.items()):
    print(f"\n  {cits_id}")
    print(f"    sensorValue[] inputs ({len(info['sensor_list'])}):")
    for idx, s in info["sensor_list"]:
        marker = "  <-- zone_temp" if s is info["zone_temp_sensor"] else ""
        print(f"      [{idx}] {s.id}{marker}")
    print(f"    setpointValue[] inputs ({len(info['setpoint_list'])}):")
    for idx, s in info["setpoint_list"]:
        marker = "  <-- temp_setpoint" if s is info["temp_setpoint_sensor"] else ""
        print(f"      [{idx}] {s.id}{marker}")

# ── 3) Apply frozen selection weights (not stored in pickle) ─────────────────
# The pickle only contains *estimated* parameters. Frozen weights (alpha, beta,
# gamma, beta_b, output_min, non-selected candidate params) must be set to the
# same x0 values used during estimation.

def _find_sensor_idx(sensor_list, *keywords):
    for idx, s in sensor_list:
        name = s.id.lower()
        if any(kw in name for kw in keywords):
            return idx
    return None

def _find_setpoint_idx(setpoint_list, *keywords):
    for idx, s in setpoint_list:
        name = s.id.lower()
        if any(kw in name for kw in keywords):
            return idx
    return None

def _one_hot(n, idx):
    v = [0.0] * n
    if idx is not None and 0 <= idx < n:
        v[idx] = 1.0
    return v

def _apply_x0(attr, atype, n_c, n_s, n_sp, sensor_indices, setpoint_indices):
    """Mirror identify_mortar_bldg1_from_brick.py's frozen x0 exactly.

    Only the PID parameters of candidate_0_0 (kp, Ti, damper output_min) come
    from the pickle; everything else -- selection weights, gate params, non-
    selected candidate params -- is frozen to the values below.
    """
    zone_temp_idx = sensor_indices.get("zone_temp", 0)
    temp_sp_idx = setpoint_indices.get("temp_setpoint", n_sp - 1)

    # --- Candidate selection: candidate_0_0 (PID reverse) for both types ---
    if attr.startswith("alpha_") and not attr.startswith("alpha_gate_"):
        return [1.0] + [0.0] * (n_c - 1)

    # --- Sensor selection: zone_temp for both ------------------------------
    if attr.startswith("beta_") and not attr.startswith("beta_b_"):
        return _one_hot(n_s, zone_temp_idx)
    if attr.startswith("gamma_") and not attr.startswith("gamma_gate_"):
        return _one_hot(n_sp, temp_sp_idx)
    if attr.startswith("beta_b_"):
        return _one_hot(n_s, zone_temp_idx)

    # --- Gate parameters (all frozen, BandGate) ----------------------------
    if attr.startswith("alpha_gate_"):
        return 1.0
    if attr.startswith("gamma_gate_"):
        return _one_hot(n_sp, temp_sp_idx)
    if attr.endswith(".threshold"):
        return 19.0                              # T_lo (same for damper & reheat)
    if attr.endswith(".threshold_high"):
        # T_hi -- differs per atype; same priors as identify_mortar_bldg1_from_brick.py
        return DAMPER_T_HI if atype == "damper" else REHEAT_T_HI
    if attr.endswith(".polarity"):
        return 1.0                               # unused by BandGate
    if attr.startswith("default_output_"):
        return 1.0 if atype == "damper" else 0.0

    # --- PID (candidate_0_0) x0 --------------------------------------------
    # These are only used if the pickle does NOT contain these params.
    # For a run produced by the new from_brick script, the pickle WILL
    # contain candidate_0_0.{kp,Ti} and damper candidate_0_0.output_min,
    # so these x0 values get overwritten.  They are kept for robustness.
    if re.match(r"candidate_\d+_0\.kp$", attr):
        return 5.0 if atype == "reheat" else 0.05
    if re.match(r"candidate_\d+_0\.Ti$", attr):
        return 100.0 if atype == "reheat" else 1800.0
    if re.match(r"candidate_\d+_0\.output_min$", attr):
        return 0.45 if atype == "damper" else 0.0
    return None

print("\nApplying frozen x0 values (selection weights, non-estimated params)...")
for cits_id, info in cits_info.items():
    controller = info["controller"]
    atype = info["atype"]
    sensor_list = info["sensor_list"]
    setpoint_list = info["setpoint_list"]
    n_s = controller.n_sensors
    n_sp = controller.n_setpoints
    n_c = controller.n_candidates

    sensor_indices = {}
    zt = _find_sensor_idx(sensor_list, "zone_air_temp")
    if zt is not None:
        sensor_indices["zone_temp"] = zt
    af = _find_sensor_idx(sensor_list, "percent_air_flow", "air_flow_sensor")
    if af is not None:
        sensor_indices["airflow"] = af
    setpoint_indices = {}
    tsp = _find_setpoint_idx(setpoint_list, "temp_setpoint", "air_temp_setpoint")
    if tsp is not None:
        setpoint_indices["temp_setpoint"] = tsp

    for p in controller.get_estimator_parameters():
        comp, attr, x0 = p[0], p[1], p[2]
        override = _apply_x0(attr, atype, n_c, n_s, n_sp, sensor_indices, setpoint_indices)
        if override is not None:
            param = rgetattr(comp, attr)
            param.set(torch.tensor(override, dtype=torch.float64), normalized=False)

print("  Done.")

# ── 4) Load estimated parameters (overwrites only estimated subset) ──────────

if APPLY_PICKLE:
    print(f"\nLoading estimation results from pickle...")
    sim_model.load_estimation_result(filename=PICKLE_FILE, verbose=0)
else:
    print("\nAPPLY_PICKLE=False -- skipping pickle load; plotting INITIAL x0 sim.")

# ── 4b) Optional override: force candidate_0_0.output_min on every CITS ──────
# Applied AFTER the pickle load so it overwrites the trained value cleanly.
# Use this to test whether removing the PID lower clamp (output_min=0)
# materially changes the per-actuator RMSE / time series.

if OVERRIDE_OUTPUT_MIN is not None:
    print(
        f"\nOverriding candidate_0_0.output_min = {OVERRIDE_OUTPUT_MIN} "
        f"on every CITS (post-pickle override)..."
    )
    n_overridden = 0
    for cid, comp in sim_model.components.items():
        if not isinstance(comp, ControllerIdentificationSystem):
            continue
        # Each CITS has actuator-indexed candidates ``candidate_{a}_0`` etc.;
        # mortar_bldg1 sets ``n_actuators = 1``, so only ``candidate_0_0``
        # exists.  We loop defensively over the first 4 actuators to cover
        # any future config that wires up additional actuators per CITS.
        for a in range(4):
            attr = f"candidate_{a}_0.output_min"
            try:
                p = rgetattr(comp, attr)
            except AttributeError:
                continue
            p.set(
                torch.tensor(float(OVERRIDE_OUTPUT_MIN), dtype=torch.float64),
                normalized=False,
            )
            n_overridden += 1
    print(f"  Overrode {n_overridden} parameter(s).")

# ── 5) Run final simulation ─────────────────────────────────────────────────

_sim_label = "estimated parameters" if APPLY_PICKLE else "INITIAL x0 values"
print(f"\nRunning simulation with {_sim_label}...")
simulator = tb.Simulator(sim_model)
# IMPORTANT: match the saturation mode used during identification.  The
# pickle was produced under ``schedule=[{"saturation_mode": "hard"}]`` in
# ``translator_example_mortar_bldg1.py``, so re-running the forward sim
# under the global default ("smooth") produces a *different* trajectory
# from the one the optimizer was minimizing -- the per-actuator RMSE
# reported here will differ from the estimator's final loss, and the
# plotted valve command will sit on a smooth asymptotic floor near
# ``output_min + curve_start`` instead of the exact hard clamp at
# ``output_min``.  Wrap the simulate call in the same hard-mode context
# the estimator used so the plot represents what was actually fit.
from twin4build.systems.utils.smooth_saturation import saturation_mode
with saturation_mode("hard"):
    simulator.simulate(
        start_time=START_TIME, end_time=END_TIME, step_size=STEP_SIZE,
    )
print("  Done.")

# ---- Post-sim sensor data ranges (now that values are populated) ----------
print("\nSensor data ranges per CITS (post-simulation):")
for cits_id, info in sorted(cits_info.items()):
    print(f"\n  {cits_id}")
    for kind, lst in (("sensorValue", info["sensor_list"]),
                       ("setpointValue", info["setpoint_list"])):
        print(f"    {kind}[] ({len(lst)}):")
        for idx, s in lst:
            try:
                arr = s.time_series_input.values[:, :, 0].detach().numpy()
                stats = (
                    f"min={arr.min():.3f}, max={arr.max():.3f}, "
                    f"mean={arr.mean():.3f}"
                )
            except Exception as exc:
                stats = f"<no data: {exc}>"
            print(f"      [{idx}] {s.id}  ({stats})")

# ── 6) Plot ──────────────────────────────────────────────────────────────────

def _room_name(cits_id):
    """Extract room code (e.g. ``RM107A``) from any CITS id variant.

    Handles three id shapes found in this codebase:
      - Legacy identify-script names containing ``_RM107A_`` literally.
      - Translator hashed ids ``[RM107A][...]`` / ``[VRM115][...]``.
      - Translator hashed ids ``[AVRM107A][...]`` / ``[VAVRM115][...]`` --
        the actuator-prefixed partners of the above; the leading ``A`` /
        ``V`` tags would defeat a ``startswith("RM")`` check, so we use
        a substring search for ``RM<digits>[<letter>]`` anywhere inside
        the id.  This is what pairs ``(AVRM107A, RM107A)`` and
        ``(VAVRM115, VRM115)`` onto the same figure (one for damper,
        one for reheat).
    """
    m = re.search(r"RM\d+[A-Z]?", cits_id)
    if m:
        return m.group(0)
    return cits_id

rooms = {}
for cits_id, info in sorted(cits_info.items()):
    room = _room_name(cits_id)
    rooms.setdefault(room, {})[info["atype"]] = (cits_id, info)

n_periods = len(START_TIME)

for room, actuators in sorted(rooms.items()):
    n_act = len(actuators)
    fig, axes = plt.subplots(n_act, 1, figsize=(18, 4.5 * n_act), sharex=True)
    if n_act == 1:
        axes = [axes]

    for ax_idx, (atype, (cits_id, info)) in enumerate(
        sorted(actuators.items(), key=lambda x: (0 if x[0] == "damper" else 1))
    ):
        ax = axes[ax_idx]
        ax2 = ax.twinx()

        act_sensor = info["actuator_list"][0][1] if info["actuator_list"] else None
        if act_sensor is None:
            ax.set_title(f"{room} {atype} -- no actuator sensor")
            continue

        pred = act_sensor.input["measuredValue"].history(i_c=0).detach().numpy().T
        actual = act_sensor.time_series_input.values[:, :, 0].detach().numpy().T

        mae = np.mean(np.abs(pred - actual))
        rmse = np.sqrt(np.mean((pred - actual) ** 2))

        pred_flat = pred.flatten()
        actual_flat = actual.flatten()
        x_axis = np.arange(len(pred_flat))

        color_pred = "#E85C4C" if atype == "reheat" else "#4C9BE8"
        ax.plot(x_axis, actual_flat, color="black", alpha=0.7, linewidth=0.8,
                label="Measured")
        ax.plot(x_axis, pred_flat, color=color_pred, alpha=0.85, linewidth=0.8,
                label="Predicted")

        n_steps = pred.shape[1]
        for p_idx in range(1, n_periods):
            ax.axvline(x=p_idx * n_steps, color="gray", linewidth=0.4, alpha=0.5)

        zt_sensor = info["zone_temp_sensor"]
        sp_sensor = info["temp_setpoint_sensor"]
        if zt_sensor is not None:
            zt = zt_sensor.time_series_input.values[:, :, 0].detach().numpy().T.flatten()
            ax2.plot(x_axis, zt, color="green", alpha=0.35, linewidth=0.7,
                     label="Zone Temp")
        if sp_sensor is not None:
            sp = sp_sensor.time_series_input.values[:, :, 0].detach().numpy().T.flatten()
            ax2.plot(x_axis, sp, color="green", alpha=0.35, linewidth=0.7,
                     linestyle="--", label="Setpoint")

        label = "Reheat Valve" if atype == "reheat" else "Damper Position"
        ax.set_title(f"{room} {label}  (MAE={mae:.4f}, RMSE={rmse:.4f})")
        ax.set_ylabel("Position (0-1)")
        ax.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("Temperature (C)")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)

    _title_tag = "Identified" if APPLY_PICKLE else "Initial x0"
    fig.suptitle(f"Room {room}: {_title_tag} vs Measured", fontsize=13)
    plt.tight_layout()

plt.show()

# ── Summary table ────────────────────────────────────────────────────────────

print(f"\n{'Room':<10} {'Actuator':<16} {'MAE':>8} {'RMSE':>8}")
print("-" * 44)
for room, actuators in sorted(rooms.items()):
    for atype, (cits_id, info) in sorted(actuators.items()):
        act_sensor = info["actuator_list"][0][1] if info["actuator_list"] else None
        if act_sensor is None:
            continue
        pred = act_sensor.input["measuredValue"].history(i_c=0).detach().numpy().T
        actual = act_sensor.time_series_input.values[:, :, 0].detach().numpy().T
        mae = np.mean(np.abs(pred - actual))
        rmse = np.sqrt(np.mean((pred - actual) ** 2))
        label = "Reheat" if atype == "reheat" else "Damper"
        print(f"{room:<10} {label:<16} {mae:>8.4f} {rmse:>8.4f}")
