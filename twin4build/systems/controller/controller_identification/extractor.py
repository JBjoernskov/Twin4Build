"""Extract learned controllers from a trained CITS model and wire them into
a Stage-2 full-physics :class:`~twin4build.model.simulation_model.SimulationModel`.

The :class:`ControllerIdentificationSystem` (CITS) is a *soft*
selector: during training it blends several candidate controllers via
``alpha``/``beta``/``gamma`` weights and an optional
:class:`~twin4build.systems.utils.sigmoid_gate.SigmoidGate`.  After
training (``trf`` etc.) these weights are near one-hot and the
identified structure is fully determined.

This module harvests that identified structure into clean, standalone
:class:`~twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system.PIDControllerSystem`
/ :class:`CascadeControllerSystem`
instances plus a cloned gate, and wires them into a Stage-2 model whose
components live in a different :class:`~twin4build.translator.translator.Translator`.
Cross-translator wiring is done by matching BRICK-node URIs via the
``_sim2sem_map`` / ``_sem2sim_map`` exposed by both translators.

Typical usage (see also
:mod:`twin4build.examples.translator_example_mortar_bldg1`):

.. code-block:: python

    extracted = extract_all_controllers(
        sim_model_stage1, translator_stage1, threshold=0.5
    )
    report = wire_extracted_controllers(
        extracted, sim_model_stage2, translator_stage2,
        rewire_actuator_consumers=True,
    )
    print(report.summary())
    sim_model_stage2.validate()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

import twin4build.core as core
from twin4build.systems.controller.controller_identification.controller_identification_system import (
    ControllerIdentificationSystem,
)
from twin4build.translator.translator import Translator as _Translator
from twin4build.systems.controller.setpoint_controller.cascade_controller.cascade_controller_system import (
    CascadeControllerSystem,
)
from twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system import (
    PIDControllerSystem,
)
from twin4build.systems.utils.sigmoid_gate import BandGate, SigmoidGate


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _to_float(x: Any) -> float:
    """Best-effort scalar extraction from a ``tps.Parameter`` / ``Tensor`` / number."""
    if hasattr(x, "get"):
        x = x.get()
    if isinstance(x, torch.Tensor):
        x = x.detach().flatten()
        return float(x[0].item()) if x.numel() > 0 else 0.0
    return float(x)


def _uri_str(node: Any) -> str:
    """Return a stable URI string for a semantic node (or pass through strings)."""
    if hasattr(node, "uri"):
        return str(node.uri)
    return str(node)


def _topology_at_actuator(
    cits: ControllerIdentificationSystem, actuator: int
) -> Tuple[
    List[Tuple[int, core.System]],
    List[Tuple[int, core.System]],
    List[Tuple[int, core.System]],
]:
    """Return ``(sensor_list, setpoint_list, on_off_signal_list)`` for ``cits``.

    Each list contains ``(index, sender_component)`` pairs, where
    ``index`` is the per-port slot assigned by the translator (matches
    the indexing used by ``beta`` / ``gamma`` / ``beta_b`` for the
    first two, and by ``gamma_gate`` for the third).  The listing is
    independent of ``actuator`` because CITS shares one ``sensorValue``
    / ``setpointValue`` / ``onOffSignal`` input bus across all
    actuators.

    ``onOffSignal`` is the gate-input bus and is structurally distinct
    from ``setpointValue``: the rewire pipeline never prunes
    ``onOffSignal``, so the gate's input space is preserved across
    rewire even when the rewire winner picks a different setpoint for
    the PI error term.
    """
    sensors: List[Tuple[int, core.System]] = []
    setpoints: List[Tuple[int, core.System]] = []
    on_off_signals: List[Tuple[int, core.System]] = []
    for cp in cits.connects_at:
        for conn in cp.connects_system_through:
            src = conn.connects_system
            if cp.input_port == "sensorValue":
                sensors.append((cp.input_port_index.get(conn), src))
            elif cp.input_port == "setpointValue":
                setpoints.append((cp.input_port_index.get(conn), src))
            elif cp.input_port == "onOffSignal":
                on_off_signals.append((cp.input_port_index.get(conn), src))
    sensors.sort(key=lambda t: (t[0] is None, t[0]))
    setpoints.sort(key=lambda t: (t[0] is None, t[0]))
    on_off_signals.sort(key=lambda t: (t[0] is None, t[0]))
    return sensors, setpoints, on_off_signals


def _actuator_brick_nodes(
    cits: ControllerIdentificationSystem,
    translator: Any,
    actuator: int,
) -> List[Any]:
    """Return the actuator-typed BRICK nodes for ``actuator`` of ``cits``.

    The CITS's modeled identity is a multi-member ``ModeledNode``
    group: ``ModeledNode([vav, sensors, setpoints, actuators])``.
    ``_sim2sem_map[cits]`` therefore contains the *union* of matched SM
    nodes for *every* member of the group -- typically ~8 URIs (one VAV
    + several sensors + setpoints + the command).  Sorting that union
    and indexing by ``actuator`` is wrong: lex-order routinely puts the
    VAV URI first (``"#VAVRM107A"`` < ``"#bldg1.X.Reheat_Valve_Command"``
    in ASCII), so the rewire ends up looking up the VAV component in
    Stage 2, finds an FCU/AHU physics object with no ``measuredValue``
    output, snapshots an empty consumer list, and silently leaves the
    extracted controller dangling (no inputSignal -> valvePosition wire).

    To pick the actuator URI specifically, walk the per-pattern group
    bindings stored in ``translator._sim2group_map[cits][sp]``: each
    SP-side ``Node`` has a ``cls`` tuple, and the actuator member has
    ``cls = (BRICK.Command,)`` or ``cls = (BRICK.Damper_Position_Setpoint,)``
    (per the patterns in
    :mod:`twin4build.systems.controller.controller_identification.controller_identification_pi_system`).
    The matched SM node for that SP-side member is the actuator
    BRICK URI we need for cross-stage lookup -- which is also a key in
    Stage-2's ``_sem2sim_map`` (the historized-command SensorSystem in
    Stage 2 is matched by ``brick_sensor_leaf_pattern`` whose
    modeled_node = ``Point``, and Command is a subclass of Point).
    """
    # Compare via URI strings: SP-side ``Node.cls`` elements are
    # ``twin4build.core.SemanticType`` instances whose ``__hash__`` differs
    # from the rdflib ``URIRef`` returned by ``core.namespace.BRICK.X``,
    # even though their ``__eq__`` agrees on the URI string.  ``c in set``
    # uses hashing so a URIRef-keyed set would never match -- we'd silently
    # fall through to the legacy "return everything" branch and the wire
    # step would pick the wrong URI all over again.
    actuator_class_uris = {
        str(core.namespace.BRICK.Command),
        str(core.namespace.BRICK.Damper_Position_Setpoint),
    }

    sps = getattr(translator, "_sim2group_map", {}).get(cits, {})
    actuator_nodes: List[Any] = []
    seen: set = set()
    for sp, groups in sps.items():
        for group in groups:
            for sp_node, sm_binding in group.items():
                cls_tuple = getattr(sp_node, "cls", ())
                if not any(str(c) in actuator_class_uris for c in cls_tuple):
                    continue
                # ``sm_binding`` may be a tuple under SetStepRule; the
                # translator's helper flattens both shapes.
                for sm in _Translator._iter_binding(sm_binding):
                    key = _uri_str(sm)
                    if key in seen:
                        continue
                    seen.add(key)
                    actuator_nodes.append(sm)

    if actuator_nodes:
        return actuator_nodes

    # Fallback for unusual patterns or older translator state: return the
    # whole modeled set so callers retain the legacy behaviour.  This path
    # is broken for multi-member ``ModeledNode`` groups (see docstring) and
    # exists only as a last resort when no actuator-typed binding exists.
    return list(translator._sim2sem_map.get(cits, set()))


def _sensor_uri_for_component(translator: Any, component: core.System) -> Optional[str]:
    """Return the most specific BRICK URI for a Stage-1 sensor component.

    A ``SensorSystem`` may match multiple semantic patterns and end up
    bound to more than one BRICK node; we pick the first deterministically
    ordered URI.  Returns ``None`` if no mapping exists.
    """
    nodes = translator._sim2sem_map.get(component, set())
    uris = sorted(_uri_str(n) for n in nodes)
    return uris[0] if uris else None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExtractedController:
    """Result of harvesting one actuator worth of learned control logic.

    Attributes:
        controller: Fresh, non-CITS controller instance (PID, Constant-SP
            PID, or Cascade) populated with the learned scalar
            parameters.  Not yet wired into any simulation model.
        gate: Cloned :class:`SigmoidGate` / :class:`BandGate` -- or
            ``None`` if ``alpha_gate`` was below ``threshold`` (i.e. the
            controller should run unconditionally, without blending).
        sensor_brick_uris: Selected feedback sensor BRICK URIs (one per
            selected slot, usually a single entry).  Used to find the
            matching Stage-2 sensor components.
        sensor_b_brick_uris: Inner-loop (cascade) feedback sensor URIs;
            ``None`` if the selected candidate is not a cascade.
        setpoint_brick_uris: Selected setpoint BRICK URIs driving the
            *controller* error term.
        gate_setpoint_brick_uris: Selected setpoint BRICK URIs driving
            the *gate* input (from ``gamma_gate``).  Empty when no gate.
        actuator_brick_uri: BRICK URI of the command / modeled_node this
            extracted controller should drive.  Used to locate the
            matching Stage-2 command-sensor and rewire its consumers.
        source_cits_id: ID of the originating CITS (for logs / unique id
            generation when adding the extracted controller).
        actuator_index: Actuator slot within the source CITS.
        candidate_index: Selected candidate index within the source CITS.
        candidate_type: One of ``CITS.CTRL_SETPOINT``,
            ``CITS.CTRL_CASCADE``.
        warnings: Non-fatal issues encountered during extraction
            (e.g. weights not above ``threshold``).
    """

    controller: core.System
    gate: Optional[SigmoidGate]
    sensor_brick_uris: List[str]
    sensor_b_brick_uris: Optional[List[str]]
    setpoint_brick_uris: List[str]
    gate_setpoint_brick_uris: List[str]
    actuator_brick_uri: Optional[str]
    source_cits_id: str
    actuator_index: int
    candidate_index: int
    candidate_type: str
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"ExtractedController({self.source_cits_id} / actuator={self.actuator_index})",
            f"  candidate: index={self.candidate_index} type={self.candidate_type}"
            f" class={self.controller.__class__.__name__}",
            f"  actuator_uri: {self.actuator_brick_uri}",
            f"  feedback sensors: {self.sensor_brick_uris}",
        ]
        if self.sensor_b_brick_uris:
            lines.append(f"  cascade inner-loop sensors: {self.sensor_b_brick_uris}")
        lines.append(f"  controller setpoints: {self.setpoint_brick_uris}")
        if self.gate is not None:
            lines.append(
                f"  gate: {self.gate.__class__.__name__} (id={self.gate.id})"
                f" driven by {self.gate_setpoint_brick_uris}"
            )
        else:
            lines.append("  gate: (none -- alpha_gate below threshold)")
        if self.warnings:
            lines.append("  warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


@dataclass
class WiringReport:
    """Outcome of :func:`wire_extracted_controllers`.

    Attributes:
        wired: ``[(cits_id, controller_id, actuator_uri), ...]`` for
            every controller successfully inserted and connected.
        rewired_consumers: ``[(controller_id, consumer_id, port), ...]``
            for every Stage-2 consumer that was detached from a historical
            command-sensor and re-attached to the new controller output.
        unmatched_sensor_uris: BRICK sensor/setpoint URIs from Stage 1
            that had no counterpart in Stage 2.  These are skipped but
            the user is warned.
        unmatched_actuator_uris: BRICK command URIs from Stage 1 whose
            Stage-2 component could not be located.  Those controllers
            are inserted but not rewired.
        skipped_existing: IDs of controllers / gates already present in
            the target model (idempotent re-runs are a no-op).
        warnings: Free-form warnings raised during wiring.
    """

    wired: List[Tuple[str, str, Optional[str]]] = field(default_factory=list)
    rewired_consumers: List[Tuple[str, str, str]] = field(default_factory=list)
    unmatched_sensor_uris: List[str] = field(default_factory=list)
    unmatched_actuator_uris: List[str] = field(default_factory=list)
    skipped_existing: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["WiringReport", "=" * 40]
        lines.append(f"controllers wired: {len(self.wired)}")
        for cits_id, ctrl_id, act_uri in self.wired:
            lines.append(f"  {cits_id:40s} -> {ctrl_id} (actuator: {act_uri})")
        lines.append(f"\nconsumers rewired: {len(self.rewired_consumers)}")
        for ctrl_id, cons_id, port in self.rewired_consumers:
            lines.append(f"  {ctrl_id} -> {cons_id}.{port}")
        if self.skipped_existing:
            lines.append(
                f"\nskipped (already in target model): {len(self.skipped_existing)}"
            )
            for cid in self.skipped_existing:
                lines.append(f"  {cid}")
        if self.unmatched_sensor_uris:
            lines.append(
                f"\nsensor/setpoint URIs with no Stage-2 match: "
                f"{len(self.unmatched_sensor_uris)}"
            )
            for uri in self.unmatched_sensor_uris:
                lines.append(f"  {uri}")
        if self.unmatched_actuator_uris:
            lines.append(
                f"\nactuator URIs with no Stage-2 match: "
                f"{len(self.unmatched_actuator_uris)}"
            )
            for uri in self.unmatched_actuator_uris:
                lines.append(f"  {uri}")
        if self.warnings:
            lines.append(f"\nwarnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Candidate cloning
# ---------------------------------------------------------------------------

def _clone_pid(pid: PIDControllerSystem, new_id: str) -> PIDControllerSystem:
    """Construct a fresh ``PIDControllerSystem`` copying the learned scalars."""
    return PIDControllerSystem(
        kp=_to_float(pid.kp),
        Ti=_to_float(pid.Ti),
        Td=_to_float(pid.Td),
        output_min=_to_float(pid.output_min),
        output_max=_to_float(pid.output_max),
        is_reverse=bool(pid.isReverse),
        id=new_id,
    )


def _clone_candidate(
    src_ctrl: core.System, candidate_type: str, new_id: str
) -> core.System:
    """Clone a CITS candidate into a standalone controller System.

    Parameters come straight from ``src_ctrl``'s current scalar values;
    ``isReverse`` is carried verbatim since it is a Python bool (not
    a ``tps.Parameter``).
    """
    if candidate_type == ControllerIdentificationSystem.CTRL_SETPOINT:
        return _clone_pid(src_ctrl, new_id)

    if candidate_type == ControllerIdentificationSystem.CTRL_CASCADE:
        a, b = src_ctrl.ctrl_a, src_ctrl.ctrl_b
        # Assume PID-like inner controllers for the cascade clone; this
        # matches the default CITS cascade candidate
        # (PID + PID).  Non-PID cascade members would need bespoke
        # cloning here; we fall back to re-using the same class.
        return CascadeControllerSystem(
            kp_a=_to_float(a.kp), Ti_a=_to_float(a.Ti), Td_a=_to_float(a.Td),
            output_min_a=_to_float(a.output_min), output_max_a=_to_float(a.output_max),
            isReverse_a=bool(a.isReverse),
            kp_b=_to_float(b.kp), Ti_b=_to_float(b.Ti), Td_b=_to_float(b.Td),
            output_min_b=_to_float(b.output_min), output_max_b=_to_float(b.output_max),
            isReverse_b=bool(b.isReverse),
            id=new_id,
        )

    raise ValueError(
        f"Unsupported CITS candidate type '{candidate_type}' "
        f"for controller class '{src_ctrl.__class__.__name__}'"
    )


def _clone_gate(
    src_gate: SigmoidGate, default_output: float, new_id: str
) -> SigmoidGate:
    """Clone a gate (``SigmoidGate`` or ``BandGate``) with learned params.

    ``default_output`` is pulled from the CITS's per-actuator
    ``default_output_{a}`` parameter and stored on the cloned gate's
    new ``default_output`` port so the blend formula can use it.
    """
    if isinstance(src_gate, BandGate):
        return BandGate(
            threshold=_to_float(src_gate.threshold),
            band=_to_float(src_gate.band),
            steepness=_to_float(src_gate.steepness),
            default_output=default_output,
            id=new_id,
        )
    return SigmoidGate(
        threshold=_to_float(src_gate.threshold),
        steepness=_to_float(src_gate.steepness),
        polarity=_to_float(src_gate.polarity),
        default_output=default_output,
        id=new_id,
    )


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

def _argmax_idx(vec: torch.Tensor) -> int:
    return int(torch.argmax(vec.detach()).item())


def _selected_indices(
    weights: torch.Tensor, threshold: float, at_least_one: bool = True
) -> List[int]:
    """Indices whose weight is above ``threshold``.

    If none pass and ``at_least_one`` is True, fall back to the argmax.
    """
    vec = weights.detach()
    idxs = [i for i in range(vec.numel()) if float(vec[i].item()) > threshold]
    if not idxs and at_least_one and vec.numel() > 0:
        idxs = [_argmax_idx(vec)]
    return idxs


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_controller(
    cits: ControllerIdentificationSystem,
    translator_stage1: Any,
    actuator: int = 0,
    threshold: float = 0.5,
) -> ExtractedController:
    """Extract the identified controller for a single actuator of ``cits``.

    Args:
        cits: Trained CITS component from the Stage-1 simulation model.
        translator_stage1: The translator used to build the Stage-1
            model; provides ``_sim2sem_map`` for BRICK-node lookup.
        actuator: Actuator index within the CITS (typical: 0).
        threshold: Weight threshold above which a candidate / sensor /
            setpoint is considered *selected*.  For CITS trained to
            near-one-hot this is rarely binding, but it is used as a
            fallback for detecting whether the gate is active
            (``alpha_gate`` > ``threshold``).

    Returns:
        An :class:`ExtractedController` describing a clean, wireable
        controller + optional gate + the BRICK URIs of the sensors /
        setpoints / command that drove this actuator during Stage-1
        training.
    """
    if not cits._built:
        raise RuntimeError(
            f"CITS '{cits.id}' was never built (no `_build_components` call)."
            " Initialize the model or trigger the build manually before extracting."
        )

    warnings: List[str] = []
    weights = cits.get_selection_weights()

    alpha_vec = weights[f"alpha_{actuator}"]
    c_sel = _argmax_idx(alpha_vec)
    alpha_sel = float(alpha_vec[c_sel].item())
    if alpha_sel < threshold:
        warnings.append(
            f"alpha_{actuator}_{c_sel}={alpha_sel:.3f} < threshold={threshold}:"
            " controller selection not one-hot; using argmax."
        )

    candidate_type = cits._candidate_types[c_sel]
    src_ctrl = cits._get_candidate(actuator, c_sel)
    new_ctrl_id = f"{cits.id}_a{actuator}_ctrl"
    new_controller = _clone_candidate(src_ctrl, candidate_type, new_ctrl_id)

    # --- Feedback sensors (beta) ---
    sensor_topo, setpoint_topo, on_off_signal_topo = _topology_at_actuator(
        cits, actuator
    )
    topo_by_idx = {idx: comp for idx, comp in sensor_topo}

    beta_vec = weights[f"beta_{actuator}"]
    sensor_sel_idxs = _selected_indices(beta_vec, threshold)
    sensor_uris: List[str] = []
    for i in sensor_sel_idxs:
        comp = topo_by_idx.get(i)
        if comp is None:
            warnings.append(f"beta selected sensor slot {i} has no connection")
            continue
        uri = _sensor_uri_for_component(translator_stage1, comp)
        if uri is None:
            warnings.append(
                f"sensor '{comp.id}' has no BRICK node in translator_stage1 sim2sem_map"
            )
            continue
        sensor_uris.append(uri)

    # --- Inner-loop sensors (cascade, beta_b) ---
    sensor_b_uris: Optional[List[str]] = None
    if candidate_type == ControllerIdentificationSystem.CTRL_CASCADE:
        sensor_b_uris = []
        beta_b_vec = weights[f"beta_b_{actuator}"]
        for i in _selected_indices(beta_b_vec, threshold):
            comp = topo_by_idx.get(i)
            if comp is None:
                warnings.append(f"beta_b selected sensor slot {i} has no connection")
                continue
            uri = _sensor_uri_for_component(translator_stage1, comp)
            if uri is None:
                warnings.append(
                    f"cascade inner-loop sensor '{comp.id}' has no BRICK node"
                )
                continue
            sensor_b_uris.append(uri)

    # --- Controller setpoints (gamma) ---
    sp_by_idx = {idx: comp for idx, comp in setpoint_topo}
    gamma_vec = weights[f"gamma_{actuator}"]
    setpoint_uris: List[str] = []
    for j in _selected_indices(gamma_vec, threshold):
        comp = sp_by_idx.get(j)
        if comp is None:
            warnings.append(f"gamma selected setpoint slot {j} has no connection")
            continue
        uri = _sensor_uri_for_component(translator_stage1, comp)
        if uri is None:
            warnings.append(
                f"setpoint '{comp.id}' has no BRICK node in translator_stage1 sim2sem_map"
            )
            continue
        setpoint_uris.append(uri)

    # --- Gate (alpha_gate, gamma_gate, gate clone) ---
    alpha_gate = float(weights[f"alpha_gate_{actuator}"].item())
    default_output = float(weights[f"default_output_{actuator}"].item())

    gate: Optional[SigmoidGate] = None
    gate_setpoint_uris: List[str] = []
    if alpha_gate > threshold:
        src_gate = cits._get_gate(actuator)
        gate = _clone_gate(
            src_gate,
            default_output=default_output,
            new_id=f"{cits.id}_a{actuator}_gate",
        )
        # ``gamma_gate`` indexes the ``onOffSignal`` port -- the
        # gate-input bus, structurally distinct from the PI-error
        # ``setpointValue`` bus that ``gamma`` indexes.  The auto-mirror
        # in the CITS signature patterns means each setpoint typically
        # appears in *both* topologies, but only the on/off one is
        # guaranteed to be intact post-rewire.
        oo_by_idx = {idx: comp for idx, comp in on_off_signal_topo}
        gamma_gate_vec = weights[f"gamma_gate_{actuator}"]
        for j in _selected_indices(gamma_gate_vec, threshold):
            comp = oo_by_idx.get(j)
            if comp is None:
                warnings.append(
                    f"gamma_gate selected onOffSignal slot {j} has no connection"
                )
                continue
            uri = _sensor_uri_for_component(translator_stage1, comp)
            if uri is None:
                warnings.append(
                    f"gate-driving onOffSignal '{comp.id}' has no BRICK node"
                )
                continue
            gate_setpoint_uris.append(uri)

    # --- Actuator BRICK URI (modeled_node of the CITS signature pattern) ---
    # ``_actuator_brick_nodes`` filters the CITS's ``ModeledNode([vav,
    # sensors, setpoints, actuators])`` group down to *only* the actuator-
    # typed members (BRICK.Command / BRICK.Damper_Position_Setpoint), so
    # for the typical n_actuators==1 case the list is a singleton.  For
    # multi-actuator CITS we still sort + index-by-actuator below.
    actuator_nodes = _actuator_brick_nodes(cits, translator_stage1, actuator)
    if len(actuator_nodes) == 0:
        actuator_uri = None
        warnings.append(
            f"CITS '{cits.id}' has no actuator-typed BRICK node in "
            f"translator_stage1._sim2group_map (filter expects "
            f"BRICK.Command or BRICK.Damper_Position_Setpoint)"
        )
    elif len(actuator_nodes) == 1:
        actuator_uri = _uri_str(actuator_nodes[0])
    else:
        # Multi-actuator CITS: sort the actuator-typed URIs for
        # determinism and pick by actuator index.
        uris_sorted = sorted(_uri_str(n) for n in actuator_nodes)
        if 0 <= actuator < len(uris_sorted):
            actuator_uri = uris_sorted[actuator]
        else:
            actuator_uri = uris_sorted[0]
            warnings.append(
                f"actuator index {actuator} out of range for "
                f"{len(uris_sorted)} modeled command nodes; using first"
            )

    return ExtractedController(
        controller=new_controller,
        gate=gate,
        sensor_brick_uris=sensor_uris,
        sensor_b_brick_uris=sensor_b_uris,
        setpoint_brick_uris=setpoint_uris,
        gate_setpoint_brick_uris=gate_setpoint_uris,
        actuator_brick_uri=actuator_uri,
        source_cits_id=cits.id,
        actuator_index=actuator,
        candidate_index=c_sel,
        candidate_type=candidate_type,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def _build_uri_to_components(translator: Any) -> Dict[str, List[core.System]]:
    """Flatten ``translator._sem2sim_map`` into ``{uri_str: [components]}``.

    This keys the Stage-2 lookup by URI string so it's independent of
    whether Stage-1 and Stage-2 share the same ``SemanticModel``
    instance.  Components are deduplicated per URI, in deterministic id
    order, to make wiring reproducible.
    """
    out: Dict[str, List[core.System]] = {}
    for sem_node, comps in translator._sem2sim_map.items():
        uri = _uri_str(sem_node)
        lst = out.setdefault(uri, [])
        for c in comps:
            if c not in lst:
                lst.append(c)
        lst.sort(key=lambda c: c.id)
    return out


def _port_for_feedback(controller: core.System) -> str:
    """Return the controller input port that should receive the feedback signal."""
    if isinstance(controller, CascadeControllerSystem):
        return "actualValue_a"
    return "actualValue"


def _port_for_setpoint(controller: core.System) -> Optional[str]:
    """Return the controller input port that should receive the external setpoint."""
    if isinstance(controller, CascadeControllerSystem):
        return "setpointValue_a"
    return "setpointValue"


def _port_for_inner_feedback(controller: core.System) -> Optional[str]:
    """Return the inner-loop feedback port (cascade only), else ``None``."""
    if isinstance(controller, CascadeControllerSystem):
        return "actualValue_b"
    return None


def _output_port(controller: core.System) -> str:
    """Return the controller's output port name that carries the control signal."""
    return "inputSignal"


def _pick_best_component(components: List[core.System]) -> Optional[core.System]:
    """Pick a single wiring target from a list of Stage-2 candidates.

    We prefer a :class:`SensorSystem`-shaped component (has a
    ``measuredValue`` output) since those are the signal sources the
    extracted controller wants to read from.  Falls back to the first
    component if no preference matches.
    """
    # Late import to avoid a circular dependency at module load time.
    from twin4build.systems.sensor.sensor_system import SensorSystem

    if not components:
        return None
    sensors = [c for c in components if isinstance(c, SensorSystem)]
    if sensors:
        return sensors[0]
    return components[0]


def _existing_outgoing(
    sensor_comp: core.System, output_port: str = "measuredValue"
) -> List[Tuple[core.System, str, Any, Any]]:
    """Snapshot ``(consumer, input_port, input_port_idx, output_port_idx)``.

    Returned before any rewire so we can iterate without mutating the
    live connection list during traversal.  ``input_port_idx`` /
    ``output_port_idx`` may be ``None`` (scalar ports) or
    ``int`` / tensor (vector ports).
    """
    snapshot: List[Tuple[core.System, str, Any, Any]] = []
    for conn in list(sensor_comp.connected_through):
        if conn.output_port != output_port:
            continue
        for cp in list(conn.connects_system_at):
            consumer = cp.connection_point_of
            if consumer is None:
                continue
            ip_idx = cp.input_port_index.get(conn)
            op_idx = cp.output_port_index.get(conn)
            snapshot.append((consumer, cp.input_port, ip_idx, op_idx))
    return snapshot


def _connect_from_uri(
    sim_model: Any,
    uri_to_comps: Dict[str, List[core.System]],
    uri: str,
    receiver: core.System,
    receiver_port: str,
    unmatched: List[str],
    warnings: List[str],
    sender_port: str = "measuredValue",
) -> bool:
    """Look up a Stage-2 component by BRICK URI and wire it to ``receiver``."""
    candidates = uri_to_comps.get(uri, [])
    sender = _pick_best_component(candidates)
    if sender is None:
        unmatched.append(uri)
        return False
    if sender_port not in sender.output:
        warnings.append(
            f"Stage-2 component '{sender.id}' (URI={uri}) has no output port"
            f" '{sender_port}'; skipping wire to '{receiver.id}.{receiver_port}'."
        )
        return False
    sim_model.add_connection(sender, receiver, sender_port, receiver_port)
    return True


def wire_extracted_controllers(
    extracted: List[ExtractedController],
    sim_model_stage2: Any,
    translator_stage2: Any,
    rewire_actuator_consumers: bool = True,
    verbose: bool = True,
) -> WiringReport:
    """Insert extracted controllers into a Stage-2 :class:`SimulationModel`.

    For each :class:`ExtractedController` in ``extracted`` this:

    1. Looks up every Stage-2 component whose BRICK URI matches the
       extractor's recorded sensor / setpoint / actuator URI via a
       reverse map built from ``translator_stage2._sem2sim_map``.
    2. Wires the new controller's feedback / setpoint / (cascade)
       inner-loop inputs accordingly.
    3. If a gate was extracted, wires its ``inputSignal`` from the
       setpoint selected by ``gamma_gate`` and wires the controller's
       ``inputSignal`` output into the gate's ``controllerSignal``
       input.  The gate's ``outputSignal`` becomes the effective
       control output.
    4. When ``rewire_actuator_consumers`` is ``True``, locates the
       Stage-2 component that was built from the same command BRICK
       node as the CITS's ``modeled_node`` and, for every existing
       downstream consumer reading that command's
       ``measuredValue`` port, removes the old connection and replaces
       it with one from the new controller/gate output.  Set to
       ``False`` to only log what would happen.

    The function is idempotent: if a controller/gate id already exists
    in ``sim_model_stage2`` it is recorded in ``report.skipped_existing``
    and left alone.

    Args:
        extracted: Result of :func:`extract_all_controllers`.
        sim_model_stage2: Target full-physics Stage-2 model.
        translator_stage2: Translator used to build Stage-2.
        rewire_actuator_consumers: See above.
        verbose: Print progress lines while wiring.

    Returns:
        A :class:`WiringReport` describing every connection created,
        every consumer rewired, and every unmatched URI.
    """
    report = WiringReport()
    uri_to_comps = _build_uri_to_components(translator_stage2)
    existing_ids = set(sim_model_stage2.components.keys())

    for E in extracted:
        if verbose:
            print(
                f"[wire] {E.source_cits_id} (actuator={E.actuator_index},"
                f" candidate={E.candidate_type})"
            )

        if E.controller.id in existing_ids:
            report.skipped_existing.append(E.controller.id)
            if verbose:
                print(f"  skipped (already present): {E.controller.id}")
            continue

        # --- 1. Wire feedback sensors into controller ---------------------
        fb_port = _port_for_feedback(E.controller)
        for uri in E.sensor_brick_uris:
            ok = _connect_from_uri(
                sim_model_stage2, uri_to_comps, uri,
                E.controller, fb_port,
                report.unmatched_sensor_uris, report.warnings,
            )
            if verbose and ok:
                print(f"  sensor {uri} -> {E.controller.id}.{fb_port}")

        # --- 2. Wire inner-loop feedback (cascade) ------------------------
        inner_port = _port_for_inner_feedback(E.controller)
        if inner_port is not None and E.sensor_b_brick_uris:
            for uri in E.sensor_b_brick_uris:
                ok = _connect_from_uri(
                    sim_model_stage2, uri_to_comps, uri,
                    E.controller, inner_port,
                    report.unmatched_sensor_uris, report.warnings,
                )
                if verbose and ok:
                    print(f"  inner sensor {uri} -> {E.controller.id}.{inner_port}")

        # --- 3. Wire external setpoint (not used by ConstantSP) -----------
        sp_port = _port_for_setpoint(E.controller)
        if sp_port is not None:
            for uri in E.setpoint_brick_uris:
                ok = _connect_from_uri(
                    sim_model_stage2, uri_to_comps, uri,
                    E.controller, sp_port,
                    report.unmatched_sensor_uris, report.warnings,
                )
                if verbose and ok:
                    print(f"  setpoint {uri} -> {E.controller.id}.{sp_port}")

        # --- 4. Wire the gate (if any) ------------------------------------
        output_source: core.System = E.controller
        output_src_port: str = _output_port(E.controller)
        if E.gate is not None:
            for uri in E.gate_setpoint_brick_uris:
                ok = _connect_from_uri(
                    sim_model_stage2, uri_to_comps, uri,
                    E.gate, "inputSignal",
                    report.unmatched_sensor_uris, report.warnings,
                )
                if verbose and ok:
                    print(f"  gate-input {uri} -> {E.gate.id}.inputSignal")
            # Blend the controller into the gate via its new
            # controllerSignal port.  The gate's do_step detects this
            # wire and emits gate*ctrl + (1-gate)*default_output.
            sim_model_stage2.add_connection(
                E.controller, E.gate,
                _output_port(E.controller), "controllerSignal",
            )
            if verbose:
                print(
                    f"  controller {E.controller.id} -> {E.gate.id}.controllerSignal"
                )
            output_source = E.gate
            output_src_port = "outputSignal"
        else:
            # No gate: controller must still be registered as a component
            # even if no downstream consumer gets rewired (e.g. actuator
            # URI missing).  add_component is idempotent.
            sim_model_stage2.add_component(E.controller)

        # --- 5. Rewire downstream consumers of the command-sensor ---------
        rewired_any = False
        if E.actuator_brick_uri is None:
            report.unmatched_actuator_uris.append(f"(none for {E.source_cits_id})")
        else:
            command_candidates = uri_to_comps.get(E.actuator_brick_uri, [])
            command_sensor = _pick_best_component(command_candidates)
            if command_sensor is None:
                report.unmatched_actuator_uris.append(E.actuator_brick_uri)
                if verbose:
                    print(
                        f"  [warn] no Stage-2 component for actuator URI"
                        f" {E.actuator_brick_uri}"
                    )
            elif rewire_actuator_consumers:
                snapshot = _existing_outgoing(command_sensor, "measuredValue")
                for consumer, in_port, in_idx, _out_idx in snapshot:
                    try:
                        sim_model_stage2.remove_connection(
                            command_sensor, consumer, "measuredValue", in_port
                        )
                    except ValueError as exc:
                        report.warnings.append(
                            f"failed to remove {command_sensor.id}.measuredValue"
                            f" -> {consumer.id}.{in_port}: {exc}"
                        )
                        continue
                    sim_model_stage2.add_connection(
                        output_source, consumer,
                        output_src_port, in_port,
                        input_port_index=in_idx,
                    )
                    report.rewired_consumers.append(
                        (output_source.id, consumer.id, in_port)
                    )
                    rewired_any = True
                    if verbose:
                        print(
                            f"  rewire {command_sensor.id} -> {consumer.id}.{in_port}"
                            f" now from {output_source.id}.{output_src_port}"
                        )
            else:
                if verbose:
                    snapshot = _existing_outgoing(command_sensor, "measuredValue")
                    for consumer, in_port, _i, _o in snapshot:
                        print(
                            f"  [dry] would rewire {command_sensor.id} ->"
                            f" {consumer.id}.{in_port}"
                        )

        report.wired.append(
            (E.source_cits_id, E.controller.id, E.actuator_brick_uri)
        )
        if not rewired_any and rewire_actuator_consumers:
            report.warnings.append(
                f"controller '{E.controller.id}' inserted but no downstream"
                f" consumers were rewired (actuator URI: {E.actuator_brick_uri})"
            )

    return report


def extract_all_controllers(
    sim_model_stage1: Any,
    translator_stage1: Any,
    threshold: float = 0.5,
) -> List[ExtractedController]:
    """Extract every actuator of every CITS in ``sim_model_stage1``.

    Args:
        sim_model_stage1: Stage-1 :class:`SimulationModel` holding one
            or more trained CITS components.
        translator_stage1: The translator that produced
            ``sim_model_stage1``.
        threshold: Forwarded to :func:`extract_controller`.

    Returns:
        A list of :class:`ExtractedController`, one per actuator slot
        of each CITS, sorted by ``(cits_id, actuator_index)`` for
        deterministic downstream wiring.
    """
    extracted: List[ExtractedController] = []
    cits_list = [
        c for c in sim_model_stage1.components.values()
        if isinstance(c, ControllerIdentificationSystem)
    ]
    # Deterministic order -> deterministic new component ids.
    cits_list.sort(key=lambda c: c.id)
    for cits in cits_list:
        n_act = cits.n_actuators if cits._built else 1
        for a in range(n_act):
            extracted.append(
                extract_controller(
                    cits, translator_stage1, actuator=a, threshold=threshold
                )
            )
    return extracted
