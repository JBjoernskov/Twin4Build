"""Public functional simulation and direct CUDA Graph execution."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

import twin4build.utils.types as tps
from twin4build.utils._cuda_graph import CudaGraphCallable
from twin4build.simulator._functional import (
    FunctionalModel,
    StateLayout,
    collect_stateful,
    functional_rollout_tape,
    recording_data_signature,
)
from twin4build.utils.rgetattr import rgetattr

FUNCTIONAL_MATERIALIZATION_REVISION = "vectorized-history-v1"


def _execution_order(model):
    order = getattr(model, "_flat_execution_order", None)
    if order is None:
        order = model.flat_execution_order
    return list(order)


def _tensor_signature(value):
    value = value.detach()
    return (
        tuple(value.shape),
        str(value.dtype),
        str(value.device),
        tuple(value.reshape(-1).cpu().tolist()),
    )


def session_signature(model, step_size, max_t, n_periods, date_time_steps):
    """Semantic cache key; parameter values intentionally invalidate sessions."""
    order = _execution_order(model)
    topology = []
    parameters = []
    for component in order:
        topology.append(
            (
                id(component),
                component.id,
                tuple(
                    (name, type(port).__name__, port.n_c, getattr(port, "n_v", None))
                    for name, port in component.input.items()
                ),
                tuple(
                    (name, type(port).__name__, port.n_c, getattr(port, "n_v", None))
                    for name, port in component.output.items()
                ),
                tuple(
                    (
                        point.input_port,
                        tuple(
                            (id(connection.connects_system), connection.output_port)
                            for connection in point.connects_system_through
                        ),
                    )
                    for point in component.connects_at
                ),
            )
        )
        for name in getattr(component, "PARAM_NAMES", ()):
            value = rgetattr(component, name).get()
            parameters.append((component.id, name, _tensor_signature(value)))
    sim_model = getattr(model, "_simulation_model", None) or model
    return (
        tuple(topology),
        tuple(parameters),
        str(getattr(sim_model, "device", torch.device("cpu"))),
        str(tps.float_dtype()),
        tuple(int(value) for value in step_size),
        int(max_t),
        int(n_periods),
        tuple(
            tuple(repr(value) for value in period)
            for period in date_time_steps
        ),
        recording_data_signature(model),
    )


@dataclass
class RolloutResult:
    states: torch.Tensor
    outputs: torch.Tensor


def _timed_phase(device, function):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return value, time.perf_counter() - started


class FunctionalSimulationSession:
    """Reusable topology, recording plan, and optional captured graph."""

    def __init__(self, simulator, step_size, max_t, n_periods, signature):
        self.signature = signature
        self.simulator = simulator
        self.model = simulator.model
        self.step_size = list(step_size)
        self.max_t = int(max_t)
        self.n_periods = int(n_periods)
        self.outputs = [
            (component, name)
            for component in self.model.components.values()
            for name in component.output
        ]
        stateful = collect_stateful(self.model)
        if not stateful:
            raise RuntimeError(
                "functional simulation requires at least one stateful component"
            )
        self.layout = StateLayout(stateful)
        unique_steps = {int(value) for value in step_size}
        if len(unique_steps) != 1:
            raise RuntimeError(
                "functional simulation does not support mixed period step sizes"
            )
        self.functional_model = FunctionalModel(
            self.model,
            self.layout.components,
            [],
            unique_steps.pop(),
            outputs=self.outputs,
        )
        # CUDA Graphs capture the storage addresses of every tensor read by
        # the rollout, including non-estimated component parameters.  Model
        # initialization replaces the public parameter tensors between
        # simulate() calls, so keep session-owned storage whose addresses stay
        # valid for graph replay.
        self.functional_model._default_params = {
            component_id: {
                name: value.detach().clone() for name, value in parameters.items()
            }
            for component_id, parameters in (
                self.functional_model._default_params.items()
            )
        }
        if self.functional_model.D != self.layout.width:
            raise RuntimeError("functional state layout width mismatch")
        self.theta = torch.zeros(
            0,
            dtype=tps.float_dtype(),
            device=self.layout.gather(0).device,
        )
        self.graph = None
        self.setup_seconds = 0.0
        self.capture_seconds = 0.0
        self.replay_seconds = 0.0
        self.capture_count = 0
        self.replay_count = 0
        self.cached_recording = None
        self.cached_public_histories = ()
        self.recording_cache_hit = False
        self.recording_cache_hits = 0
        self.recording_cache_misses = 0

    def close(self):
        """Release captured CUDA resources owned by this session."""
        graph, self.graph = self.graph, None
        if graph is not None:
            graph.close()
        self.cached_recording = None
        self.cached_public_histories = ()

    reset = close

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def refresh_parameters(self):
        """Refresh non-estimated parameter references after model.initialize."""
        for component in self.functional_model.cone:
            stable = self.functional_model._default_params[component.id]
            for name in getattr(component, "PARAM_NAMES", ()):
                stable[name].copy_(rgetattr(component, name).get())
        self.functional_model._theta_param_cache = None

    def _exogenous_sources(self, component, key):
        if len(key) >= 3 and isinstance(component.input[key[1]], tps.Vector):
            return dict(
                self.functional_model._vector_slot_sources(component, key[1])
            ).get(key[2], [])
        return self.functional_model._trace_sources(component, key[1])

    @staticmethod
    def _history_value(port, step, period):
        if port._history is None or (
            not port._history_is_populated and not port.is_leaf
        ):
            return None
        return port._history[step, period]

    def _unconnected_value(self, component, key, step, period):
        port_name = key[1]
        port = component.input[port_name]
        value = self._history_value(port, step, period)
        if value is not None:
            return value
        # Occupancy's measured data are initialized into private time-series
        # publishers. Reading them is the minimal producer closure; its
        # state-dependent forward/do_step is never called.
        if port_name in (
            "indoorCo2Measured",
            "previousIndoorCo2Measured",
            "damperPositionMeasured",
        ):
            if not hasattr(component, "_co2_ts"):
                raise RuntimeError(
                    f"cannot isolate exogenous input {component.id}.{port_name}"
                )
            if port_name == "damperPositionMeasured":
                values = component._damper_ts.values
            elif port_name == "previousIndoorCo2Measured":
                values = component._co2_ts.values
                step = max(step - 1, 0)
            else:
                values = component._co2_ts.values
            return values[step, period]
        # A genuinely unconnected, non-leaf port has no producer capable of
        # changing it during object-graph execution. Its initialized value is
        # therefore a static exogenous constant. Components with private
        # time-varying publishers must be handled explicitly above.  The
        # check is per KEY: a vector port with some slots wired and others
        # not (an AHU branch whose damper has no controller) is asked about
        # the unconnected SLOT, not about the port as a whole -- the caller
        # slices the slot out of the full vector value.
        if not self._exogenous_sources(component, key):
            return port.get()[period]
        raise RuntimeError(
            f"cannot safely isolate exogenous input {component.id}.{port_name}; "
            "add a leaf history or a dedicated exogenous publisher"
        )

    def _step_external(self, component, step, done, visiting):
        """Run only the exogenous producer closure needed for exogenous tape."""
        if component.id in done:
            return
        if component.id in self.functional_model.cone_ids:
            raise RuntimeError(
                "exogenous producer closure reaches functional component "
                f"{component.id}; refusing hidden object-graph dynamics"
            )
        if component.id in visiting:
            raise RuntimeError(
                f"exogenous producer closure contains a cycle at {component.id}"
            )
        visiting.add(component.id)
        for point in component.connects_at:
            for connection in point.connects_system_through:
                producer = connection.connects_system
                output = producer.output[connection.output_port]
                value = self._history_value(output, step, 0)
                if value is not None:
                    output._tensor.copy_(output._history[step])
                else:
                    self._step_external(producer, step, done, visiting)
        self.simulator._assign_component_inputs(component, step)
        component.do_step(
            self.simulator.second_time_steps[:, step],
            self.simulator.date_time_steps[:, step],
            self.step_size,
            step,
        )
        visiting.remove(component.id)
        done.add(component.id)

    def record_exogenous_inputs(self):
        """Build exogenous tape and initial feedback without an object-graph timestep."""
        if self.cached_recording is not None:
            self.recording_cache_hit = True
            self.recording_cache_hits += 1
            for port, history in self.cached_public_histories:
                port._history.copy_(history)
                port._history_is_populated = True
                port._tensor.copy_(history[-1])
            y0, exogenous_tape = self.cached_recording
            return y0.detach().clone(), exogenous_tape.detach().clone()
        self.recording_cache_hit = False
        self.recording_cache_misses += 1
        functional_model = self.functional_model
        components = dict(self.model.components)
        sim_model = getattr(self.model, "_simulation_model", None) or self.model
        components.update(getattr(sim_model, "_fused_components", None) or {})
        device = self.theta.device
        exogenous_tape = torch.zeros(
            self.max_t,
            self.n_periods,
            functional_model._n_exogenous,
            dtype=tps.float_dtype(),
            device=device,
        )
        external_done = [set() for _ in range(self.max_t)]
        for key in functional_model._exogenous_keys:
            component = components[key[0]]
            sources = self._exogenous_sources(component, key)
            target = functional_model._exogenous_index[key]
            for step in range(self.max_t):
                for period in range(self.n_periods):
                    if sources:
                        pieces = []
                        for producer, output_name, routes in sources:
                            value = self._history_value(
                                producer.output[output_name], step, period
                            )
                            if (
                                value is None
                                and not producer.connects_at
                                and producer.output[output_name]._history is not None
                            ):
                                # Schedules/weather are initialized data
                                # publishers even in legacy translated models
                                # where their output lost the ``is_leaf`` flag.
                                value = producer.output[output_name]._history[
                                    step, period
                                ]
                            if value is None:
                                self._step_external(
                                    producer,
                                    step,
                                    external_done[step],
                                    set(),
                                )
                                value = producer.output[output_name].get()[period]
                            pieces.append(functional_model._apply_routes(value, routes))
                        value = pieces[0]
                        for piece in pieces[1:]:
                            value = value + piece
                    else:
                        value = self._unconnected_value(component, key, step, period)
                        if len(key) >= 3 and isinstance(
                            component.input[key[1]], tps.Vector
                        ):
                            value = value[..., key[2]]
                    exogenous_tape[step, period, target] = value.reshape(-1).to(device)
                    # Preserve public histories for unconnected exogenous tape ports.
                    port = component.input[key[1]]
                    if not sources and port.log_history:
                        if isinstance(port, tps.Vector) and len(key) >= 3:
                            port._history[step, period, :, key[2]] = value
                        else:
                            port._history[step, period] = value
                        if step == self.max_t - 1 and period == self.n_periods - 1:
                            port._history_is_populated = True
                            port._tensor.copy_(port._history[-1])
        feedback = torch.zeros(
            self.n_periods,
            functional_model.n_feedback,
            dtype=tps.float_dtype(),
            device=device,
        )
        for key, mask, sources in zip(
            functional_model._feedback_keys,
            functional_model._feedback_masks,
            functional_model._fb_producer,
        ):
            component = components[key[0]]
            port = component.input[key[1]]
            slc = functional_model._fb_index[key]
            for period in range(self.n_periods):
                merged = None
                for producer, output_name, routes in sources:
                    current = producer.output[output_name].get()[period]
                    routed = functional_model._apply_routes(current, routes)
                    merged = routed if merged is None else merged + routed
                feedback[period, slc] = merged.reshape(-1) * mask.to(device)
            # A pass-through/data sensor may be the immediate object-graph
            # producer even though composition follows it to the modelled
            # signal. If that publisher executes first, object-graph step zero
            # consumes its initialized history value.
            if not isinstance(port, tps.Vector):
                immediate = functional_model._connection_sources(component, key[1])
                pieces = []
                for producer, output_name, routes in immediate:
                    if (
                        functional_model.pos.get(producer.id, -1)
                        >= functional_model.pos[component.id]
                    ):
                        pieces = []
                        break
                    published = self._history_value(producer.output[output_name], 0, 0)
                    if published is None:
                        pieces = []
                        break
                    pieces.append((producer, output_name, routes))
                if pieces:
                    for period in range(self.n_periods):
                        merged = None
                        for producer, output_name, routes in pieces:
                            published = self._history_value(
                                producer.output[output_name], 0, period
                            )
                            routed = functional_model._apply_routes(published, routes)
                            merged = routed if merged is None else merged + routed
                        feedback[period, slc] = merged.reshape(-1) * mask.to(device)
        # External requested outputs are materialized by the same minimal
        # producer closure used for exogenous tape. This never steps a cone component.
        for index, (component, _) in enumerate(self.outputs):
            if self.functional_model.meas_sources[index][0] != "external":
                continue
            for step in range(self.max_t):
                self._step_external(component, step, external_done[step], set())
        y0 = torch.stack([self.layout.gather(p) for p in range(self.n_periods)])
        if functional_model.n_feedback:
            y0 = torch.cat([y0, feedback], dim=1)
        self.cached_recording = (y0.detach().clone(), exogenous_tape.detach().clone())
        return y0, exogenous_tape

    def _full_rollout(self, y0, theta, exogenous_tape, *, transform_mode=False):
        state_rows = []
        output_rows = []
        for period in range(self.n_periods):
            states, outputs = functional_rollout_tape(
                self.functional_model,
                y0[period],
                theta,
                exogenous_tape[:, period],
                transform_mode=transform_mode,
            )
            state_rows.append(states)
            output_rows.append(outputs)
        return torch.stack(state_rows, dim=1), torch.stack(output_rows, dim=1)

    def _full_rollout_graph(self, y0, theta, exogenous_tape):
        # Transform mode avoids value-dependent discretization caches
        # (torch.allclose / boolean indexing), which are illegal during stream
        # capture, while retaining the same fixed-shape tensor equations.
        return self._full_rollout(y0, theta, exogenous_tape, transform_mode=True)

    def rollout(self, backend, y0, exogenous_tape):
        if backend == "eager":
            states, outputs = self._full_rollout(y0, self.theta, exogenous_tape)
            return RolloutResult(states, outputs)
        if self.theta.device.type != "cuda":
            raise RuntimeError(
                "execution_backend='cuda_graph' requires a CUDA model; "
                "call model.to('cuda') first"
            )
        if self.graph is None:
            started = time.perf_counter()
            self.graph = CudaGraphCallable(self._full_rollout_graph)
            states, outputs = self.graph(y0, self.theta, exogenous_tape)
            torch.cuda.synchronize()
            self.capture_seconds += time.perf_counter() - started
            self.capture_count += 1
        else:
            started = time.perf_counter()
            states, outputs = self.graph(y0, self.theta, exogenous_tape)
            torch.cuda.synchronize()
            self.replay_seconds += time.perf_counter() - started
            self.replay_count += 1
        return RolloutResult(states.clone(), outputs.clone())

    def materialize_outputs(self, result):
        """Write the rollout output tape into public output ports."""
        for index, (component, output_name) in enumerate(self.outputs):
            port = component.output[output_name]
            source = self.functional_model.meas_sources[index]
            if source[0] == "external":
                if port._history is None:
                    raise RuntimeError(
                        f"functional rollout cannot materialize external output "
                        f"{component.id}.{output_name}"
                    )
                port._history_is_populated = True
                continue
            slc = self.functional_model.meas_slices[index]
            flat = result.outputs[:, :, slc]
            if isinstance(port, tps.Vector):
                values = flat.reshape(self.max_t, self.n_periods, port.n_c, port.n_v)
            else:
                values = flat.reshape(self.max_t, self.n_periods, port.n_c)
            port._history.copy_(values)
            port._history_is_populated = True
            port._tensor.copy_(values[-1])

    @staticmethod
    def _index(index, device):
        if index is None:
            return slice(None)
        if isinstance(index, slice):
            return index
        return torch.as_tensor(index, dtype=torch.long, device=device).reshape(-1)

    @staticmethod
    def _vector_index(index, device):
        if index is None:
            return slice(None)
        if isinstance(index, (int, slice)):
            return index
        if isinstance(index, torch.Tensor) and index.ndim == 0:
            return index.to(dtype=torch.long, device=device)
        return torch.as_tensor(index, dtype=torch.long, device=device)

    @classmethod
    def _select_components(cls, value, index, dimension):
        index = cls._index(index, value.device)
        if isinstance(index, slice):
            slices = [slice(None)] * value.ndim
            slices[dimension] = index
            return value[tuple(slices)]
        return torch.index_select(value, dimension, index)

    @classmethod
    def _route_history(cls, source_port, output_port_index, output_component_index):
        value = cls._select_components(
            source_port._history, output_component_index, dimension=2
        )
        if isinstance(source_port, tps.Vector):
            value = value[..., cls._vector_index(output_port_index, value.device)]
        return value

    @classmethod
    def _route_tensor(cls, source_port, output_port_index, output_component_index):
        value = cls._select_components(
            source_port._tensor, output_component_index, dimension=1
        )
        if isinstance(source_port, tps.Vector):
            value = value[..., cls._vector_index(output_port_index, value.device)]
        return value

    @classmethod
    def _assign_routed(
        cls,
        target,
        value,
        *,
        component_dimension,
        component_index,
        vector_index,
    ):
        component_index = cls._index(component_index, target.device)
        vector_index = cls._vector_index(vector_index, target.device)
        if isinstance(component_index, slice):
            slices = [slice(None)] * target.ndim
            slices[component_dimension] = component_index
            destination = target[tuple(slices)]
            if target.ndim == component_dimension + 2:
                destination[..., vector_index] = value
            else:
                destination.copy_(value)
            return

        if target.ndim == component_dimension + 2:
            destination = target[..., vector_index]
            destination.index_copy_(component_dimension, component_index, value)
        else:
            target.index_copy_(component_dimension, component_index, value)

    def materialize_inputs(self):
        """Route complete producer histories into connected input ports."""
        connected_ports = []
        seen_ports = set()
        for component in self.model.components.values():
            for connection_point in component.connects_at:
                input_port = component.input[connection_point.input_port]
                if id(input_port) not in seen_ports:
                    connected_ports.append(
                        (component, connection_point.input_port, input_port)
                    )
                    seen_ports.add(id(input_port))
                for connection in connection_point.connects_system_through:
                    output_port = connection.connects_system.output[
                        connection.output_port
                    ]
                    output_port_index = connection_point.output_port_index[connection]
                    input_port_index = connection_point.input_port_index[connection]
                    output_component_index = (
                        connection_point.output_component_index.get(
                            connection, slice(None)
                        )
                    )
                    input_component_index = connection_point.input_component_index.get(
                        connection, slice(None)
                    )
                    try:
                        if (
                            input_port._history is not None
                            and output_port._history is not None
                        ):
                            history = self._route_history(
                                output_port,
                                output_port_index,
                                output_component_index,
                            )
                            self._assign_routed(
                                input_port._history,
                                history,
                                component_dimension=2,
                                component_index=input_component_index,
                                vector_index=input_port_index,
                            )
                        current = self._route_tensor(
                            output_port,
                            output_port_index,
                            output_component_index,
                        )
                        self._assign_routed(
                            input_port._tensor,
                            current,
                            component_dimension=1,
                            component_index=input_component_index,
                            vector_index=input_port_index,
                        )
                    except IndexError as exc:
                        raise IndexError(
                            "Batched component-index mapping failed for "
                            f"{connection.connects_system.id}.{connection.output_port} "
                            f"-> {component.id}.{connection_point.input_port}; "
                            f"output_i_c={output_component_index}, "
                            f"input_i_c={input_component_index}"
                        ) from exc

        for _, _, port in connected_ports:
            if port.log_history and port._history is not None:
                port._history_is_populated = True
                port._tensor.copy_(port._history[-1])
        return connected_ports

    def finalize_materialization(self, result):
        self.layout.scatter(result.states[-1, :, : self.functional_model.D])

    def cache_public_histories(self):
        """Snapshot source histories not regenerated by the functional rollout."""
        source_ports = {}
        for index, (component, output_name) in enumerate(self.outputs):
            if self.functional_model.meas_sources[index][0] == "external":
                port = component.output[output_name]
                source_ports[id(port)] = port
        components = dict(self.model.components)
        sim_model = getattr(self.model, "_simulation_model", None) or self.model
        components.update(getattr(sim_model, "_fused_components", None) or {})
        for key in self.functional_model._exogenous_keys:
            component = components[key[0]]
            if not self._exogenous_sources(component, key):
                port = component.input[key[1]]
                source_ports[id(port)] = port
        self.cached_public_histories = tuple(
            (port, port._history.detach().clone())
            for port in source_ports.values()
            if port._history is not None and port._history_is_populated
        )

    def materialize(self, simulator, result):
        """Compatibility wrapper for complete functional materialization."""
        self.materialize_outputs(result)
        self.materialize_inputs()
        simulator._validate_component_inputs(self.model)
        self.finalize_materialization(result)


def run_functional_simulation(simulator, backend, step_size, max_t, n_periods):
    """Execute a public functional/cuda_graph simulation after initialization."""
    sim_model = getattr(simulator.model, "_simulation_model", None) or simulator.model
    device = torch.device(getattr(sim_model, "device", torch.device("cpu")))

    def prepare_session():
        signature = session_signature(
            simulator.model,
            step_size,
            max_t,
            n_periods,
            simulator.date_time_steps,
        )
        session = getattr(simulator, "_functional_session", None)
        if session is None or session.signature != signature:
            replacement = FunctionalSimulationSession(
                simulator, step_size, max_t, n_periods, signature
            )
            simulator._functional_session = replacement
            if session is not None:
                session.close()
            session = replacement
            simulator._functional_setup_count = (
                getattr(simulator, "_functional_setup_count", 0) + 1
            )
        session.refresh_parameters()
        return session

    session, setup_seconds = _timed_phase(device, prepare_session)
    simulator._functional_setup_seconds = setup_seconds
    (y0, exogenous_tape), exogenous_seconds = _timed_phase(
        device, session.record_exogenous_inputs
    )
    result, rollout_seconds = _timed_phase(
        device, lambda: session.rollout(backend, y0, exogenous_tape)
    )
    _, output_materialization_seconds = _timed_phase(
        device, lambda: session.materialize_outputs(result)
    )
    _, input_materialization_seconds = _timed_phase(device, session.materialize_inputs)
    validation_started = time.perf_counter()
    validation_counts = simulator._validate_component_inputs(session.model)
    validation_seconds = time.perf_counter() - validation_started
    _, finalization_seconds = _timed_phase(
        device, lambda: session.finalize_materialization(result)
    )
    session.cache_public_histories()
    validation_check_count, validation_host_sync_count = validation_counts
    simulator._exogenous_recording_seconds = exogenous_seconds
    simulator._exogenous_recording_cache_hit = session.recording_cache_hit
    simulator._exogenous_recording_cache_hits = session.recording_cache_hits
    simulator._exogenous_recording_cache_misses = session.recording_cache_misses
    simulator._functional_rollout_seconds = rollout_seconds
    simulator._output_materialization_seconds = output_materialization_seconds
    simulator._input_materialization_seconds = input_materialization_seconds
    simulator._functional_validation_seconds = validation_seconds
    simulator._functional_finalization_seconds = finalization_seconds
    simulator._functional_validation_check_count = validation_check_count
    simulator._functional_validation_host_sync_count = validation_host_sync_count
    simulator._cuda_graph_capture_count = session.capture_count
    simulator._cuda_graph_replay_count = session.replay_count
    simulator._cuda_graph_capture_seconds = session.capture_seconds
    simulator._cuda_graph_replay_seconds = session.replay_seconds
    simulator._last_exogenous_tape = exogenous_tape
    simulator._last_functional_initial_state = y0
    simulator._last_execution_metadata = {
        "execution_mode": "functional",
        "execution_backend": backend,
        "functional_materialization_revision": FUNCTIONAL_MATERIALIZATION_REVISION,
        "sequential_rollout": True,
        "vmap": False,
        "functional_setup_count": simulator._functional_setup_count,
        "functional_setup_seconds": simulator._functional_setup_seconds,
        "model_initialization_seconds": getattr(
            simulator, "_functional_initialization_seconds", None
        ),
        "exogenous_recording_seconds": exogenous_seconds,
        "exogenous_recording_cache_hit": session.recording_cache_hit,
        "exogenous_recording_cache_hits": session.recording_cache_hits,
        "exogenous_recording_cache_misses": session.recording_cache_misses,
        "functional_rollout_seconds": rollout_seconds,
        "output_materialization_seconds": output_materialization_seconds,
        "input_materialization_seconds": input_materialization_seconds,
        "validation_seconds": validation_seconds,
        "finalization_seconds": finalization_seconds,
        "validation_check_count": validation_check_count,
        "validation_host_sync_count": validation_host_sync_count,
        "cuda_graph_capture_count": session.capture_count,
        "cuda_graph_replay_count": session.replay_count,
        "cuda_graph_capture_seconds": session.capture_seconds,
        "cuda_graph_replay_seconds": session.replay_seconds,
    }
    return result
