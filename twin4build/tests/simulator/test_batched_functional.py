"""Focused batching/functional parity for the complete full-workflow graph."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
import torch

import twin4build as tb
import twin4build.utils.types as tps
from benchmarks import common
from twin4build.simulator._functional_simulation import FunctionalSimulationSession


def _batched_full_workflow(n_zones, model_id):
    model, parts = common.build_multizone_model(n_zones, model_id=model_id)
    common._align_repeated_roles_for_batch(model, parts["zone_parts"])
    batched, _ = common.batch_model(model, measure=False)
    return batched


def _port_histories(model):
    result = {}
    for component in model.components.values():
        for direction in ("input", "output"):
            for name, port in getattr(component, direction).items():
                if port._history is not None and port._history_is_populated:
                    result[(component.id, direction, name)] = (
                        port._history.detach().cpu()
                    )
    return result


def test_complete_two_zone_object_functional_parity():
    audit = common.batched_parity_audit(2, hours=1)
    assert audit["n_components"] == 46
    assert audit["n_connections"] == 64
    assert audit["n_states"] == 26
    assert audit["n_theta"] == 56
    assert audit["max_abs_error"] < 3e-5


@pytest.mark.parametrize(
    ("device", "backend"),
    [
        ("cpu", "eager"),
        pytest.param(
            "cuda",
            "cuda_graph",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA Graph requires CUDA"
            ),
        ),
    ],
)
def test_public_batched_functional_replay_materializes_all_input_histories(
    device, backend
):
    model, parts = common.build_multizone_model(
        1, model_id="batched_full_workflow_functional_histories"
    )
    common._align_repeated_roles_for_batch(model, parts["zone_parts"])
    batched, _ = common.batch_model(model, measure=False)
    batched.to(device, torch.float64)

    simulator = tb.Simulator(
        batched,
        execution_mode="functional",
        execution_backend=backend,
    )
    kwargs = {
        "start_time": common.START,
        "end_time": common.START + dt.timedelta(hours=1),
        "step_size": common.STEP_SIZE,
        "show_progress_bar": False,
    }
    simulator.simulate(**kwargs)
    first_histories = {}
    for component in batched.components.values():
        for name, port in component.input.items():
            if not port.log_history or port._history is None:
                continue
            assert port._history_is_populated, f"first run: {component.id}.{name}"
            first_histories[(component.id, name)] = port._history.detach().clone()
    simulator.simulate(**kwargs)

    if backend == "cuda_graph":
        assert simulator._cuda_graph_capture_count == 1
        assert simulator._cuda_graph_replay_count >= 1
    for component in batched.components.values():
        for name, port in component.input.items():
            if not port.log_history or port._history is None:
                continue
            assert port._history_is_populated, f"{component.id}.{name}"
            assert torch.isfinite(port._history).all(), f"{component.id}.{name}"
            torch.testing.assert_close(
                port._history,
                first_histories[(component.id, name)],
                rtol=0,
                atol=0,
                msg=lambda message, key=(component.id, name): f"{key}: {message}",
            )


def test_nested_occupancy_parameters_and_ports_keep_n_c():
    model, parts = common.build_multizone_model(
        2, model_id="batched_full_workflow_nested_parameters"
    )
    common._align_repeated_roles_for_batch(model, parts["zone_parts"])
    batched, _ = common.batch_model(model, measure=False)

    source = parts["zone_parts"][0]["office_occupancy"]
    meta, _ = model.get_batched_component_info(source.id)
    occupancy = batched.components[meta.id]
    expected = np.asarray([row["mass.V"] for row in parts["parameters"]], dtype=float)
    np.testing.assert_allclose(
        occupancy.mass.V.get().detach().cpu().numpy().reshape(-1), expected
    )

    batched.initialize(
        [common.START],
        [common.START + dt.timedelta(hours=1)],
        [common.STEP_SIZE],
    )
    assert occupancy.input["outdoorCo2Concentration"].n_c == 2
    assert occupancy.output["scheduleValue"].n_c == 2

    pid_source = parts["zone_parts"][0]["office_co2_controller"]
    pid_meta, _ = model.get_batched_component_info(pid_source.id)
    pid = batched.components[pid_meta.id]
    assert pid.input["actualValue"].n_c == 2
    assert pid.output["inputSignal"].n_c == 2


def test_full_estimation_setup_has_all_groups_and_measurements():
    setup = common.batched_estimation_problem(
        1, common.BenchmarkConfig(mode="smoke"), measure_batch=False
    )
    assert len(setup["parameters"]) == 26
    assert len(setup["measurements"]) == 4
    assert setup["topology"]["n_theta"] == 28
    assert setup["topology"]["n_states"] == 13


def test_full_pareto_controls_costs_and_constraints_are_batched():
    setup = common.batched_pareto_problem(2, measure_batch=False)
    assert setup["variables"][0][0]._n_c_batched == 1
    assert setup["variables"][0][0].id == "shared_valve_position_schedule"
    assert setup["objective1"][0]._n_c_batched == 2
    assert setup["objective2"][0]._n_c_batched == 2
    assert len(setup["ineq_cons"]) == 2
    assert setup["objective1"][1:] == ("output", "min")
    assert setup["objective2"][1:] == ("indoorTemperature", "max")


def test_full_workflow_truth_is_not_broadcast():
    truth = common.physical_zone_parameters(10, common.SEED)
    for key in common.THETA_KEYS:
        values = torch.tensor([row[key] for row in truth], dtype=torch.float64)
        assert torch.unique(values).numel() == 10


def test_vectorized_routing_preserves_periods_indices_and_padding():
    source = tps.Vector(n_v=2)
    source.initialize(n_t=3, n_s=2, n_c=2)
    target = tps.Vector(n_v=3)
    target.initialize(n_t=3, n_s=2, n_c=2)
    values = torch.arange(24, dtype=torch.float64).reshape(3, 2, 2, 2)
    values[2, 1] = torch.nan
    source._history.copy_(values)
    source._tensor.copy_(values[-1])

    routed = FunctionalSimulationSession._route_history(source, 1, torch.tensor([1, 0]))
    FunctionalSimulationSession._assign_routed(
        target._history,
        routed,
        component_dimension=2,
        component_index=torch.tensor([0, 1]),
        vector_index=2,
    )
    torch.testing.assert_close(
        target._history[..., 2], values[:, :, [1, 0], 1], equal_nan=True
    )


@pytest.mark.parametrize("n_zones", [1, 2])
def test_vectorized_materialization_matches_reference_routing(n_zones):
    kwargs = {
        "start_time": common.START,
        "end_time": common.START + dt.timedelta(hours=1),
        "step_size": common.STEP_SIZE,
        "show_progress_bar": False,
    }
    functional_model = _batched_full_workflow(n_zones, f"functional_ports_{n_zones}")
    simulator = tb.Simulator(functional_model, execution_mode="functional")
    simulator.simulate(**kwargs)
    for component in functional_model.components.values():
        for direction in ("input", "output"):
            for name, port in getattr(component, direction).items():
                if port._history is not None and port._history_is_populated:
                    torch.testing.assert_close(
                        port._tensor.cpu(),
                        port._history[-1].cpu(),
                        msg=lambda msg, key=(component.id, direction, name): (
                            f"{key}: {msg}"
                        ),
                    )
    actual = {
        (component.id, point.input_port): component.input[point.input_port]
        ._history.detach()
        .clone()
        for component in functional_model.components.values()
        for point in component.connects_at
        if component.input[point.input_port]._history is not None
    }

    for component in functional_model.components.values():
        for point in component.connects_at:
            port = component.input[point.input_port]
            if port._history is not None:
                port._history.zero_()
                port._history_is_populated = False
    for step in range(simulator.n_timesteps):
        for component in functional_model.components.values():
            for port in component.output.values():
                if port._history_is_populated:
                    port._tensor.copy_(port._history[step])
        for component in functional_model.components.values():
            simulator._assign_component_inputs(component, step)

    for component in functional_model.components.values():
        for point in component.connects_at:
            port = component.input[point.input_port]
            if port._history is not None:
                torch.testing.assert_close(
                    actual[(component.id, point.input_port)], port._history
                )


def test_functional_input_materialization_never_calls_object_assignment(monkeypatch):
    model = _batched_full_workflow(2, "no_object_assignment_materialization")
    simulator = tb.Simulator(model, execution_mode="functional")
    simulator.simulate(
        start_time=common.START,
        end_time=common.START + dt.timedelta(hours=1),
        step_size=common.STEP_SIZE,
        show_progress_bar=False,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("_assign_component_inputs was called")

    monkeypatch.setattr(simulator, "_assign_component_inputs", forbidden)
    connected = simulator._functional_session.materialize_inputs()
    assert connected


def test_functional_timing_and_validation_metadata_are_bounded():
    model = _batched_full_workflow(2, "functional_phase_metadata")
    simulator = tb.Simulator(model, execution_mode="functional")
    simulator.simulate(
        start_time=common.START,
        end_time=common.START + dt.timedelta(hours=2),
        step_size=common.STEP_SIZE,
        show_progress_bar=False,
    )
    metadata = simulator._last_execution_metadata
    for phase in (
        "model_initialization_seconds",
        "exogenous_recording_seconds",
        "functional_rollout_seconds",
        "output_materialization_seconds",
        "input_materialization_seconds",
        "validation_seconds",
    ):
        assert metadata[phase] >= 0
    connected = {
        (component.id, point.input_port)
        for component in model.components.values()
        for point in component.connects_at
    }
    assert metadata["validation_check_count"] == len(connected)
    assert metadata["validation_host_sync_count"] == 1
    assert metadata["functional_materialization_revision"] == (
        common.FUNCTIONAL_MATERIALIZATION_REVISION
    )


def test_vectorized_validation_preserves_component_port_diagnostic():
    model = _batched_full_workflow(1, "functional_nan_diagnostic")
    simulator = tb.Simulator(model, execution_mode="functional")
    simulator.simulate(
        start_time=common.START,
        end_time=common.START + dt.timedelta(hours=1),
        step_size=common.STEP_SIZE,
        show_progress_bar=False,
    )
    connected = simulator._functional_session.materialize_inputs()
    component, name, port = connected[0]
    port._history[0, 0, 0] = torch.nan
    with pytest.raises(
        ValueError,
        match=rf"Input {name} of component {component.id} is non-finite at timestep=0",
    ):
        simulator._validate_component_inputs(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_graph_matches_eager_all_port_histories():
    kwargs = {
        "start_time": common.START,
        "end_time": common.START + dt.timedelta(hours=1),
        "step_size": common.STEP_SIZE,
        "show_progress_bar": False,
    }
    eager_model = _batched_full_workflow(2, "cuda_eager_port_parity")
    eager_model.to("cuda", torch.float64)
    tb.Simulator(eager_model, execution_mode="functional").simulate(**kwargs)
    eager = _port_histories(eager_model)

    graph_simulator = tb.Simulator(
        eager_model,
        execution_mode="functional",
        execution_backend="cuda_graph",
    )
    graph_simulator.simulate(**kwargs)
    graph_simulator.simulate(**kwargs)

    graph = _port_histories(eager_model)
    assert graph.keys() == eager.keys(), (
        f"graph_extra={graph.keys() - eager.keys()}, "
        f"graph_missing={eager.keys() - graph.keys()}"
    )
    for key in eager:
        torch.testing.assert_close(graph[key], eager[key], rtol=1e-5, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_ten_zone_cuda_graph_materialization_overhead_is_bounded():
    model = _batched_full_workflow(10, "cuda_graph_materialization_timing")
    model.to("cuda", torch.float64)
    simulator = tb.Simulator(
        model,
        execution_mode="functional",
        execution_backend="cuda_graph",
    )
    kwargs = {
        "start_time": common.START,
        "end_time": common.START + dt.timedelta(hours=120),
        "step_size": common.STEP_SIZE,
        "show_progress_bar": False,
    }
    simulator.simulate(**kwargs)
    _, elapsed = common.timed("cuda", lambda: simulator.simulate(**kwargs))
    metadata = simulator._last_execution_metadata
    assert metadata["input_materialization_seconds"] < 5.0
    assert metadata["validation_host_sync_count"] == 1
    assert elapsed < 0.75 * 22.209
