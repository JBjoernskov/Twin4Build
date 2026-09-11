"""Lightweight structural tests for the canonical benchmark notebooks."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS = ROOT / "benchmarks"
NOTEBOOKS = {
    "simulation_scaling_benchmark.ipynb",
    "estimation_scaling_benchmark.ipynb",
    "optimization_scaling_benchmark.ipynb",
}
# Colab runners/diagnostics that reuse the harness but are not authoritative
# result notebooks (no published-results section, no serialize_results cell).
AUXILIARY_NOTEBOOKS = {
    "colab_estimation_benchmark.ipynb",
    "colab_debug_multizone.ipynb",
}


def _common():
    return importlib.import_module("benchmarks.common")


def test_exactly_three_authoritative_notebooks():
    assert {path.name for path in BENCHMARKS.glob("*.ipynb")} == (
        NOTEBOOKS | AUXILIARY_NOTEBOOKS
    )


def test_notebooks_are_colab_ready_and_unexecuted():
    required = [
        "Published results",
        "Placeholder",
        "Methodology and environment",
        "GIT_REF",
        "google.colab",
        "smoke",
        "full",
        "seed",
        "serialize_results",
        "Interpretation",
    ]
    for name in NOTEBOOKS:
        notebook = nbformat.read(BENCHMARKS / name, as_version=4)
        text = "\n".join(
            "".join(cell.source) if isinstance(cell.source, list) else cell.source
            for cell in notebook.cells
        )
        assert notebook.cells[0].cell_type == "markdown"
        assert notebook.cells[0].source.startswith("# Published results")
        for marker in required:
            assert marker in text, f"{name} is missing {marker!r}"
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is None
                assert not cell.outputs


def test_canonical_matrices_and_scaling_sizes():
    common = _common()
    assert common.ZONE_COUNTS == [1, 10, 50, 100]
    assert {
        (
            row["device"],
            row["model_layout"],
            row["execution_mode"],
            row["execution_backend"],
        )
        for row in common.SIMULATION_MATRIX
    } == {
        ("cpu", "standard", "object", "eager"),
        ("cuda", "standard", "object", "eager"),
        ("cpu", "batched", "object", "eager"),
        ("cuda", "batched", "object", "eager"),
        ("cpu", "batched", "functional", "eager"),
        ("cuda", "batched", "functional", "eager"),
        ("cuda", "batched", "functional", "cuda_graph"),
    }
    assert set(common.ESTIMATION_MATRIX) == {
        ("cpu", "slsqp-single-shooting", 1),
        ("cuda", "slsqp-single-shooting", 1),
        ("cuda", "custom-batched-sqp", 1),
        ("cuda", "custom-batched-sqp", 8),
        ("cuda", "custom-batched-tr", 1),
        ("cuda", "custom-batched-tr", 8),
        ("cuda", "ipopt-collocation", 1),
    }
    assert common.ESTIMATION_METHODS["custom-batched-tr"] == ("custom", "batched-tr", "ad")
    assert common.ESTIMATION_CPU_MAX_ZONES == 10
    assert common.ESTIMATION_SOLVER_BUDGET == 300
    assert common.BenchmarkConfig(mode="full").estimation_repeats == 1
    assert set(common.OPTIMIZATION_MATRIX) == {
        ("cpu", "SLSQP"),
        ("cuda", "SLSQP"),
    }


def test_true_parameters_are_distinct_and_reproducible():
    common = _common()
    first = common.physical_zone_parameters(10, common.SEED)
    second = common.physical_zone_parameters(10, common.SEED)
    assert first == second
    assert len(first[0]) == common.CANONICAL_THETA_PER_ZONE
    assert all(len({row[key] for row in first}) == 10 for key in first[0])
    assert all(1e4 < row["thermal.C_air"] < 5e5 for row in first)
    assert all(0.01 < row["thermal.R_out"] < 1 for row in first)


def test_batched_mapping_preserves_heterogeneous_parameters():
    common = _common()
    model, parts = common.build_multizone_model(
        3, model_id="canonical_benchmark_structure_test"
    )
    assert len(model.components) == 69
    assert parts["topology"] == {
        "n_zones": 3,
        "n_components": 69,
        "n_connections": 96,
        "n_states": 39,
        "n_parameter_groups": 78,
        "n_theta": 84,
        "n_measurements": 12,
    }
    batched, batch_seconds = common.batch_model(model, measure=False)
    assert batch_seconds is None
    batching_mapping = common.batching_mapping_audit(
        model, [z.id for z in parts["zones"]]
    )
    assert set(batching_mapping) == {
        "zone_0__office",
        "zone_1__office",
        "zone_2__office",
    }
    assert {row["batch_size"] for row in batching_mapping.values()} == {3}

    batch_component_ids = {
        row["batch_component_id"] for row in batching_mapping.values()
    }
    assert len(batch_component_ids) == 1
    meta = batched.components[batch_component_ids.pop()]
    actual = meta.thermal.C_air.get().detach().cpu().numpy().reshape(-1)
    expected = [row["thermal.C_air"] for row in parts["parameters"]]
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    assert len(set(actual.tolist())) == 3


def _all_parameter_values(component):
    import torch

    values = {}
    if isinstance(component, torch.nn.Module):
        for name, obj in component.named_parameters(recurse=True):
            values[name] = obj.get().detach().reshape(-1).tolist()
    return values


def test_zone_copies_keep_every_template_parameter():
    """Zone replication deep-copies the template; a ``tps.Parameter`` deep copy
    used to re-normalize its data (unbounded values became 1.0), so gate
    steepness, the on/off controller output and the boundary node all drifted
    from the canonical example.  Every parameter must match the template."""
    common = _common()
    template = common._translated_template()
    components, roles = common._prefix_copy(template, 0)
    n_checked = 0
    for component in components:
        source_id = component.id.removeprefix("zone_0__")
        expected = _all_parameter_values(template.components[source_id])
        actual = _all_parameter_values(component)
        assert set(actual) == set(expected)
        for name in expected:
            np.testing.assert_allclose(
                actual[name], expected[name], rtol=1e-12, err_msg=f"{source_id}.{name}"
            )
            n_checked += 1
    assert n_checked > 40
    assert float(roles["office_occupancy_detector"].steepness.get()) == 10.0
    assert float(roles["office_occupancy_controller"].on_value.get()) == pytest.approx(0.3)


def test_truth_values_are_physical_after_replication():
    common = _common()
    model, parts = common.build_multizone_model(
        2, model_id="canonical_benchmark_truth_test"
    )
    truth = parts["parameters"][0]
    comps = model.components
    assert float(comps["zone_0__office"].thermal.C_air.get()) == pytest.approx(
        truth["thermal.C_air"], rel=1e-12
    )
    assert float(
        comps["zone_0__office_temperature_heating_controller"].kp.get()
    ) == pytest.approx(truth["heating_pid.kp"], rel=1e-12)
    assert float(comps["zone_0__office_occupancy_detector"].threshold.get()) == pytest.approx(
        truth["occupancy_detector.threshold"], rel=1e-12
    )
    # untouched by the truth: still the canonical example's values
    assert float(comps["zone_0__office_occupancy_detector"].steepness.get()) == 10.0
    assert float(comps["zone_0__office"].thermal.C_boundary.get()) == pytest.approx(1e6, rel=1e-6)


def test_batched_model_cuts_cycles_where_the_source_model_does():
    """The one-step lag of every feedback loop must sit on the same signal in
    the batched and the unbatched model; a meta component that batches several
    source components (every PID controller) adds cross-loop cycles, and the
    cycle-count criterion alone moved the cut onto a forward edge."""
    common = _common()
    model, parts = common.build_multizone_model(2, model_id="canonical_benchmark_cut_test")
    common._align_repeated_roles_for_batch(model, parts["zone_parts"])
    batched, _ = common.batch_model(model, measure=False)
    source_cuts = set(model.simulation_model._removed_cycle_edges)
    assert source_cuts, "the canonical model has feedback loops"

    def source_ids(meta_id):
        component = batched.components[meta_id]
        return getattr(component, "_source_component_ids", (meta_id,))

    batched_cuts = set()
    for c_from, c_to in batched.simulation_model._removed_cycle_edges:
        for a in source_ids(c_from):
            for b in source_ids(c_to):
                if a.split("__", 1)[0] == b.split("__", 1)[0]:  # same zone
                    batched_cuts.add((a, b))
    assert batched_cuts == source_cuts


def test_batched_estimation_problem_maps_all_private_slices_and_sensors():
    common = _common()
    setup = common.batched_estimation_problem(
        3, common.BenchmarkConfig(mode="smoke"), measure_batch=False
    )

    assert setup["batched_model"].id.endswith("_batched")
    assert len(setup["parameters"]) == common.CANONICAL_PARAMETER_GROUPS_PER_ZONE
    assert len(setup["parameter_groups"]) == common.CANONICAL_PARAMETER_GROUPS_PER_ZONE
    assert len(set(sensor.id for sensor, _ in setup["measurements"])) == 12
    assert all(sensor.df is not None for sensor, _ in setup["measurements"])
    assert all(
        len({row[key] for row in setup["truth"]}) == 3 for key in setup["truth"][0]
    )
    assert setup["topology"]["n_theta"] == 84


def test_batched_pareto_problem_aggregates_every_zone():
    common = _common()
    setup = common.batched_pareto_problem(3, measure_batch=False)

    assert len({row["thermal.C_air"] for row in setup["truth"]}) == 3
    assert len(setup["variables"]) == 1
    # Cost against discomfort: both minimised.  Discomfort (Kelvin-hours below
    # the heating setpoint) replaces the hard lower temperature bound, which
    # would otherwise pin it at zero and collapse the front; the cooling
    # setpoint stays a hard upper bound.
    assert setup["objective1"][2] == "min"
    assert setup["objective2"][2] == "min"
    assert setup["objective2"][0].id.endswith("FunctionSystem")
    assert setup["objective2"][0].n_c == 3
    assert [c[2] for c in setup["ineq_cons"]] == ["upper"]


def test_full_pareto_dimensions_use_shared_broadcast_schedule():
    dimensions = _common()._optimization_dimensions(300, 72)
    assert dimensions["valve_schedule_semantics"] == "shared_broadcast"
    assert dimensions["n_base_full_workflow_components"] == 6900
    assert dimensions["n_components_with_optimization"] == 7801
    assert dimensions["independent_valve_schedules"] == 1
    assert dimensions["n_decision_variables"] == 216
    assert dimensions["counterfactual_independent_zone_valve_schedules"] == 300
    assert dimensions["counterfactual_independent_n_decision_variables"] == 64800
    assert dimensions["n_soft_comfort_penalty_samples"] == 129600
    assert dimensions["pareto_control_variables"] == 216
    assert dimensions["pareto_boundary_state_variables"] > 216
    assert dimensions["pareto_exact_polish_decision_variables"] > 216
    assert dimensions["pareto_dynamics_constraints"] > 1
    assert dimensions["pareto_hard_epsilon_constraints_per_point"] == 1


def test_scaling_sources_use_batched_functional_semantics():
    common = _common()
    source = (BENCHMARKS / "common.py").read_text(encoding="utf-8")
    assert "BuildingSpaceThermalTorchSystem" not in source
    assert "WallTorchSystem" not in source
    assert "full_workflow_example.py" in source
    assert "CANONICAL_COMPONENTS_PER_ZONE = 23" in source
    assert "CANONICAL_PARAMETER_GROUPS_PER_ZONE = 26" in source
    assert "CANONICAL_MEASUREMENTS_PER_ZONE = 4" in source
    assert "weekDayRulesetDict" not in source
    assert common.ESTIMATION_SOLVER_BUDGET == 300
    assert common.OPTIMIZATION_SOLVER_BUDGET == 300

    estimation = nbformat.read(
        BENCHMARKS / "estimation_scaling_benchmark.ipynb", as_version=4
    )
    optimization = nbformat.read(
        BENCHMARKS / "optimization_scaling_benchmark.ipynb", as_version=4
    )
    estimation_text = "\n".join(cell.source for cell in estimation.cells)
    optimization_text = "\n".join(cell.source for cell in optimization.cells)
    assert "batched model layout" in estimation_text
    assert 'hessian="exact"' in estimation_text
    assert "CPU collocation is excluded" in estimation_text
    assert "batched_prepass=True" in optimization_text
    assert "SLSQP" in optimization_text
    assert "IPOPT" in optimization_text
    assert "exact sparse segment-local Lagrangian Hessian" in optimization_text
    assert "IPOPT solver" in optimization_text
    assert "23-component" in estimation_text + optimization_text
    assert "28 theta" in estimation_text + optimization_text
    assert "model_layout" in source
    assert "execution_backend" in source


def test_benchmark_sources_exclude_retired_terminology():
    text = "\n".join(
        [(BENCHMARKS / "common.py").read_text(encoding="utf-8")]
        + [(BENCHMARKS / name).read_text(encoding="utf-8") for name in NOTEBOOKS]
    )
    for retired in (
        "object" + "_graph",
        "com" + "posed",
        "build_" + "com" + "piled_model",
        "get_" + "com" + "piled_component_info",
        "exact_" + "hessian",
        "capture_" + "hessian",
    ):
        assert retired not in text


def test_scaling_preflights_retain_unsafe_dimensions():
    common = _common()
    safe = common._collocation_preflight(1, 2)
    assert safe["implementation_supported"]
    previous_vram = 0
    for zones in (1, 10, 50, 100, 300):
        collocation = common._collocation_preflight(zones, 120)
        assert collocation["implementation_supported"]
        assert collocation["replica_count"] == zones
        assert collocation["replica_theta_width"] == 28
        assert collocation["replica_state_width"] == 16
        # Sparse storage never trips the cap at these sizes; the empirical
        # VRAM model (1.16 GiB/zone measured on an A100) decides, and its
        # verdict depends on the card the test runs on -- so check the
        # consistency of the verdict rather than its value.
        assert collocation["estimated_peak_vram_bytes"] > previous_vram
        previous_vram = collocation["estimated_peak_vram_bytes"]
        assert collocation["mathematically_safe"] == collocation["vram_safe"]
        assert (collocation["preflight_reason"] is None) == collocation["vram_safe"]
        if collocation["cuda_device_total_bytes"] is None:
            assert collocation["vram_safe"]
        pareto = common._pareto_preflight(zones, 72)
        assert pareto["preflight_n_scalar_constraints"] > 1
        assert pareto["sparse_jacobian_nnz"] > pareto["preflight_n_variables"]
        assert pareto["sparse_hessian_nnz"] > pareto["preflight_n_variables"]
        assert pareto["mathematically_safe"]
        assert pareto["replica_count"] == zones
        assert pareto["replica_state_width"] == 16


def test_no_disallowed_benchmark_solvers():
    source = (BENCHMARKS / "common.py").read_text(encoding="utf-8")
    notebook_text = "\n".join(
        (BENCHMARKS / name).read_text(encoding="utf-8") for name in NOTEBOOKS
    )
    assert "trust-constr" not in source + notebook_text
    assert "L-BFGS-B" not in source + notebook_text


def test_batched_unbatched_smoke_parity():
    common = _common()
    audit = common.batched_parity_audit(2, hours=1)
    assert audit["status"] == "passed"
    assert audit["max_abs_error"] < 3e-5


def test_resume_invalidates_only_stale_functional_materialization_rows(
    tmp_path, monkeypatch
):
    common = _common()
    config = common.BenchmarkConfig(mode="smoke")
    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path)
    base = {
        "status": "ok",
        "n_zones": 1,
        "horizon_hours": config.hours,
        "step_size_seconds": common.STEP_SIZE,
        "model_layout": "batched",
        "execution_backend": "eager",
    }
    rows = [
        {
            **base,
            "row_id": "object-stale",
            "execution_mode": "object",
        },
        {
            **base,
            "row_id": "object-current",
            "execution_mode": "object",
            "benchmark_implementation_revision": (
                common.BENCHMARK_IMPLEMENTATION_REVISION
            ),
        },
        {
            **base,
            "row_id": "functional-stale",
            "execution_mode": "functional",
        },
        {
            **base,
            "row_id": "functional-current",
            "execution_mode": "functional",
            "functional_materialization_revision": (
                common.FUNCTIONAL_MATERIALIZATION_REVISION
            ),
            "benchmark_implementation_revision": (
                common.BENCHMARK_IMPLEMENTATION_REVISION
            ),
        },
        {
            **base,
            "status": "retained_summary",
            "row_id": "object-historical-retained",
            "execution_mode": "object",
            "benchmark_implementation_revision": (
                common.HISTORICAL_BENCHMARK_IMPLEMENTATION_REVISION
            ),
            "current_publication_eligible": False,
        },
    ]
    path = tmp_path / "simulation_scaling_smoke_in_progress.json"
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    resumed = common.resume_simulation_rows(config)
    assert {row["row_id"] for row in resumed} == {
        "object-current",
        "functional-current",
        "object-historical-retained",
    }


def _estimation_row(common, config, status, *, child_pid=1):
    return {
        **common._estimation_case_base(
            config, 1, "cpu", "slsqp-single-shooting", 1
        ),
        "status": status,
        "repetition": 0,
        "seconds": 0.25,
        "child_pid": child_pid,
    }


def test_estimation_cases_launch_fresh_child_commands(tmp_path, monkeypatch):
    common = _common()
    config = common.BenchmarkConfig(mode="smoke")
    result_dir = tmp_path / "results"
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "estimation_case_probe.py").write_text(
        "\n".join(
            (
                "import argparse, json, os",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--request', type=Path, required=True)",
                "parser.add_argument('--result', type=Path, required=True)",
                "args = parser.parse_args()",
                "request = json.loads(args.request.read_text(encoding='utf-8'))",
                "row = {'status': 'ok', 'child_pid': os.getpid(), "
                "'seconds': 0.01}",
                "args.result.write_text(json.dumps({'ok': True, 'row': row}), "
                "encoding='utf-8')",
            )
        ),
        encoding="utf-8",
    )
    pythonpath = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH",
        str(probe_dir) + (os.pathsep + pythonpath if pythonpath else ""),
    )
    monkeypatch.setattr(common, "RESULTS_DIR", result_dir)
    monkeypatch.setattr(common, "ESTIMATION_CASE_MODULE", "estimation_case_probe")
    rows = [
        common._run_estimation_case_subprocess(
            config,
            n_zones=1,
            device="cpu",
            solver="slsqp-single-shooting",
            n_starts=1,
            repetition=0,
        )
        for _ in range(2)
    ]

    assert rows[0]["child_pid"] != rows[1]["child_pid"]
    assert all(row["child_pid"] != os.getpid() for row in rows)


def test_failed_estimation_row_is_retained_and_retried(tmp_path, monkeypatch):
    common = _common()
    config = common.BenchmarkConfig(mode="smoke")
    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path)
    failed = _estimation_row(common, config, "failed", child_pid=5001)
    failed["seconds_median"] = failed["seconds"]
    checkpoint = tmp_path / "estimation_scaling_smoke_in_progress.json"
    checkpoint.write_text(json.dumps({"rows": [failed]}), encoding="utf-8")
    monkeypatch.setattr(
        common,
        "ESTIMATION_MATRIX",
        [("cpu", "slsqp-single-shooting", 1)],
    )
    monkeypatch.setattr(common, "available_devices", lambda: (["cpu"], []))
    monkeypatch.setattr(
        common,
        "_run_estimation_case_subprocess",
        lambda *args, **kwargs: _estimation_row(
            common, config, "nonconverged", child_pid=5002
        ),
    )

    rows = common.run_estimation_scaling(config)

    assert [row["status"] for row in rows] == ["failed", "nonconverged"]
    assert [row["child_pid"] for row in rows] == [5001, 5002]
    assert "seconds_median" not in rows[0]
    assert rows[1]["seconds_median"] == rows[1]["seconds"]


def test_estimation_parent_checkpoints_after_every_case(tmp_path, monkeypatch):
    common = _common()
    config = common.BenchmarkConfig(mode="smoke")
    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(common, "resume_estimation_rows", lambda _config: [])
    monkeypatch.setattr(
        common,
        "ESTIMATION_MATRIX",
        [
            ("cpu", "slsqp-single-shooting", 1),
            ("cpu", "custom-batched-sqp", 1),
        ],
    )
    monkeypatch.setattr(common, "available_devices", lambda: (["cpu"], []))
    pids = iter((6001, 6002))

    def fake_case(_config, **kwargs):
        return {
            **common._estimation_case_base(
                config,
                kwargs["n_zones"],
                kwargs["device"],
                kwargs["solver"],
                kwargs["n_starts"],
            ),
            "status": "ok",
            "repetition": kwargs["repetition"],
            "seconds": 0.1,
            "child_pid": next(pids),
        }

    snapshots = []
    real_checkpoint = common.checkpoint_results

    def recording_checkpoint(benchmark, checkpoint_config, rows, git_ref=None):
        path = real_checkpoint(benchmark, checkpoint_config, rows, git_ref)
        snapshots.append(json.loads(path.read_text(encoding="utf-8"))["rows"])
        assert not path.with_suffix(".tmp").exists()
        return path

    monkeypatch.setattr(common, "_run_estimation_case_subprocess", fake_case)
    monkeypatch.setattr(common, "checkpoint_results", recording_checkpoint)

    rows = common.run_estimation_scaling(config)

    assert len(rows) == 2
    assert [len(snapshot) for snapshot in snapshots] == [1, 2]


def test_postfit_quality_forces_eager_and_restores_backend():
    common = _common()

    class Simulator:
        execution_backend = "cuda_graph"

        def simulate(self, **kwargs):
            assert kwargs["execution_mode"] == "functional"
            assert kwargs["execution_backend"] == "eager"
            self.execution_backend = "mutated-by-test-double"
            raise RuntimeError("stop after backend assertion")

    simulator = Simulator()
    estimator = SimpleNamespace(simulator=simulator)
    with pytest.raises(RuntimeError, match="backend assertion"):
        common._postfit_prediction_quality(
            estimator,
            common.BenchmarkConfig(mode="smoke"),
            n_zones=1,
            device="cuda",
            solver="slsqp-single-shooting",
            n_starts=1,
            repetition=0,
        )
    assert simulator.execution_backend == "cuda_graph"


def test_parameter_entries_from_estimation_result_use_result_x():
    common = _common()
    space = SimpleNamespace(id="space")
    wall = SimpleNamespace(id="wall")
    heating = SimpleNamespace(id="heating")
    cooling = SimpleNamespace(id="cooling")
    supply = SimpleNamespace(id="supply")
    exhaust = SimpleNamespace(id="exhaust")
    entries = [
        (space, "thermal.C_air", [1.0, 2.0], 0.0, 20.0),
        (wall, "C", 1.0, 0.0, 200.0, "private"),
        ([heating, cooling], "Ti", [10.0, 10.0], 0.0, 50.0, "private"),
        ([supply, exhaust], "a", [1.0, 1.0], 0.0, 10.0, "shared"),
    ]
    result = {
        "result_x": [11.0, 12.0, 99.0, 21.0, 22.0, 31.0, 32.0, 4.0, 5.0],
        "component_id": [
            "space",
            "wall",
            "heating",
            "cooling",
            "supply",
            "exhaust",
        ],
        "component_attr": [
            "thermal.C_air",
            "C",
            "Ti",
            "Ti",
            "a",
            "a",
        ],
        "theta_mask": [0, 1, 2, 3, 4, 4],
        "unique_param_n_c": [2, 1, 2, 2, 2],
    }

    updated = common._parameter_entries_from_estimation_result(entries, result)

    assert updated[0][:5] == (space, "thermal.C_air", [11.0, 12.0], 0.0, 20.0)
    assert updated[1][:6] == (wall, "C", 99.0, 0.0, 200.0, "private")
    assert updated[2][:6] == (heating, "Ti", [21.0, 22.0], 0.0, 50.0, "private")
    assert updated[3][:6] == (cooling, "Ti", [31.0, 32.0], 0.0, 50.0, "private")
    assert updated[4][:6] == (
        [supply, exhaust],
        "a",
        [4.0, 5.0],
        0.0,
        10.0,
        "shared",
    )


def test_slsqp5_collocation_is_cuda_only_functional_graph():
    common = _common()
    config = common.BenchmarkConfig(mode="smoke")
    base = common._estimation_case_base(
        config, 1, "cuda", "slsqp5-ipopt-collocation", 1
    )
    assert base["execution_mode"] == "functional"
    assert base["execution_backend"] == "cuda_graph"
    assert base["solver_variant"] == "slsqp5-then-collocation"
    assert base["method"] == ("staged", "slsqp5-ipopt-collocation", "ad")
    assert base["preflight_kind"] == "collocation_hessian_exact"
    with pytest.raises(ValueError, match="CUDA-only"):
        common._run_slsqp5_then_collocation(
            {"model": SimpleNamespace(), "parameters": [], "measurements": []},
            config,
            "cpu",
        )
