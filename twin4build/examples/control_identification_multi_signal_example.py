"""
Controller Identification Example with Multiple Signals (Decoy Sensors/Setpoints)

This example tests the beta (sensor selection) and gamma (setpoint selection)
weights by introducing decoy signals that the optimizer should learn to ignore.

Phase 1: Generate realistic data using a known PI controller
    - True sensor: Indoor temperature
    - True setpoint: Room temperature setpoint

Phase 2: Identification with decoy signals
    - Sensors: [Indoor temp (TRUE), Outdoor temp (DECOY), Noisy temp (DECOY)]
    - Setpoints: [Room setpoint (TRUE), Constant 22°C (DECOY), Inverted setpoint (DECOY)]

The optimizer should learn:
    - beta_0 ≈ 1.0, beta_1 ≈ 0, beta_2 ≈ 0 (select indoor temp only)
    - gamma_0 ≈ 1.0, gamma_1 ≈ 0, gamma_2 ≈ 0 (select real setpoint only)
"""

# Standard library imports
import datetime
import os
import tempfile

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dateutil import tz

# Local application imports
import twin4build as tb


def create_weather_data(
    n_timesteps: int,
    step_size: float,
    start_time: datetime.datetime,
) -> dict:
    """
    Create realistic weather data for the simulation.
    """
    timestamps = [
        start_time + datetime.timedelta(seconds=i * step_size)
        for i in range(n_timesteps)
    ]

    hours = np.array(
        [
            (start_time + datetime.timedelta(seconds=i * step_size)).hour
            + (start_time + datetime.timedelta(seconds=i * step_size)).minute / 60
            for i in range(n_timesteps)
        ]
    )

    # Outdoor temperature: cold winter day
    outdoor_temp = -3.0 + 5.0 * np.sin(2 * np.pi * (hours - 14) / 24)
    outdoor_temp += np.random.normal(0, 0.5, n_timesteps)

    # Solar radiation
    solar = np.maximum(0, 400 * np.sin(np.pi * (hours - 6) / 12))
    solar = np.where((hours >= 6) & (hours <= 18), solar, 0)
    solar += np.random.normal(0, 20, n_timesteps)
    solar = np.maximum(0, solar)

    # Occupancy
    occupancy = np.where(
        (hours >= 8) & (hours <= 17), np.random.randint(1, 5, n_timesteps), 0
    ).astype(float)

    # Setpoint: varies through the day
    setpoint = np.ones(n_timesteps) * 20.0
    setpoint = np.where((hours >= 6) & (hours < 8), 21.0, setpoint)
    setpoint = np.where((hours >= 8) & (hours < 18), 21.5, setpoint)
    setpoint = np.where((hours >= 18) & (hours < 22), 20.5, setpoint)
    setpoint = np.where((hours >= 22) | (hours < 6), 18.0, setpoint)

    data = {
        "outdoor_temp": pd.DataFrame({"datetime": timestamps, "value": outdoor_temp}),
        "solar": pd.DataFrame({"datetime": timestamps, "value": solar}),
        "occupancy": pd.DataFrame({"datetime": timestamps, "value": occupancy}),
        "setpoint": pd.DataFrame({"datetime": timestamps, "value": setpoint}),
        "supply_flow": pd.DataFrame(
            {"datetime": timestamps, "value": np.zeros(n_timesteps)}
        ),
        "exhaust_flow": pd.DataFrame(
            {"datetime": timestamps, "value": np.zeros(n_timesteps)}
        ),
        "supply_temp": pd.DataFrame(
            {"datetime": timestamps, "value": np.ones(n_timesteps) * 35.0}
        ),
    }

    return data


def generate_data_with_twin4build(
    n_timesteps: int = 1440,
    step_size: int = 60,
    kp: float = 0.1,
    Ti: float = 10,
):
    """
    Generate synthetic data using actual twin4build components.
    """
    print("   Setting up data generation model...")

    start_time = datetime.datetime(2024, 1, 15, 0, 0, 0, tzinfo=tz.UTC)
    end_time = start_time + datetime.timedelta(seconds=n_timesteps * step_size)

    weather = create_weather_data(n_timesteps, step_size, start_time)

    temp_dir = tempfile.mkdtemp()
    file_paths = {}
    for name, df in weather.items():
        path = os.path.join(temp_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        file_paths[name] = path

    model = tb.Model(id="data_generation_model")

    # Building space
    building_space = tb.BuildingSpaceThermalTorchSystem(
        C_air=5e5,
        C_wall=2e6,
        R_out=0.01,
        R_in=0.01,
        f_wall=0.4,
        f_air=0.2,
        Q_occ_gain=80.0,
        id="room",
    )

    valve = tb.ValveTorchSystem(
        waterFlowRateMax=0.05, valveAuthority=0.5, id="heating_valve"
    )

    space_heater = tb.SpaceHeaterTorchSystem(
        Q_flow_nominal_sh=2000.0,
        T_a_nominal_sh=55.0,
        T_b_nominal_sh=45.0,
        TAir_nominal_sh=20.0,
        id="radiator",
    )

    pi_controller = tb.PIDControllerSystem(
        kp=kp, Ti=Ti, Td=0.0, isReverse=True, id="true_controller"
    )

    # Schedules
    outdoor_temp_schedule = tb.ScheduleSystem(
        filename=file_paths["outdoor_temp"], id="outdoor_temp"
    )
    solar_schedule = tb.ScheduleSystem(filename=file_paths["solar"], id="solar")
    occupancy_schedule = tb.ScheduleSystem(
        filename=file_paths["occupancy"], id="occupancy"
    )
    setpoint_schedule = tb.ScheduleSystem(
        filename=file_paths["setpoint"], id="setpoint"
    )
    supply_flow_schedule = tb.ScheduleSystem(
        filename=file_paths["supply_flow"], id="supply_flow"
    )
    exhaust_flow_schedule = tb.ScheduleSystem(
        filename=file_paths["exhaust_flow"], id="exhaust_flow"
    )
    supply_temp_schedule = tb.ScheduleSystem(
        filename=file_paths["supply_temp"], id="supply_temp"
    )

    supply_water_temp_data = pd.DataFrame(
        {
            "datetime": weather["outdoor_temp"]["datetime"],
            "value": np.ones(n_timesteps) * 55.0,
        }
    )
    supply_water_temp_file = os.path.join(temp_dir, "supply_water_temp.csv")
    supply_water_temp_data.to_csv(supply_water_temp_file, index=False)
    supply_water_temp_schedule = tb.ScheduleSystem(
        filename=supply_water_temp_file, id="supply_water_temp"
    )

    # Add components
    for comp in [
        building_space,
        valve,
        space_heater,
        pi_controller,
        outdoor_temp_schedule,
        solar_schedule,
        occupancy_schedule,
        setpoint_schedule,
        supply_flow_schedule,
        exhaust_flow_schedule,
        supply_temp_schedule,
        supply_water_temp_schedule,
    ]:
        model.add_component(comp)

    # Connections
    model.add_connection(
        outdoor_temp_schedule, building_space, "scheduleValue", "outdoorTemperature"
    )
    model.add_connection(
        solar_schedule, building_space, "scheduleValue", "globalIrradiation"
    )
    model.add_connection(
        occupancy_schedule, building_space, "scheduleValue", "numberOfPeople"
    )
    model.add_connection(
        supply_flow_schedule, building_space, "scheduleValue", "supplyAirFlowRate"
    )
    model.add_connection(
        exhaust_flow_schedule, building_space, "scheduleValue", "exhaustAirFlowRate"
    )
    model.add_connection(
        supply_temp_schedule, building_space, "scheduleValue", "supplyAirTemperature"
    )
    model.add_connection(
        building_space, pi_controller, "indoorTemperature", "actualValue"
    )
    model.add_connection(
        setpoint_schedule, pi_controller, "scheduleValue", "setpointValue"
    )
    model.add_connection(pi_controller, valve, "inputSignal", "valvePosition")
    model.add_connection(valve, space_heater, "waterFlowRate", "waterFlowRate")
    model.add_connection(
        supply_water_temp_schedule,
        space_heater,
        "scheduleValue",
        "supplyWaterTemperature",
    )
    model.add_connection(
        building_space, space_heater, "indoorTemperature", "indoorTemperature"
    )
    model.add_connection(space_heater, building_space, "Power", "heatGain")

    model.load(draw_semantic_model=False, draw_simulation_model=False)

    print("   Running closed-loop simulation...")
    simulator = tb.Simulator(model)
    simulator.simulate(start_time=start_time, end_time=end_time, step_size=step_size)

    # Extract results
    timestamps = [
        start_time + datetime.timedelta(seconds=i * step_size)
        for i in range(n_timesteps)
    ]
    temperature = (
        building_space.output["indoorTemperature"]
        .history(i_s=0, i_c=0)
        .detach()
        .numpy()
    )
    actuator = (
        pi_controller.output["inputSignal"].history(i_s=0, i_c=0).detach().numpy()
    )
    setpoint = weather["setpoint"]["value"].values

    df_temperature = pd.DataFrame({"datetime": timestamps, "value": temperature})
    df_setpoint = pd.DataFrame({"datetime": timestamps, "value": setpoint})
    df_actuator = pd.DataFrame({"datetime": timestamps, "value": actuator})

    # Cleanup
    # Standard library imports
    import shutil

    shutil.rmtree(temp_dir)

    print(f"   Generated {n_timesteps} timesteps of data")
    print(
        f"   Temperature range: {temperature.min():.1f}°C to {temperature.max():.1f}°C"
    )
    print(f"   Actuator range: {actuator.min():.2f} to {actuator.max():.2f}")

    return (
        df_temperature,
        df_setpoint,
        df_actuator,
        weather,
        start_time,
        end_time,
        {"kp": kp, "Ti": Ti, "ki": kp / Ti},
    )


def create_decoy_signals(df_temperature, df_setpoint, weather, n_timesteps):
    """
    Create decoy sensor and setpoint signals that the optimizer should learn to ignore.

    Sensors:
        0: Indoor temperature (TRUE - should get beta_0 ≈ 1)
        1: Outdoor temperature (DECOY - should get beta_1 ≈ 0)
        2: Noisy/random temperature (DECOY - should get beta_2 ≈ 0)

    Setpoints:
        0: Real setpoint (TRUE - should get gamma_0 ≈ 1)
        1: Constant 22°C (DECOY - should get gamma_1 ≈ 0)
        2: Inverted/opposite setpoint (DECOY - should get gamma_2 ≈ 0)
    """
    timestamps = df_temperature["datetime"].values

    # === SENSORS ===
    # Sensor 0: True indoor temperature
    sensor_0 = df_temperature.copy()
    sensor_0_name = "indoor_temp_TRUE"

    # Sensor 1: Outdoor temperature (decoy - not what controller uses)
    sensor_1 = weather["outdoor_temp"].copy()
    sensor_1_name = "outdoor_temp_DECOY"

    # Sensor 2: Noisy random temperature (decoy - random noise)
    noisy_temp = 20.0 + np.random.normal(0, 3.0, n_timesteps)  # Random around 20°C
    sensor_2 = pd.DataFrame({"datetime": timestamps, "value": noisy_temp})
    sensor_2_name = "random_temp_DECOY"

    sensors = [
        (sensor_0, sensor_0_name),
        (sensor_1, sensor_1_name),
        (sensor_2, sensor_2_name),
    ]

    # === SETPOINTS ===
    # Setpoint 0: True setpoint
    setpoint_0 = df_setpoint.copy()
    setpoint_0_name = "real_setpoint_TRUE"

    # Setpoint 1: Constant 22°C (decoy)
    constant_sp = np.ones(n_timesteps) * 22.0
    setpoint_1 = pd.DataFrame({"datetime": timestamps, "value": constant_sp})
    setpoint_1_name = "constant_22C_DECOY"

    # Setpoint 2: Inverted setpoint (high when real is low, vice versa)
    real_sp = df_setpoint["value"].values
    inverted_sp = 40.0 - real_sp  # Inverts around 20°C
    setpoint_2 = pd.DataFrame({"datetime": timestamps, "value": inverted_sp})
    setpoint_2_name = "inverted_sp_DECOY"

    setpoints = [
        (setpoint_0, setpoint_0_name),
        (setpoint_1, setpoint_1_name),
        (setpoint_2, setpoint_2_name),
    ]

    return sensors, setpoints


def run_multi_signal_identification_example():
    """
    Run controller identification with multiple sensors and setpoints.

    The optimizer must learn to select the correct sensor (beta weights)
    and correct setpoint (gamma weights) from among decoy signals.
    """
    print("=" * 70)
    print("Controller Identification with Multiple Signals (Decoy Detection)")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)

    # =========================================================================
    # Phase 1: Generate Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Data Generation with Known Controller")
    print("=" * 70)

    step_size = 60
    n_timesteps = 1440
    true_kp = 0.1
    true_Ti = 10

    print(f"\n   True controller: PI with Kp={true_kp}, Ti={true_Ti}s")
    print(f"   (Equivalent Ki = Kp/Ti = {true_kp/true_Ti:.6f})")

    (
        df_temperature,
        df_setpoint,
        df_actuator,
        weather,
        start_time,
        end_time,
        true_params,
    ) = generate_data_with_twin4build(
        n_timesteps=n_timesteps,
        step_size=step_size,
        kp=true_kp,
        Ti=true_Ti,
    )

    # =========================================================================
    # Create Decoy Signals
    # =========================================================================
    print("\n2. Creating decoy signals...")
    sensors, setpoints = create_decoy_signals(
        df_temperature, df_setpoint, weather, n_timesteps
    )

    print("\n   SENSORS:")
    for i, (df, name) in enumerate(sensors):
        print(
            f"     [{i}] {name}: range {df['value'].min():.1f} to {df['value'].max():.1f}"
        )

    print("\n   SETPOINTS:")
    for i, (df, name) in enumerate(setpoints):
        print(
            f"     [{i}] {name}: range {df['value'].min():.1f} to {df['value'].max():.1f}"
        )

    print("\n   Expected result:")
    print("     - beta_0 ≈ 1.0 (indoor temp), beta_1 ≈ 0, beta_2 ≈ 0")
    print("     - gamma_0 ≈ 1.0 (real setpoint), gamma_1 ≈ 0, gamma_2 ≈ 0")

    # =========================================================================
    # Plot Signals
    # =========================================================================
    print("\n3. Plotting all signals...")

    time_hours = np.arange(n_timesteps) * step_size / 3600

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot sensors
    ax1 = axes[0]
    colors = ["blue", "green", "orange"]
    for i, (df, name) in enumerate(sensors):
        style = "-" if "TRUE" in name else "--"
        alpha = 1.0 if "TRUE" in name else 0.6
        ax1.plot(
            time_hours,
            df["value"].values,
            style,
            color=colors[i],
            alpha=alpha,
            linewidth=1,
            label=name,
        )
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(loc="upper right")
    ax1.set_title("Sensor Signals (Identification must select TRUE sensor)")
    ax1.grid(True, alpha=0.3)

    # Plot setpoints
    ax2 = axes[1]
    colors = ["red", "purple", "brown"]
    for i, (df, name) in enumerate(setpoints):
        style = "-" if "TRUE" in name else "--"
        alpha = 1.0 if "TRUE" in name else 0.6
        ax2.plot(
            time_hours,
            df["value"].values,
            style,
            color=colors[i],
            alpha=alpha,
            linewidth=1,
            label=name,
        )
    ax2.set_ylabel("Setpoint (°C)")
    ax2.legend(loc="upper right")
    ax2.set_title("Setpoint Signals (Identification must select TRUE setpoint)")
    ax2.grid(True, alpha=0.3)

    # Plot actuator
    ax3 = axes[2]
    ax3.plot(time_hours, df_actuator["value"].values, "m-", linewidth=1)
    ax3.set_ylabel("Actuator Signal")
    ax3.set_xlabel("Time (hours)")
    ax3.set_title("True Actuator Output (Target for Identification)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("twin4build/examples/multi_signal_data.png", dpi=150)
    plt.show()

    # =========================================================================
    # Save to CSV files
    # =========================================================================
    temp_dir = tempfile.mkdtemp()

    sensor_files = []
    for i, (df, name) in enumerate(sensors):
        path = os.path.join(temp_dir, f"sensor_{i}.csv")
        df.to_csv(path, index=False)
        sensor_files.append(path)

    setpoint_files = []
    for i, (df, name) in enumerate(setpoints):
        path = os.path.join(temp_dir, f"setpoint_{i}.csv")
        df.to_csv(path, index=False)
        setpoint_files.append(path)

    act_file = os.path.join(temp_dir, "actuator.csv")
    df_actuator.to_csv(act_file, index=False)

    # =========================================================================
    # Phase 2: Controller Identification
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Controller Identification with Multiple Signals")
    print("=" * 70)

    print("\n4. Creating identification model...")

    model = tb.Model(id="multi_signal_identification_model")

    # Create ScheduleSystems for all sensors
    sensor_schedules = []
    for i, path in enumerate(sensor_files):
        sched = tb.ScheduleSystem(filename=path, id=f"sensor_{i}_{sensors[i][1]}")
        sensor_schedules.append(sched)
        model.add_component(sched)

    # Create ScheduleSystems for all setpoints
    setpoint_schedules = []
    for i, path in enumerate(setpoint_files):
        sched = tb.ScheduleSystem(filename=path, id=f"setpoint_{i}_{setpoints[i][1]}")
        setpoint_schedules.append(sched)
        model.add_component(sched)

    # Create controller with multiple sensors and setpoints
    controller = tb.ControllerIdentificationTorchSystem(
        n_sensors=len(sensors),
        n_setpoints=len(setpoints),
        n_actuators=1,
        isReverse=True,
        id="identified_controller",
    )
    model.add_component(controller)

    # Actuator sensor
    actuator_sensor = tb.SensorSystem(filename=act_file, id="actuator_sensor")
    model.add_component(actuator_sensor)

    # Connect all sensors
    for i, sched in enumerate(sensor_schedules):
        model.add_connection(
            sched, controller, "scheduleValue", "sensorValue", input_port_index=i
        )

    # Connect all setpoints
    for i, sched in enumerate(setpoint_schedules):
        model.add_connection(
            sched, controller, "scheduleValue", "setpointValue", input_port_index=i
        )

    # Connect output
    model.add_connection(
        controller, actuator_sensor, "inputSignal", "measuredValue", output_port_index=0
    )

    model.load(draw_semantic_model=False, draw_simulation_model=True)

    print(model)

    # =========================================================================
    # Setup Estimation
    # =========================================================================
    print("\n5. Setting up estimation...")

    simulator = tb.Simulator(model)
    parameters = controller.get_estimator_parameters()

    print(f"   Number of parameters to estimate: {len(parameters)}")
    print("   Parameters:")
    for p in parameters:
        print(f"     - {p[0].id}.{p[1]} x0:{p[2]} lb:{p[3]} ub:{p[4]}")

    measurements = [(actuator_sensor, 0.02)]

    # Initial simulation
    print("\n6. Running initial simulation...")
    simulator.simulate(start_time=start_time, end_time=end_time, step_size=step_size)

    initial_predictions = (
        actuator_sensor.input["measuredValue"].history(i_s=0, i_c=0).detach().numpy()
    )
    actual_actuator = df_actuator["value"].values

    print(f"\n   Initial weights:")
    beta_vals_init = [
        f"{controller._get_beta(i).item():.3f}" for i in range(len(sensors))
    ]
    gamma_vals_init = [
        f"{controller._get_gamma(i).item():.3f}" for i in range(len(setpoints))
    ]
    print(f"     beta (sensors):   {beta_vals_init}")
    print(f"     gamma (setpoints): {gamma_vals_init}")

    # =========================================================================
    # Run Estimation with Binarization Penalty
    # =========================================================================
    print("\n7. Running parameter estimation...")
    print("   Using regularization_lambda=0.01 for binarization penalty P(x) = x(1-x)")

    estimator = tb.Estimator(simulator)

    options = {
        # "ftol": 1e-10,
        "maxiter": 1000,
        "disp": True,
    }

    result = estimator.estimate(
        start_time=start_time,
        end_time=end_time,
        step_size=step_size,
        parameters=parameters,
        measurements=measurements,
        n_warmup=10,
        method=("scipy", "SLSQP", "ad"),
        regularization_lambda=0,  # Binarization penalty to push weights toward 0 or 1
        options=options,
    )

    # =========================================================================
    # Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n" + controller.summary())

    # Check selection weights
    print("\n   SIGNAL SELECTION RESULTS:")
    print("\n   Beta weights (sensor selection):")
    for i, (df, name) in enumerate(sensors):
        beta = controller._get_beta(i).item()
        expected = "≈1.0" if "TRUE" in name else "≈0.0"
        status = (
            "✓"
            if (("TRUE" in name and beta > 0.5) or ("DECOY" in name and beta < 0.5))
            else "✗"
        )
        print(f"     beta_{i} = {beta:.4f} (expected {expected}) [{name}] {status}")

    print("\n   Gamma weights (setpoint selection):")
    for i, (df, name) in enumerate(setpoints):
        gamma = controller._get_gamma(i).item()
        expected = "≈1.0" if "TRUE" in name else "≈0.0"
        status = (
            "✓"
            if (("TRUE" in name and gamma > 0.5) or ("DECOY" in name and gamma < 0.5))
            else "✗"
        )
        print(f"     gamma_{i} = {gamma:.4f} (expected {expected}) [{name}] {status}")

    print("\n   Controller parameters:")
    print(
        f"     kp = {controller.candidate_0_0.kp.get().item():.6f} (true: {true_params['kp']})"
    )
    print(
        f"     Ti = {controller.candidate_0_0.Ti.get().item():.6f} (true: {true_params['Ti']})"
    )

    # =========================================================================
    # Plot Results
    # =========================================================================
    print("\n8. Generating result plots...")

    final_predictions = (
        actuator_sensor.input["measuredValue"].history(i_s=0, i_c=0).detach().numpy()
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Actuator comparison
    ax1 = axes[0]
    ax1.plot(time_hours, actual_actuator, "g-", alpha=0.7, linewidth=1, label="Actual")
    ax1.plot(
        time_hours, initial_predictions, "b--", alpha=0.5, linewidth=1, label="Initial"
    )
    ax1.plot(time_hours, final_predictions, "m-", linewidth=1, label="Identified")
    ax1.set_ylabel("Actuator Signal")
    ax1.legend(loc="upper right")
    ax1.set_title("Controller Identification with Multiple Signals")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Selection weights
    ax2 = axes[1]
    beta_vals = [controller._get_beta(i).item() for i in range(len(sensors))]
    gamma_vals = [controller._get_gamma(i).item() for i in range(len(setpoints))]

    x_beta = np.arange(len(sensors))
    x_gamma = np.arange(len(setpoints)) + len(sensors) + 1

    bars1 = ax2.bar(x_beta, beta_vals, color="blue", alpha=0.7, label="Beta (sensors)")
    bars2 = ax2.bar(
        x_gamma, gamma_vals, color="red", alpha=0.7, label="Gamma (setpoints)"
    )

    # Add labels
    sensor_labels = [f"β{i}\n{s[1].split('_')[0]}" for i, s in enumerate(sensors)]
    setpoint_labels = [f"γ{i}\n{s[1].split('_')[0]}" for i, s in enumerate(setpoints)]
    ax2.set_xticks(list(x_beta) + list(x_gamma))
    ax2.set_xticklabels(sensor_labels + setpoint_labels, fontsize=8)
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold")
    ax2.set_ylabel("Weight Value")
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc="upper right")
    ax2.set_title(
        "Signal Selection Weights (TRUE signals should be ≈1.0, DECOY signals should be ≈0.0)"
    )
    ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Error
    ax3 = axes[2]
    initial_error = np.abs(actual_actuator - initial_predictions)
    final_error = np.abs(actual_actuator - final_predictions)
    ax3.plot(
        time_hours,
        initial_error,
        "b--",
        alpha=0.5,
        label=f"Initial MAE: {initial_error.mean():.4f}",
    )
    ax3.plot(
        time_hours, final_error, "m-", label=f"Final MAE: {final_error.mean():.4f}"
    )
    ax3.set_ylabel("Absolute Error")
    ax3.set_xlabel("Time (hours)")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("twin4build/examples/multi_signal_identification_results.png", dpi=150)
    plt.show()

    # Cleanup
    # Standard library imports
    import shutil

    shutil.rmtree(temp_dir)

    return controller, result


if __name__ == "__main__":
    controller, result = run_multi_signal_identification_example()

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
