"""
Controller Identification Example - Complex Case with Multiple Actuators

This example stress-tests the controller identification system with:
- 7 sensors (2 true indoor temps, 5 decoys)
- 6 setpoints (1 true, 5 decoys)
- 2 actuators (one PI-controlled, one On-Off-controlled)
- 2 controller candidates: PID vs On-Off

The optimizer must learn:
- Which sensor provides feedback (beta weights)
- Which setpoint is being tracked (gamma weights)
- Which controller candidate is active for each actuator (alpha weights)
- The parameters of the active controllers

True controllers:
- Actuator 0: PI with Kp=0.15, Ti=8.0 → PID candidate learns Td=0
- Actuator 1: On-Off with offValue=0, onValue=1, steepness=10
"""

# Standard library imports
import cProfile
import datetime
import io
import os
import pstats
import tempfile
import shutil

# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from dateutil import tz

# Local application imports
import twin4build as tb
from twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system import (
    PIDControllerSystem,
)
from twin4build.systems.controller.rulebased_controller.on_off_controller.on_off_controller_torch_system import (
    OnOffControllerTorchSystem,
)


def create_weather_data(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    step_size: int,
) -> dict:
    """Create realistic weather data for the simulation."""
    n_timesteps = int((end_time - start_time).total_seconds() / step_size)
    timestamps = [start_time + datetime.timedelta(seconds=i * step_size) 
                  for i in range(n_timesteps)]
    
    hours = np.array([ts.hour + ts.minute / 60 for ts in timestamps])
    
    # Outdoor temperature: cold winter day with diurnal cycle
    outdoor_temp = -3.0 + 5.0 * np.sin(2 * np.pi * (hours - 14) / 24)
    outdoor_temp += np.random.normal(0, 0.5, n_timesteps)
    
    # Solar radiation with realistic daily pattern
    solar = np.maximum(0, 400 * np.sin(np.pi * (hours - 6) / 12))
    solar = np.where((hours >= 6) & (hours <= 18), solar, 0)
    solar += np.random.normal(0, 20, n_timesteps)
    solar = np.maximum(0, solar)
    
    # Occupancy pattern
    occupancy = np.where((hours >= 8) & (hours <= 17), 
                         np.random.randint(1, 5, n_timesteps), 0).astype(float)
    
    # Setpoint: varies through the day (heating schedule)
    setpoint = np.ones(n_timesteps) * 20.0
    setpoint = np.where((hours >= 6) & (hours < 8), 21.0, setpoint)
    setpoint = np.where((hours >= 8) & (hours < 18), 21.5, setpoint)
    setpoint = np.where((hours >= 18) & (hours < 22), 20.5, setpoint)
    setpoint = np.where((hours >= 22) | (hours < 6), 18.0, setpoint)
    
    data = {
        'outdoor_temp': pd.DataFrame({'datetime': timestamps, 'value': outdoor_temp}),
        'solar': pd.DataFrame({'datetime': timestamps, 'value': solar}),
        'occupancy': pd.DataFrame({'datetime': timestamps, 'value': occupancy}),
        'setpoint': pd.DataFrame({'datetime': timestamps, 'value': setpoint}),
        'supply_flow': pd.DataFrame({'datetime': timestamps, 'value': np.zeros(n_timesteps)}),
        'exhaust_flow': pd.DataFrame({'datetime': timestamps, 'value': np.zeros(n_timesteps)}),
        'supply_temp': pd.DataFrame({'datetime': timestamps, 'value': np.ones(n_timesteps) * 35.0}),
    }
    
    return data


def generate_single_controller_data(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    step_size: int,
    weather: dict,
    controller_type: str,  # "pi" or "onoff"
    pi_kp: float = 0.15,
    pi_Ti: float = 8.0,
    onoff_steepness: float = 10.0,
):
    """
    Generate data for a single controller type.
    
    Args:
        controller_type: "pi" or "onoff"
    """
    
    temp_dir = tempfile.mkdtemp()
    file_paths = {}
    for name, df in weather.items():
        path = os.path.join(temp_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        file_paths[name] = path
    
    model = tb.Model(id=f"{controller_type}_model")
    
    # Building space with thermal dynamics
    building_space = tb.BuildingSpaceThermalTorchSystem(
        C_air=3e6, C_wall=2e6, C_int=1e5,
        R_out=0.01, R_in=0.01, R_int=0.02,
        f_wall=0.4, f_air=0.2, Q_occ_gain=80.0,
        id="room"
    )
    
    valve = tb.ValveTorchSystem(waterFlowRateMax=0.05, valveAuthority=0.5, id="valve")
    heater = tb.SpaceHeaterTorchSystem(
        Q_flow_nominal_sh=2000.0, T_a_nominal_sh=55.0,
        T_b_nominal_sh=45.0, TAir_nominal_sh=20.0, id="heater"
    )
    
    # Create controller based on type
    if controller_type == "pi":
        controller = tb.PIDControllerSystem(
            kp=pi_kp, Ti=pi_Ti, Td=0.0, isReverse=True, id="controller"
        )
    else:  # onoff
        controller = OnOffControllerTorchSystem(
            offValue=0.0, onValue=1.0, steepness=onoff_steepness, isReverse=True, id="controller"
        )
    
    # Schedules
    outdoor_temp_schedule = tb.ScheduleSystem(filename=file_paths['outdoor_temp'], id="outdoor_temp")
    solar_schedule = tb.ScheduleSystem(filename=file_paths['solar'], id="solar")
    occupancy_schedule = tb.ScheduleSystem(filename=file_paths['occupancy'], id="occupancy")
    setpoint_schedule = tb.ScheduleSystem(filename=file_paths['setpoint'], id="setpoint")
    supply_flow_schedule = tb.ScheduleSystem(filename=file_paths['supply_flow'], id="supply_flow")
    exhaust_flow_schedule = tb.ScheduleSystem(filename=file_paths['exhaust_flow'], id="exhaust_flow")
    supply_temp_schedule = tb.ScheduleSystem(filename=file_paths['supply_temp'], id="supply_temp")
    
    supply_water_temp_data = pd.DataFrame({
        'datetime': weather['outdoor_temp']['datetime'],
        'value': np.ones(len(weather['outdoor_temp'])) * 55.0
    })
    supply_water_temp_file = os.path.join(temp_dir, "supply_water_temp.csv")
    supply_water_temp_data.to_csv(supply_water_temp_file, index=False)
    supply_water_temp_schedule = tb.ScheduleSystem(filename=supply_water_temp_file, id="supply_water_temp")
    
    # Add components
    for comp in [building_space, valve, heater, controller,
                 outdoor_temp_schedule, solar_schedule, occupancy_schedule,
                 setpoint_schedule, supply_flow_schedule, exhaust_flow_schedule,
                 supply_temp_schedule, supply_water_temp_schedule]:
        model.add_component(comp)
    
    # Connections
    model.add_connection(outdoor_temp_schedule, building_space, "scheduleValue", "outdoorTemperature")
    model.add_connection(solar_schedule, building_space, "scheduleValue", "globalIrradiation")
    model.add_connection(occupancy_schedule, building_space, "scheduleValue", "numberOfPeople")
    model.add_connection(supply_flow_schedule, building_space, "scheduleValue", "supplyAirFlowRate")
    model.add_connection(exhaust_flow_schedule, building_space, "scheduleValue", "exhaustAirFlowRate")
    model.add_connection(supply_temp_schedule, building_space, "scheduleValue", "supplyAirTemperature")
    model.add_connection(building_space, controller, "indoorTemperature", "actualValue")
    model.add_connection(setpoint_schedule, controller, "scheduleValue", "setpointValue")
    model.add_connection(controller, valve, "inputSignal", "valvePosition")
    model.add_connection(valve, heater, "waterFlowRate", "waterFlowRate")
    model.add_connection(supply_water_temp_schedule, heater, "scheduleValue", "supplyWaterTemperature")
    model.add_connection(building_space, heater, "indoorTemperature", "indoorTemperature")
    model.add_connection(heater, building_space, "Power", "heatGain")
    
    model.load(draw_semantic_model=False, draw_simulation_model=False, verbose=0)
    
    simulator = tb.Simulator(model)
    simulator.simulate(start_time=start_time, end_time=end_time, step_size=step_size)
    
    # Extract results using ACTUAL simulation timesteps
    timestamps = simulator.date_time_steps[0]
    temperature = building_space.output["indoorTemperature"].history(i_s=0, i_c=0).detach().numpy()
    actuator = controller.output["inputSignal"].history(i_s=0, i_c=0).detach().numpy()
    
    df_temperature = pd.DataFrame({'datetime': timestamps, 'value': temperature})
    df_actuator = pd.DataFrame({'datetime': timestamps, 'value': actuator})
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    return df_temperature, df_actuator


def generate_dual_actuator_data(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    step_size: int = 60,
    # PI controller parameters (actuator 0)
    pi_kp: float = 0.15,
    pi_Ti: float = 8.0,
    # On-Off controller parameters (actuator 1)
    onoff_steepness: float = 10.0,
):
    """
    Generate synthetic data for TWO actuators using twin4build components.
    
    Runs two separate simulations:
    - Simulation 1: PI controller → Actuator 0
    - Simulation 2: On-Off controller → Actuator 1
    
    Both simulations use identical weather/setpoint conditions.
    """
    print("   Setting up dual-actuator data generation...")
    
    # Create shared weather data
    weather = create_weather_data(start_time, end_time, step_size)
    
    # Generate PI controller data
    print("   Running PI controller simulation...")
    df_temp_pi, df_actuator_pi = generate_single_controller_data(
        start_time=start_time,
        end_time=end_time,
        step_size=step_size,
        weather=weather,
        controller_type="pi",
        pi_kp=pi_kp,
        pi_Ti=pi_Ti,
    )
    
    # Generate On-Off controller data
    print("   Running On-Off controller simulation...")
    df_temp_onoff, df_actuator_onoff = generate_single_controller_data(
        start_time=start_time,
        end_time=end_time,
        step_size=step_size,
        weather=weather,
        controller_type="onoff",
        onoff_steepness=onoff_steepness,
    )
    
    # Return both temperatures - each actuator was seeing its own temperature
    df_setpoint = pd.DataFrame({
        'datetime': df_temp_pi['datetime'],
        'value': weather['setpoint']['value'].values
    })
    
    temp_pi = df_temp_pi['value'].values
    temp_onoff = df_temp_onoff['value'].values
    actuator_pi = df_actuator_pi['value'].values
    actuator_onoff = df_actuator_onoff['value'].values
    
    print(f"   Generated {len(df_temp_pi)} timesteps of data")
    print(f"   PI Temperature range: {temp_pi.min():.1f}°C to {temp_pi.max():.1f}°C")
    print(f"   On-Off Temperature range: {temp_onoff.min():.1f}°C to {temp_onoff.max():.1f}°C")
    print(f"   PI Actuator range: {actuator_pi.min():.2f} to {actuator_pi.max():.2f}")
    print(f"   On-Off Actuator range: {actuator_onoff.min():.2f} to {actuator_onoff.max():.2f}")
    
    true_params = {
        'pi': {'kp': pi_kp, 'Ti': pi_Ti},
        'onoff': {'offValue': 0.0, 'onValue': 1.0, 'steepness': onoff_steepness}
    }
    
    return df_temp_pi, df_temp_onoff, df_setpoint, df_actuator_pi, df_actuator_onoff, weather, true_params


def create_complex_decoy_signals(df_temp_pi, df_temp_onoff, df_setpoint, weather):
    """
    Create complex decoy sensor and setpoint signals.
    
    SENSORS (7 total):
        0: Indoor temperature PI (TRUE for Actuator 0)
        1: Indoor temperature On-Off (TRUE for Actuator 1)
        2: Outdoor temperature (DECOY) - wrong source
        3: Noisy random temperature (DECOY) - noise
        4: Lagged indoor temperature (DECOY) - delayed version
        5: Supply water temperature (DECOY) - different system
        6: Scaled/offset indoor temp (DECOY) - wrong scale
    
    SETPOINTS (6 total):
        0: Real setpoint (TRUE)
        1: Constant 22°C (DECOY) - no variation
        2: Inverted setpoint (DECOY) - opposite pattern
        3: Delayed setpoint (DECOY) - lagged version
        4: Outdoor-following (DECOY) - follows outdoor
        5: Random walk setpoint (DECOY) - random pattern
    """
    # Keep timezone info - don't use .values which strips timezone
    timestamps = df_temp_pi['datetime'].tolist()  # Preserves timezone-aware datetime objects
    n_timesteps = len(timestamps)
    indoor_temp_pi = df_temp_pi['value'].values
    indoor_temp_onoff = df_temp_onoff['value'].values
    outdoor_temp = weather['outdoor_temp']['value'].values
    real_setpoint = df_setpoint['value'].values
    
    # === SENSORS (7) ===
    sensors = []
    
    # Sensor 0: TRUE for Actuator 0 - Indoor temperature from PI simulation
    sensors.append((
        pd.DataFrame({'datetime': timestamps, 'value': indoor_temp_pi}),
        "indoor_temp_PI_TRUE"
    ))
    
    # Sensor 1: TRUE for Actuator 1 - Indoor temperature from On-Off simulation
    sensors.append((
        pd.DataFrame({'datetime': timestamps, 'value': indoor_temp_onoff}),
        "indoor_temp_ONOFF_TRUE"
    ))
    
    # Sensor 2: DECOY - Outdoor temperature
    sensors.append((
        weather['outdoor_temp'].copy(),
        "outdoor_temp_DECOY"
    ))
    
    # Sensor 3: DECOY - Random noise around 20°C
    noisy_temp = 20.0 + np.random.normal(0, 2.5, n_timesteps)
    sensors.append((
        pd.DataFrame({'datetime': timestamps, 'value': noisy_temp}),
        "random_noise_DECOY"
    ))
    
    # Sensor 4: DECOY - Lagged indoor temperature (30 min delay)
    lag_steps = 30  # 30 minutes at 1-min resolution
    lagged_temp = np.roll(indoor_temp_pi, lag_steps)
    lagged_temp[:lag_steps] = indoor_temp_pi[0]  # Fill initial values
    sensors.append((
        pd.DataFrame({'datetime': timestamps, 'value': lagged_temp}),
        "lagged_indoor_DECOY"
    ))
    
    # Sensor 5: DECOY - Supply water temperature (constant-ish)
    supply_water_temp = 55.0 + np.random.normal(0, 1.0, n_timesteps)
    sensors.append((
        pd.DataFrame({'datetime': timestamps, 'value': supply_water_temp}),
        "supply_water_DECOY"
    ))
    
    # Sensor 6: DECOY - Scaled/offset indoor temp (wrong calibration)
    scaled_temp = (indoor_temp_pi - 20.0) * 0.5 + 25.0  # Different scale and offset
    sensors.append((
        pd.DataFrame({'datetime': timestamps, 'value': scaled_temp}),
        "miscalibrated_DECOY"
    ))
    
    # === SETPOINTS (6) ===
    setpoints = []
    
    # Setpoint 0: TRUE - Real setpoint
    setpoints.append((
        pd.DataFrame({'datetime': timestamps, 'value': real_setpoint}),
        "real_setpoint_TRUE"
    ))
    
    # Setpoint 1: DECOY - Constant 22°C
    constant_sp = np.ones(n_timesteps) * 22.0
    setpoints.append((
        pd.DataFrame({'datetime': timestamps, 'value': constant_sp}),
        "constant_22C_DECOY"
    ))
    
    # Setpoint 2: DECOY - Inverted setpoint (high when real is low)
    inverted_sp = 40.0 - real_setpoint
    setpoints.append((
        pd.DataFrame({'datetime': timestamps, 'value': inverted_sp}),
        "inverted_DECOY"
    ))
    
    # Setpoint 3: DECOY - Delayed setpoint (1 hour lag)
    lag_steps = 60
    delayed_sp = np.roll(real_setpoint, lag_steps)
    delayed_sp[:lag_steps] = real_setpoint[0]
    setpoints.append((
        pd.DataFrame({'datetime': timestamps, 'value': delayed_sp}),
        "delayed_DECOY"
    ))
    
    # Setpoint 4: DECOY - Outdoor-following setpoint
    outdoor_following = 20.0 + outdoor_temp * 0.2  # Follows outdoor pattern
    setpoints.append((
        pd.DataFrame({'datetime': timestamps, 'value': outdoor_following}),
        "outdoor_follow_DECOY"
    ))
    
    # Setpoint 5: DECOY - Random walk setpoint
    random_walk = np.cumsum(np.random.normal(0, 0.05, n_timesteps))
    random_walk = 21.0 + random_walk - random_walk.mean()  # Center around 21
    random_walk = np.clip(random_walk, 18, 24)  # Keep in reasonable range
    setpoints.append((
        pd.DataFrame({'datetime': timestamps, 'value': random_walk}),
        "random_walk_DECOY"
    ))
    
    return sensors, setpoints


def create_controller_candidates():
    """
    Define controller candidates: PID vs On-Off.
    
    CANDIDATES (2 total):
        0: PID controller - general form (P, PI, PD, or PID depending on parameters)
        1: On-Off controller - bang-bang control
    
    The optimizer should learn:
        - Actuator 0 (true=PI): Select PID (α₀,₀ ≈ 1) with Td → 0
        - Actuator 1 (true=On-Off): Select On-Off (α₁,₁ ≈ 1)
    
    Returns:
        Tuple of (controller_classes, controller_kwargs)
    """
    controller_classes = [
        PIDControllerSystem,         # Candidate 0: PID (true for actuator 0, will learn Td=0)
        OnOffControllerTorchSystem,  # Candidate 1: On-Off (true for actuator 1)
    ]
    
    controller_kwargs = [
        # Candidate 0: PID - will learn Kp, Ti, Td (Td should go to 0 for PI behavior)
        {"kp": 1e-3, "Ti": 10.0, "Td": 0.0, "isReverse": True},
        
        # Candidate 1: On-Off controller (differentiable version)
        {"offValue": 0.0, "onValue": 1.0, "steepness": 10.0, "isReverse": True},
    ]
    
    return controller_classes, controller_kwargs


def run_complex_identification_example():
    """
    Run controller identification with 2 actuators and complex signal selection.
    
    - Actuator 0: True controller = PI
    - Actuator 1: True controller = On-Off
    """
    print("=" * 80)
    print("COMPLEX Controller Identification Example")
    print("2 Actuators × 7 Sensors × 6 Setpoints × 2 Controller Candidates")
    print("=" * 80)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # =========================================================================
    # Phase 1: Generate Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: Data Generation with Known Controllers")
    print("=" * 80)
    
    # Configure simulation time
    start_time = datetime.datetime(2024, 1, 15, 0, 0, 0, tzinfo=tz.UTC)
    end_time = datetime.datetime(2024, 1, 17, 0, 0, 0, tzinfo=tz.UTC)  # 2 days
    step_size = 600  # 10 minutes
    
    # True controller parameters
    true_pi_kp = 1e-3
    true_pi_Ti = 8.0
    true_onoff_steepness = 100
    
    print(f"\n   Actuator 0: PI controller with Kp={true_pi_kp}, Ti={true_pi_Ti}s")
    print(f"   Actuator 1: On-Off controller with steepness={true_onoff_steepness}")
    
    (df_temp_pi, df_temp_onoff, df_setpoint, df_actuator_pi, df_actuator_onoff, 
     weather, true_params) = generate_dual_actuator_data(
        start_time=start_time,
        end_time=end_time,
        step_size=step_size,
        pi_kp=true_pi_kp,
        pi_Ti=true_pi_Ti,
        onoff_steepness=true_onoff_steepness,
    )
    
    # =========================================================================
    # Create Complex Decoy Signals
    # =========================================================================
    print("\n2. Creating complex decoy signals...")
    sensors, setpoints = create_complex_decoy_signals(
        df_temp_pi, df_temp_onoff, df_setpoint, weather
    )
    
    print("\n   SENSORS (7 candidates - 2 TRUE, 5 DECOY):")
    for i, (df, name) in enumerate(sensors):
        marker = "★" if "TRUE" in name else " "
        print(f"     {marker} [{i}] {name}: range {df['value'].min():.1f} to {df['value'].max():.1f}")
    
    print("\n   SETPOINTS (6 candidates):")
    for i, (df, name) in enumerate(setpoints):
        marker = "★" if "TRUE" in name else " "
        print(f"     {marker} [{i}] {name}: range {df['value'].min():.1f} to {df['value'].max():.1f}")
    
    # Controller candidates
    controller_classes, controller_kwargs = create_controller_candidates()
    
    print("\n   CONTROLLER CANDIDATES (2 candidates per actuator):")
    candidate_descriptions = [
        "PID (true for Act.0, will learn Td=0)",
        "On-Off (true for Act.1)",
    ]
    for i, (desc, kwargs) in enumerate(zip(candidate_descriptions, controller_kwargs)):
        ctrl_type = "On-Off" if i == 1 else "PID"
        if ctrl_type == "PID":
            print(f"     [{i}] {desc}: Kp={kwargs['kp']}, Ti={kwargs['Ti']}, Td={kwargs['Td']}")
        else:
            print(f"     [{i}] {desc}: off={kwargs['offValue']}, on={kwargs['onValue']}, k={kwargs['steepness']}")
    
    print("\n   Expected identification:")
    print("     - beta_0 + beta_1 ≈ 1.0 (both indoor temps are TRUE), others ≈ 0")
    print("     - gamma_0 ≈ 1.0 (real setpoint), others ≈ 0")
    print("     - Actuator 0: α₀,₀ ≈ 1.0 (PID with Td→0), α₀,₁ ≈ 0")
    print("     - Actuator 1: α₁,₁ ≈ 1.0 (On-Off), α₁,₀ ≈ 0")
    
    # =========================================================================
    # Plot All Signals
    # =========================================================================
    print("\n3. Plotting all signals...")
    
    n_timesteps = len(df_temp_pi)  # Derive from actual data
    time_hours = np.arange(n_timesteps) * step_size / 3600
    
    fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True)
    
    # Plot sensors
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(sensors)))
    for i, (df, name) in enumerate(sensors):
        style = '-' if 'TRUE' in name else '--'
        lw = 2.0 if 'TRUE' in name else 1.0
        alpha = 1.0 if 'TRUE' in name else 0.6
        ax1.plot(time_hours, df['value'].values, style, color=colors[i], 
                 alpha=alpha, linewidth=lw, label=f"[{i}] {name}")
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.set_title('Sensor Signals (7 candidates - 2 TRUE temps, 5 DECOY)')
    ax1.grid(True, alpha=0.3)
    
    # Plot setpoints
    ax2 = axes[1]
    colors = plt.cm.Set1(np.linspace(0, 1, len(setpoints)))
    for i, (df, name) in enumerate(setpoints):
        style = '-' if 'TRUE' in name else '--'
        lw = 2.0 if 'TRUE' in name else 1.0
        alpha = 1.0 if 'TRUE' in name else 0.6
        ax2.plot(time_hours, df['value'].values, style, color=colors[i],
                 alpha=alpha, linewidth=lw, label=f"[{i}] {name}")
    ax2.set_ylabel('Setpoint (°C)')
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.set_title('Setpoint Signals (6 candidates - must identify TRUE setpoint)')
    ax2.grid(True, alpha=0.3)
    
    # Plot PI actuator
    ax3 = axes[2]
    ax3.plot(time_hours, df_actuator_pi['value'].values, 'b-', linewidth=1.5, label='PI Controller')
    ax3.set_ylabel('Actuator 0 (PI)')
    ax3.set_title('Actuator 0: PI Controller Output')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Plot On-Off actuator
    ax4 = axes[3]
    ax4.plot(time_hours, df_actuator_onoff['value'].values, 'r-', linewidth=1.5, label='On-Off Controller')
    ax4.set_ylabel('Actuator 1 (On-Off)')
    ax4.set_title('Actuator 1: On-Off Controller Output')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # Plot indoor temp vs setpoint (both temperatures)
    ax5 = axes[4]
    ax5.plot(time_hours, sensors[0][0]['value'].values, 'b-', linewidth=1.5, label='Indoor Temp (PI sim)', alpha=0.8)
    ax5.plot(time_hours, sensors[1][0]['value'].values, 'g-', linewidth=1.5, label='Indoor Temp (On-Off sim)', alpha=0.8)
    ax5.plot(time_hours, setpoints[0][0]['value'].values, 'r--', linewidth=2, label='Setpoint')
    ax5.set_ylabel('Temperature (°C)')
    ax5.set_xlabel('Time (hours)')
    ax5.legend(loc='upper right')
    ax5.set_title('Control Performance: Both TRUE Temperatures vs Setpoint')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, "complex_dual_actuator_data.png"), dpi=150)
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
    
    # Save both actuator files
    act_pi_file = os.path.join(temp_dir, "actuator_pi.csv")
    df_actuator_pi.to_csv(act_pi_file, index=False)
    
    act_onoff_file = os.path.join(temp_dir, "actuator_onoff.csv")
    df_actuator_onoff.to_csv(act_onoff_file, index=False)
    
    # =========================================================================
    # Phase 2: Controller Identification
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: Dual-Actuator Controller Identification")
    print("=" * 80)
    
    print("\n4. Creating identification model with 2 actuators...")
    
    model = tb.Model(id="dual_actuator_identification_model")
    
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
    
    # Create controller with 2 actuators and multiple candidates
    controller = tb.ControllerIdentificationTorchSystem(
        n_sensors=len(sensors),
        n_setpoints=len(setpoints),
        n_actuators=2,  # TWO ACTUATORS
        candidate_controllers=controller_classes,
        candidate_controller_kwargs=controller_kwargs,
        id="identified_controller",
    )
    model.add_component(controller)
    
    # Actuator sensors (one per actuator)
    actuator_sensor_pi = tb.SensorSystem(filename=act_pi_file, id="actuator_sensor_pi")
    actuator_sensor_onoff = tb.SensorSystem(filename=act_onoff_file, id="actuator_sensor_onoff")
    model.add_component(actuator_sensor_pi)
    model.add_component(actuator_sensor_onoff)
    
    # Connect all sensors
    for i, sched in enumerate(sensor_schedules):
        model.add_connection(sched, controller, "scheduleValue", "sensorValue", input_port_index=i)
    
    # Connect all setpoints
    for i, sched in enumerate(setpoint_schedules):
        model.add_connection(sched, controller, "scheduleValue", "setpointValue", input_port_index=i)
    
    # Connect outputs to actuator sensors
    model.add_connection(controller, actuator_sensor_pi, "inputSignal", "measuredValue", output_port_index=0)
    model.add_connection(controller, actuator_sensor_onoff, "inputSignal", "measuredValue", output_port_index=1)
    
    model.load(draw_semantic_model=False, draw_simulation_model=True, verbose=0)
    
    print(model)
    
    # =========================================================================
    # Setup Estimation
    # =========================================================================
    print("\n5. Setting up estimation...")
    
    simulator = tb.Simulator(model)
    parameters = controller.get_estimator_parameters()
    
    # Override x0 values to be VERY CLOSE to optimal solution for debugging
    # Expected optimal (per-actuator) - from true_pi_kp, true_pi_Ti, true_onoff_steepness:
    #   - alpha_0: [1.0, 0.0] (PI for actuator 0)
    #   - beta_0: [1.0, 0, 0, 0, 0, 0, 0] (sensor 0 = PI_TRUE for actuator 0)
    #   - gamma_0: [1.0, 0, 0, 0, 0, 0] (TRUE setpoint)
    #   - alpha_1: [0.0, 1.0] (On-Off for actuator 1)
    #   - beta_1: [0, 1.0, 0, 0, 0, 0, 0] (sensor 1 = ONOFF_TRUE for actuator 1)
    #   - gamma_1: [1.0, 0, 0, 0, 0, 0] (TRUE setpoint)
    #   - PI params: kp=0.001, Ti=8.0, Td=0.0  (kp is at lower bound!)
    #   - On-Off params: offValue=0.0, onValue=1.0, steepness=100
    
    parameters_exact_optimal = []
    for comp, attr, x0, lb, ub, *_ in parameters:
        if attr == "alpha_0":
            x0 = [0.5, 0.5]  # EXACT: PI selected for actuator 0
        elif attr == "beta_0":
            x0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # EXACT: Sensor 0 for actuator 0
        elif attr == "gamma_0":
            x0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # EXACT: TRUE setpoint for actuator 0
        elif attr == "alpha_1":
            x0 = [0.5, 0.5]  # EXACT: On-Off selected for actuator 1
        elif attr == "beta_1":
            x0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # EXACT: Sensor 1 for actuator 1
        elif attr == "gamma_1":
            x0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # EXACT: TRUE setpoint for actuator 1
        elif "kp" in attr:
            x0 = true_pi_kp*3  # EXACT: 0.001
        elif "Ti" in attr:
            x0 = true_pi_Ti/2  # EXACT: 8.0
        elif "Td" in attr:
            x0 = 0.0  # EXACT: 0
        elif "offValue" in attr:
            x0 = 0.0  # EXACT: 0
        elif "onValue" in attr:
            x0 = 1.0  # EXACT: 1
        elif "steepness" in attr:
            x0 = true_onoff_steepness/2  # EXACT: 100
        parameters_exact_optimal.append((comp, attr, x0, lb, ub))
    
    # Initialize weights at 0.5 as per the formulation
    # This is key: at x=0.5, the binarization penalty gradient is zero,
    # so initial direction is determined purely by data fit
    parameters_init_half = []
    for comp, attr, x0, lb, ub, *_ in parameters:
        if 'alpha' in attr or 'beta' in attr or 'gamma' in attr:
            # Selection weights: initialize at 0.5
            if isinstance(x0, (list, np.ndarray)):
                x0 = [0.5] * len(x0)
            else:
                x0 = 0.5
        parameters_init_half.append((comp, attr, x0, lb, ub))
    parameters = parameters_init_half
    print("   Using x0=0.5 for selection weights (as per formulation)")
    
    # Debug: verify x0 values
    print("\n   Verifying x0 values for selection weights:")
    for comp, attr, x0, lb, ub, *_ in parameters:
        if 'alpha' in attr or 'beta' in attr or 'gamma' in attr:
            print(f"     {attr}: x0={x0}")


    parameters = parameters_exact_optimal
    
    print(f"   Number of parameters to estimate: {len(parameters)}")
    
    # Count parameter types
    n_alpha = sum(1 for p in parameters if 'alpha' in p[1])
    n_beta = sum(1 for p in parameters if 'beta' in p[1])
    n_gamma = sum(1 for p in parameters if 'gamma' in p[1])
    n_ctrl = sum(1 for p in parameters if 'candidate' in p[1])
    
    print(f"   - Alpha (controller selection): {n_alpha} (2 actuators × 2 candidates)")
    print(f"   - Beta (sensor selection): {n_beta} (2 actuators × 7 sensors)")
    print(f"   - Gamma (setpoint selection): {n_gamma} (2 actuators × 6 setpoints)")
    print(f"   - Controller parameters: {n_ctrl}")
    
    # Measurements: both actuator sensors
    measurements = [
        (actuator_sensor_pi, 0.02),
        (actuator_sensor_onoff, 0.02),
    ]
    
    # Initial simulation
    print("\n6. Running initial simulation...")
    simulator.simulate(start_time=start_time, end_time=end_time, step_size=step_size)
    
    initial_pred_pi = actuator_sensor_pi.input["measuredValue"].history(i_s=0, i_c=0).detach().numpy()
    initial_pred_onoff = actuator_sensor_onoff.input["measuredValue"].history(i_s=0, i_c=0).detach().numpy()
    actual_pi = df_actuator_pi['value'].values
    actual_onoff = df_actuator_onoff['value'].values
    
    print(f"\n   Initial weights:")
    print(f"     Alpha Act.0:        {[f'{controller._get_alpha(0, c).item():.3f}' for c in range(len(controller_classes))]}")
    print(f"     Beta Act.0:         {[f'{controller._get_beta(0, i).item():.3f}' for i in range(len(sensors))]}")
    print(f"     Gamma Act.0:        {[f'{controller._get_gamma(0, i).item():.3f}' for i in range(len(setpoints))]}")
    print(f"     Alpha Act.1:        {[f'{controller._get_alpha(1, c).item():.3f}' for c in range(len(controller_classes))]}")
    print(f"     Beta Act.1:         {[f'{controller._get_beta(1, i).item():.3f}' for i in range(len(sensors))]}")
    print(f"     Gamma Act.1:        {[f'{controller._get_gamma(1, i).item():.3f}' for i in range(len(setpoints))]}")
    
    # =========================================================================
    # Run Estimation
    # =========================================================================
    print("\n7. Running parameter estimation...")
    print("   Using regularization_lambda=1.0 for binarization penalty P(x)=x(1-x)")
    
    estimator = tb.Estimator(simulator)
    
    # Debug: Print parameter mapping
    print("\n   [DEBUG] Parameter mapping (theta index -> parameter):")
    theta_idx = 0
    for comp, attr, x0, lb, ub, *_ in parameters:
        if isinstance(x0, (list, np.ndarray)):
            n_vals = len(x0)
        else:
            n_vals = 1
        print(f"     theta[{theta_idx}:{theta_idx+n_vals}] -> {attr} (x0={x0}, lb={lb}, ub={ub})")
        theta_idx += n_vals
    
    # =========================================================================
    # TEST: Verify near-optimal x0 produces good fit BEFORE estimation
    # =========================================================================
    print("\n   [TEST] Setting parameters to x0 and running simulation...")
    
    # Extract x0 values and set them
    x0_values = [x0 for (comp, attr, x0, lb, ub) in parameters]
    components = [comp for (comp, attr, x0, lb, ub) in parameters]
    attrs = [attr for (comp, attr, x0, lb, ub) in parameters]
    
    # Set parameters using the same method as estimator
    model.set_parameters(
        values=x0_values,
        components=components,
        parameter_names=attrs,
        normalized=False,  # x0 values are already in physical units
    )
    
    # Print controller params after setting x0
    print("\n   [DEBUG] Controller params AFTER setting x0:")
    for a in range(2):
        alpha = controller._get_alpha_vector(a)
        beta = controller._get_beta_vector(a)
        gamma = controller._get_gamma_vector(a)
        print(f"     Actuator {a}:")
        print(f"       alpha: {alpha.detach().numpy()}")
        print(f"       beta:  {beta.detach().numpy()}")
        print(f"       gamma: {gamma.detach().numpy()}")
    
    # Print PI controller params
    ctrl_pi_0 = controller._get_candidate(0, 0)
    ctrl_onoff_1 = controller._get_candidate(1, 1)
    print(f"\n     PI Controller (Act.0, Cand.0):")
    print(f"       kp = {ctrl_pi_0.kp.get().item():.6f}")
    print(f"       Ti = {ctrl_pi_0.Ti.get().item():.6f}")
    print(f"       Td = {ctrl_pi_0.Td.get().item():.6f}")
    # print(f"     OnOff Controller (Act.1, Cand.1):")
    # print(f"       offValue = {ctrl_onoff_1.offValue.get().item():.6f}")
    # print(f"       onValue = {ctrl_onoff_1.onValue.get().item():.6f}")
    # print(f"       steepness = {ctrl_onoff_1.steepness.get().item():.6f}")
    
    # Run simulation with x0 parameters
    print("\n   Running simulation with x0 parameters...")
    simulator.simulate(start_time=start_time, end_time=end_time, step_size=step_size)
    
    # Get predictions
    x0_pred_pi = actuator_sensor_pi.input["measuredValue"].history(i_s=0, i_c=0).detach().numpy()
    x0_pred_onoff = actuator_sensor_onoff.input["measuredValue"].history(i_s=0, i_c=0).detach().numpy()
    
    # Compute and print MSE
    mse_pi = np.mean((x0_pred_pi - actual_pi)**2)
    mse_onoff = np.mean((x0_pred_onoff - actual_onoff)**2)
    print(f"\n   [TEST RESULTS] MSE with x0 parameters:")
    print(f"     Actuator 0 (PI):    {mse_pi:.6f}")
    print(f"     Actuator 1 (OnOff): {mse_onoff:.6f}")
    print(f"     Total MSE:          {mse_pi + mse_onoff:.6f}")
    
    # Print predictions vs actual
    print(f"\n   [TEST] Predictions vs Actual (first 20 timesteps):")
    print(f"     Act.0 (PI) pred:   {x0_pred_pi[:20]}")
    print(f"     Act.0 (PI) actual: {actual_pi[:20]}")
    print(f"     Act.1 (OnOff) pred:   {x0_pred_onoff[:20]}")
    print(f"     Act.1 (OnOff) actual: {actual_onoff[:20]}")
    
    # Plot x0 test results
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    time_hours = np.arange(len(actual_pi)) * step_size / 3600
    
    axes[0].plot(time_hours, actual_pi, 'b-', label='Actual', linewidth=2)
    axes[0].plot(time_hours, x0_pred_pi, 'r--', label='Prediction (x0)', linewidth=2)
    axes[0].set_ylabel('Actuator 0 (PI)')
    axes[0].set_title(f'Actuator 0: MSE = {mse_pi:.6f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time_hours, actual_onoff, 'b-', label='Actual', linewidth=2)
    axes[1].plot(time_hours, x0_pred_onoff, 'r--', label='Prediction (x0)', linewidth=2)
    axes[1].set_ylabel('Actuator 1 (OnOff)')
    axes[1].set_xlabel('Time (hours)')
    axes[1].set_title(f'Actuator 1: MSE = {mse_onoff:.6f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'x0_test_results.png'), dpi=150)
    plt.close()
    print(f"\n   [TEST] Saved x0 test plot to x0_test_results.png")
    
    # =========================================================================
    # Continue with estimation
    # =========================================================================
    
    # Debug: Print initial controller params BEFORE optimization
    print("\n   [DEBUG] Controller params BEFORE estimation:")
    for a in range(2):
        alpha = controller._get_alpha_vector(a)
        beta = controller._get_beta_vector(a)
        gamma = controller._get_gamma_vector(a)
        print(f"     Actuator {a}:")
        print(f"       alpha: {alpha.detach().numpy()}")
        print(f"       beta:  {beta.detach().numpy()}")
        print(f"       gamma: {gamma.detach().numpy()}")
    
    options = {
        "maxiter": 5000,  # More iterations for default x0
        "ftol": 1e-10,
        "disp": True,
    }
    
    # Add debug wrapper to print obj and jac during optimization
    _orig_obj_ad = estimator._obj_ad
    _orig_jac_ad = estimator._jac_ad
    _orig_obj = estimator._obj  # The internal objective that sets params
    _debug_iter = [0]  # Use list for mutable counter in closure
    
    def _debug_obj(theta_tensor, output="scalar"):
        """Wrap _obj to print controller params AFTER they're set."""
        result = _orig_obj(theta_tensor, output)
        
        # Only print on first iteration to see predictions vs actual
        if _debug_iter[0] == 0:
            print(f"\n  [CONTROLLER PARAMS after theta applied (iter {_debug_iter[0]})]:")
            # Per-actuator weights
            for a in range(2):
                alpha = controller._get_alpha_vector(a)
                beta = controller._get_beta_vector(a)
                gamma = controller._get_gamma_vector(a)
                print(f"    Actuator {a}:")
                print(f"      alpha: {alpha.detach().numpy()}")
                print(f"      beta:  {beta.detach().numpy()}")
                print(f"      gamma: {gamma.detach().numpy()}")
            
            # Print predictions vs actual data
            print(f"\n  [PREDICTIONS vs ACTUAL (first 20 timesteps)]:")
            pred_pi = actuator_sensor_pi.input["measuredValue"].history(i_s=0, i_c=0).detach().numpy()
            pred_onoff = actuator_sensor_onoff.input["measuredValue"].history(i_s=0, i_c=0).detach().numpy()
            print(f"    Actuator 0 (PI) predictions:     {pred_pi[:20]}")
            print(f"    Actuator 0 (PI) actual:          {actual_pi[:20]}")
            print(f"    Actuator 1 (OnOff) predictions:  {pred_onoff[:20]}")
            print(f"    Actuator 1 (OnOff) actual:       {actual_onoff[:20]}")
            
            # Print MSE for each actuator
            mse_pi = np.mean((pred_pi - actual_pi)**2)
            mse_onoff = np.mean((pred_onoff - actual_onoff)**2)
            print(f"\n    MSE Actuator 0 (PI):    {mse_pi:.6f}")
            print(f"    MSE Actuator 1 (OnOff): {mse_onoff:.6f}")
            print(f"    Total MSE:              {mse_pi + mse_onoff:.6f}")
            
            # Print the weighted signals being fed to controllers
            print(f"\n  [WEIGHTED SIGNALS (what controllers see)]:")
            # Get the input signals
            sensor_vals = controller.input["sensorValue"].history(i_s=0, i_c=0).detach().numpy()
            setpoint_vals = controller.input["setpointValue"].history(i_s=0, i_c=0).detach().numpy()
            print(f"    Raw sensor input shape:   {sensor_vals.shape}")
            print(f"    Raw setpoint input shape: {setpoint_vals.shape}")
            print(f"    Sensor[0] (first 10):     {sensor_vals[:10, 0] if sensor_vals.ndim > 1 else sensor_vals[:10]}")
            print(f"    Setpoint[0] (first 10):   {setpoint_vals[:10, 0] if setpoint_vals.ndim > 1 else setpoint_vals[:10]}")
        
        return result
    
    estimator._obj = _debug_obj
    
    # Build parameter names for debug output
    _param_names = []
    for comp, attr, x0, lb, ub, *_ in parameters:
        if isinstance(x0, (list, np.ndarray)):
            for i in range(len(x0)):
                _param_names.append(f"{attr}[{i}]")
        else:
            _param_names.append(attr)
    
    def _debug_obj_ad(theta, output="scalar"):
        result = _orig_obj_ad(theta, output)
        _debug_iter[0] += 1
        print(f"\n[DEBUG iter {_debug_iter[0]}] obj = {result:.6f}")
        
        # Print theta with parameter names
        print(f"  theta ({len(theta)} params):")
        for i, (name, val) in enumerate(zip(_param_names, theta)):
            print(f"    {name:25s} = {val:10.6f}")
        return result
    
    def _debug_jac_ad(theta, output="scalar"):
        result = _orig_jac_ad(theta, output)
        grad_norm = np.linalg.norm(result)
        print(f"  |grad| = {grad_norm:.6f}")
        
        # Print gradients with parameter names
        print(f"  grad ({len(result)} params):")
        for i, (name, val) in enumerate(zip(_param_names, result)):
            print(f"    {name:25s} = {val:10.6f}")
        return result
    
    estimator._obj_ad = _debug_obj_ad
    estimator._jac_ad = _debug_jac_ad
    
    # Profile the estimate method
    # profiler = cProfile.Profile()
    # profiler.enable()
    
    result = estimator.estimate(
        start_time=start_time,
        end_time=end_time,
        step_size=step_size,
        parameters=parameters,
        measurements=measurements,
        n_warmup=10,
        method=("scipy", "SLSQP", "ad"),
        regularization_lambda=0.01,  # Binarization penalty: P(x) = x(1-x)
        options=options,
    )
    
    # Restore original methods
    estimator._obj = _orig_obj
    estimator._obj_ad = _orig_obj_ad
    estimator._jac_ad = _orig_jac_ad
    
    
    # profiler.disable()
    
    # # Print profiling results
    # print("\n" + "=" * 80)
    # print("PROFILING RESULTS")
    # print("=" * 80)
    
    # # Sort by cumulative time and show top 30 functions
    # stream = io.StringIO()
    # stats = pstats.Stats(profiler, stream=stream)
    # stats.sort_stats('time')
    # stats.print_stats(30)
    # print(stream.getvalue())
    
    # # Also save to file for detailed analysis
    # profile_path = os.path.join(script_dir, "control_identification_profile.prof")
    # profiler.dump_stats(profile_path)
    # print(f"Full profile saved to: {profile_path}")
    # print(f"View with: python -m snakeviz {profile_path}")
    
    # =========================================================================
    # Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print("\n" + controller.summary())
    
    # Detailed analysis
    print("\n   SIGNAL SELECTION ANALYSIS:")
    
    # Expected: Actuator 0 uses sensor 0 (PI_TRUE), Actuator 1 uses sensor 1 (ONOFF_TRUE)
    for a in range(2):
        print(f"\n   Actuator {a} Beta weights (sensor selection):")
        # For actuator 0, expect sensor 0 (PI_TRUE) to be selected
        # For actuator 1, expect sensor 1 (ONOFF_TRUE) to be selected
        expected_sensor = a  # 0 for actuator 0, 1 for actuator 1
        for i, (df, name) in enumerate(sensors):
            beta = controller._get_beta(a, i).item()
            expected = "≈1.0" if i == expected_sensor else "≈0.0"
            correct = (i == expected_sensor and beta > 0.5) or (i != expected_sensor and beta < 0.5)
            status = "✓ CORRECT" if correct else "✗ WRONG"
            print(f"     β_{a},{i} = {beta:.4f} (expected {expected}) [{name}] {status}")
    
    for a in range(2):
        print(f"\n   Actuator {a} Gamma weights (setpoint selection):")
        for i, (df, name) in enumerate(setpoints):
            gamma = controller._get_gamma(a, i).item()
            expected = "≈1.0" if "TRUE" in name else "≈0.0"
            correct = ("TRUE" in name and gamma > 0.5) or ("DECOY" in name and gamma < 0.5)
            status = "✓ CORRECT" if correct else "✗ WRONG"
            print(f"     γ_{a},{i} = {gamma:.4f} (expected {expected}) [{name}] {status}")
    
    print("\n   Alpha weights (controller selection per actuator):")
    print("\n   Actuator 0 (True = PI, candidate 0):")
    for c, desc in enumerate(candidate_descriptions):
        alpha = controller._get_alpha(0, c).item()
        expected = "≈1.0" if c == 0 else "≈0.0"
        correct = (c == 0 and alpha > 0.5) or (c != 0 and alpha < 0.5)
        status = "✓ CORRECT" if correct else "✗ WRONG"
        print(f"     α_0,{c} = {alpha:.4f} (expected {expected}) [{desc}] {status}")
    
    print("\n   Actuator 1 (True = On-Off, candidate 1):")
    for c, desc in enumerate(candidate_descriptions):
        alpha = controller._get_alpha(1, c).item()
        expected = "≈1.0" if c == 1 else "≈0.0"
        correct = (c == 1 and alpha > 0.5) or (c != 1 and alpha < 0.5)
        status = "✓ CORRECT" if correct else "✗ WRONG"
        print(f"     α_1,{c} = {alpha:.4f} (expected {expected}) [{desc}] {status}")
    
    print("\n   Identified controller parameters:")
    for a in range(2):
        print(f"\n   Actuator {a}:")
        for c in range(len(controller_classes)):
            ctrl = controller._get_candidate(a, c)
            alpha = controller._get_alpha(a, c).item()
            if alpha > 0.3:  # Only show significant candidates
                print(f"     Candidate {c} ({ctrl.__class__.__name__}, α={alpha:.3f}):")
                # PID controller parameters
                if hasattr(ctrl, 'kp'):
                    true_kp = true_params['pi']['kp'] if a == 0 else "N/A"
                    print(f"       kp = {ctrl.kp.get().item():.6f} (true: {true_kp})")
                if hasattr(ctrl, 'Ti'):
                    true_Ti = true_params['pi']['Ti'] if a == 0 else "N/A"
                    print(f"       Ti = {ctrl.Ti.get().item():.6f} (true: {true_Ti})")
                if hasattr(ctrl, 'Td'):
                    print(f"       Td = {ctrl.Td.get().item():.6f}")
                # On-Off controller parameters
                # if hasattr(ctrl, 'offValue'):
                #     true_off = true_params['onoff']['offValue'] if a == 1 else "N/A"
                #     print(f"       offValue = {ctrl.offValue.get().item():.6f} (true: {true_off})")
                # if hasattr(ctrl, 'onValue'):
                #     true_on = true_params['onoff']['onValue'] if a == 1 else "N/A"
                #     print(f"       onValue = {ctrl.onValue.get().item():.6f} (true: {true_on})")
                # if hasattr(ctrl, 'steepness'):
                #     true_k = true_params['onoff']['steepness'] if a == 1 else "N/A"
                #     print(f"       steepness = {ctrl.steepness.get().item():.6f} (true: {true_k})")
    
    # =========================================================================
    # Plot Results
    # =========================================================================
    print("\n8. Generating result plots...")
    
    final_pred_pi = actuator_sensor_pi.input["measuredValue"].history(i_s=0, i_c=0).detach().numpy()
    final_pred_onoff = actuator_sensor_onoff.input["measuredValue"].history(i_s=0, i_c=0).detach().numpy()
    
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 1.5])
    
    # Plot 1: Actuator 0 (PI) comparison
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_hours, actual_pi, 'g-', alpha=0.7, linewidth=1.5, label='Actual (PI)')
    ax1.plot(time_hours, initial_pred_pi, 'b--', alpha=0.5, linewidth=1, label='Initial')
    ax1.plot(time_hours, final_pred_pi, 'm-', linewidth=1.5, label='Identified')
    ax1.set_ylabel('Actuator 0')
    ax1.legend(loc='upper right')
    ax1.set_title('Actuator 0: PI Controller Identification')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Actuator 1 (On-Off) comparison
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(time_hours, actual_onoff, 'g-', alpha=0.7, linewidth=1.5, label='Actual (On-Off)')
    ax2.plot(time_hours, initial_pred_onoff, 'b--', alpha=0.5, linewidth=1, label='Initial')
    ax2.plot(time_hours, final_pred_onoff, 'm-', linewidth=1.5, label='Identified')
    ax2.set_ylabel('Actuator 1')
    ax2.legend(loc='upper right')
    ax2.set_title('Actuator 1: On-Off Controller Identification')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Beta weights (sensor selection) for Actuator 0
    ax3 = fig.add_subplot(gs[2, 0])
    beta_vals_0 = [controller._get_beta(0, i).item() for i in range(len(sensors))]
    # Actuator 0 should select sensor 0 (PI_TRUE)
    colors = ['green' if i == 0 else 'gray' for i in range(len(sensors))]
    ax3.bar(range(len(sensors)), beta_vals_0, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(sensors)))
    ax3.set_xticklabels([f"β0,{i}" for i in range(len(sensors))])
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Weight Value')
    ax3.set_ylim(0, 1.1)
    ax3.set_title('Beta Act.0 (Expected: sensor 0)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Beta weights (sensor selection) for Actuator 1
    ax4 = fig.add_subplot(gs[2, 1])
    beta_vals_1 = [controller._get_beta(1, i).item() for i in range(len(sensors))]
    # Actuator 1 should select sensor 1 (ONOFF_TRUE)
    colors = ['green' if i == 1 else 'gray' for i in range(len(sensors))]
    ax4.bar(range(len(sensors)), beta_vals_1, color=colors, alpha=0.7)
    ax4.set_xticks(range(len(sensors)))
    ax4.set_xticklabels([f"β1,{i}" for i in range(len(sensors))])
    ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax4.set_ylabel('Weight Value')
    ax4.set_ylim(0, 1.1)
    ax4.set_title('Beta Act.1 (Expected: sensor 1)')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Alpha weights for Actuator 0
    ax5 = fig.add_subplot(gs[3, 0])
    alpha_vals_0 = [controller._get_alpha(0, c).item() for c in range(len(controller_classes))]
    colors = ['green' if c == 0 else 'gray' for c in range(len(controller_classes))]
    ax5.bar(range(len(controller_classes)), alpha_vals_0, color=colors, alpha=0.7)
    ax5.set_xticks(range(len(controller_classes)))
    ax5.set_xticklabels([f"α0,{c}" for c in range(len(controller_classes))])
    ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax5.set_ylabel('Weight Value')
    ax5.set_ylim(0, 1.1)
    ax5.set_title('Alpha Actuator 0 (Expected: PI = α0,0)')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Alpha weights for Actuator 1
    ax6 = fig.add_subplot(gs[3, 1])
    alpha_vals_1 = [controller._get_alpha(1, c).item() for c in range(len(controller_classes))]
    colors = ['green' if c == 1 else 'gray' for c in range(len(controller_classes))]
    ax6.bar(range(len(controller_classes)), alpha_vals_1, color=colors, alpha=0.7)
    ax6.set_xticks(range(len(controller_classes)))
    ax6.set_xticklabels([f"α1,{c}" for c in range(len(controller_classes))])
    ax6.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax6.set_ylabel('Weight Value')
    ax6.set_ylim(0, 1.1)
    ax6.set_title('Alpha Actuator 1 (Expected: On-Off = α1,1)')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Plot 7: Error for both actuators
    ax7 = fig.add_subplot(gs[4, :])
    error_pi = np.abs(actual_pi - final_pred_pi)
    error_onoff = np.abs(actual_onoff - final_pred_onoff)
    ax7.plot(time_hours, error_pi, 'b-', alpha=0.7, label=f'Actuator 0 (PI) MAE: {error_pi.mean():.4f}')
    ax7.plot(time_hours, error_onoff, 'r-', alpha=0.7, label=f'Actuator 1 (On-Off) MAE: {error_onoff.mean():.4f}')
    ax7.set_ylabel('Absolute Error')
    ax7.set_xlabel('Time (hours)')
    ax7.legend(loc='upper right')
    ax7.set_title('Prediction Error for Both Actuators')
    ax7.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "complex_dual_actuator_results.png"), dpi=150)
    plt.show()
    
    # =========================================================================
    # Compute Success Metrics
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUCCESS METRICS")
    print("=" * 80)
    
    # Count correct selections (per-actuator)
    # Actuator 0 should select sensor 0, Actuator 1 should select sensor 1
    n_correct_beta_0 = sum(1 for i in range(len(sensors))
                          if ((i == 0 and beta_vals_0[i] > 0.5) or 
                              (i != 0 and beta_vals_0[i] < 0.5)))
    n_correct_beta_1 = sum(1 for i in range(len(sensors))
                          if ((i == 1 and beta_vals_1[i] > 0.5) or 
                              (i != 1 and beta_vals_1[i] < 0.5)))
    n_correct_sensors = n_correct_beta_0 + n_correct_beta_1
    
    # Gamma (setpoint) - both actuators should select setpoint 0 (TRUE)
    gamma_vals_0 = [controller._get_gamma(0, j).item() for j in range(len(setpoints))]
    gamma_vals_1 = [controller._get_gamma(1, j).item() for j in range(len(setpoints))]
    n_correct_gamma_0 = sum(1 for i, (df, name) in enumerate(setpoints)
                           if (("TRUE" in name and gamma_vals_0[i] > 0.5) or 
                               ("DECOY" in name and gamma_vals_0[i] < 0.5)))
    n_correct_gamma_1 = sum(1 for i, (df, name) in enumerate(setpoints)
                           if (("TRUE" in name and gamma_vals_1[i] > 0.5) or 
                               ("DECOY" in name and gamma_vals_1[i] < 0.5)))
    n_correct_setpoints = n_correct_gamma_0 + n_correct_gamma_1
    
    # Actuator 0: Should select candidate 0 (PI)
    n_correct_ctrl_0 = sum(1 for c in range(len(controller_classes))
                          if ((c == 0 and alpha_vals_0[c] > 0.5) or 
                              (c != 0 and alpha_vals_0[c] < 0.5)))
    
    # Actuator 1: Should select candidate 4 (On-Off)
    n_correct_ctrl_1 = sum(1 for c in range(len(controller_classes))
                          if ((c == 1 and alpha_vals_1[c] > 0.5) or 
                              (c != 1 and alpha_vals_1[c] < 0.5)))
    
    print(f"\n   Beta selection (sensors): {n_correct_sensors}/{2 * len(sensors)} correct (2 actuators)")
    print(f"   Gamma selection (setpoints): {n_correct_setpoints}/{2 * len(setpoints)} correct (2 actuators)")
    print(f"   Actuator 0 controller:  {n_correct_ctrl_0}/{len(controller_classes)} correct")
    print(f"   Actuator 1 controller:  {n_correct_ctrl_1}/{len(controller_classes)} correct")
    
    total_correct = n_correct_sensors + n_correct_setpoints + n_correct_ctrl_0 + n_correct_ctrl_1
    total_possible = 2 * len(sensors) + 2 * len(setpoints) + 2 * len(controller_classes)
    print(f"\n   Overall: {total_correct}/{total_possible} ({100*total_correct/total_possible:.1f}%)")
    
    # Final MAE
    print(f"\n   Final MAE Actuator 0 (PI):     {error_pi.mean():.6f}")
    print(f"   Final MAE Actuator 1 (On-Off): {error_onoff.mean():.6f}")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    return controller, result


if __name__ == "__main__":
    controller, result = run_complex_identification_example()
    
    print("\n" + "=" * 80)
    print("Dual-actuator complex example completed!")
    print("=" * 80)
