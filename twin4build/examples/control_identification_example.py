"""
Controller Identification Example using Continuous Relaxation

This example demonstrates how to use ControllerIdentificationTorchSystem
with the twin4build Estimator to identify control logic from observed data.

Phase 1: Generate realistic data using actual twin4build components:
    - BuildingSpaceThermalTorchSystem for thermal dynamics
    - PIDControllerSystem as the "unknown" controller
    
Phase 2: Use ControllerIdentificationTorchSystem to recover the controller
structure and parameters from the observed data.
"""

# Standard library imports
import datetime
import os
import tempfile

# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    
    Returns dict of DataFrames for outdoor temperature, solar radiation, etc.
    """
    timestamps = [start_time + datetime.timedelta(seconds=i * step_size) 
                  for i in range(n_timesteps)]
    
    # Time in hours for daily patterns
    hours = np.array([(start_time + datetime.timedelta(seconds=i * step_size)).hour 
                      + (start_time + datetime.timedelta(seconds=i * step_size)).minute / 60
                      for i in range(n_timesteps)])
    
    # Outdoor temperature: cold winter day, varies between -8°C (night) and 2°C (day)
    outdoor_temp = -3.0 + 5.0 * np.sin(2 * np.pi * (hours - 14) / 24)  # Peak at 2pm
    outdoor_temp += np.random.normal(0, 0.5, n_timesteps)  # Add noise
    
    # Solar radiation: peaks at noon, zero at night (W/m²)
    solar = np.maximum(0, 400 * np.sin(np.pi * (hours - 6) / 12))
    solar = np.where((hours >= 6) & (hours <= 18), solar, 0)
    solar += np.random.normal(0, 20, n_timesteps)
    solar = np.maximum(0, solar)
    
    # Occupancy: people present during work hours
    occupancy = np.where((hours >= 8) & (hours <= 17), 
                         np.random.randint(1, 5, n_timesteps), 0).astype(float)
    
    # Setpoint: varies through the day
    setpoint = np.ones(n_timesteps) * 20.0  # Default
    setpoint = np.where((hours >= 6) & (hours < 8), 21.0, setpoint)   # Morning warmup
    setpoint = np.where((hours >= 8) & (hours < 18), 21.5, setpoint)  # Occupied hours
    setpoint = np.where((hours >= 18) & (hours < 22), 20.5, setpoint) # Evening
    setpoint = np.where((hours >= 22) | (hours < 6), 18.0, setpoint)  # Night setback
    
    # Create DataFrames
    data = {
        'outdoor_temp': pd.DataFrame({'datetime': timestamps, 'value': outdoor_temp}),
        'solar': pd.DataFrame({'datetime': timestamps, 'value': solar}),
        'occupancy': pd.DataFrame({'datetime': timestamps, 'value': occupancy}),
        'setpoint': pd.DataFrame({'datetime': timestamps, 'value': setpoint}),
        # Zero values for unused inputs
        'supply_flow': pd.DataFrame({'datetime': timestamps, 'value': np.zeros(n_timesteps)}),
        'exhaust_flow': pd.DataFrame({'datetime': timestamps, 'value': np.zeros(n_timesteps)}),
        'supply_temp': pd.DataFrame({'datetime': timestamps, 'value': np.ones(n_timesteps) * 35.0}),
    }
    
    return data


def generate_data_with_twin4build(
    n_timesteps: int = 1440,  # 24 hours at 1-minute steps
    step_size: int = 60,
    kp: float = 0.1,
    Ti: float = 10,  # Integral time constant
):
    """
    Generate synthetic data using actual twin4build components.
    
    Creates a closed-loop simulation with the following chain:
        Controller(inputSignal) -> Valve(waterFlowRate) -> 
        SpaceHeater(Power) -> BuildingSpace(indoorTemperature)
    
    This represents a realistic hydronic heating system where:
    - The PI controller modulates valve position based on temperature error
    - The valve converts position to water flow rate
    - The space heater converts water flow to heat output
    - The building space responds thermally to the heat input
    
    Returns DataFrames with temperature, setpoint, and actuator data.
    """
    print("   Setting up data generation model...")
    print("   Chain: Controller -> Valve -> SpaceHeater -> BuildingSpace")
    
    start_time = datetime.datetime(2024, 1, 15, 0, 0, 0, tzinfo=tz.UTC)
    end_time = start_time + datetime.timedelta(seconds=n_timesteps * step_size)
    
    # Create weather data
    weather = create_weather_data(n_timesteps, step_size, start_time)
    
    # Save weather data to temp files
    temp_dir = tempfile.mkdtemp()
    file_paths = {}
    for name, df in weather.items():
        path = os.path.join(temp_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        file_paths[name] = path
    
    # Create model
    model = tb.Model(id="data_generation_model")
    
    # =========================================================================
    # Create Components
    # =========================================================================
    
    # Building space with moderate thermal mass
    building_space = tb.BuildingSpaceThermalTorchSystem(
        C_air=5e5,      # Thermal capacitance of air [J/K]
        C_wall=2e6,     # Wall thermal capacitance [J/K]
        C_int=1e5,      # Internal mass [J/K]
        R_out=0.01,     # Outdoor resistance [K/W]
        R_in=0.01,      # Indoor resistance [K/W]
        R_int=0.02,     # Internal resistance [K/W]
        f_wall=0.4,     # Solar to wall fraction
        f_air=0.2,      # Solar to air fraction
        Q_occ_gain=80.0,  # Heat per occupant [W]
        id="room"
    )
    
    # Valve: converts controller signal (0-1) to water flow rate
    # Sized for ~2000W heating capacity at design conditions
    valve = tb.ValveTorchSystem(
        waterFlowRateMax=0.05,  # Max flow rate [kg/s] 
        valveAuthority=0.5,     # Moderate valve authority for good control
        id="heating_valve"
    )
    
    # Space heater (radiator): converts water flow to heat output
    space_heater = tb.SpaceHeaterTorchSystem(
        Q_flow_nominal_sh=2000.0,  # Nominal heat output [W]
        T_a_nominal_sh=55.0,       # Nominal supply water temp [°C]
        T_b_nominal_sh=45.0,       # Nominal return water temp [°C]
        TAir_nominal_sh=20.0,      # Nominal room temp [°C]
        id="radiator"
    )
    
    # PI controller (this is what we'll try to identify)
    pi_controller = tb.PIDControllerSystem(
        kp=kp,
        Ti=Ti,
        Td=0.0,  # No derivative action
        isReverse=True,  # Heating mode
        id="true_controller"
    )
    
    # Schedule systems for external inputs
    outdoor_temp_schedule = tb.ScheduleSystem(filename=file_paths['outdoor_temp'], id="outdoor_temp")
    solar_schedule = tb.ScheduleSystem(filename=file_paths['solar'], id="solar")
    occupancy_schedule = tb.ScheduleSystem(filename=file_paths['occupancy'], id="occupancy")
    setpoint_schedule = tb.ScheduleSystem(filename=file_paths['setpoint'], id="setpoint")
    supply_flow_schedule = tb.ScheduleSystem(filename=file_paths['supply_flow'], id="supply_flow")
    exhaust_flow_schedule = tb.ScheduleSystem(filename=file_paths['exhaust_flow'], id="exhaust_flow")
    supply_temp_schedule = tb.ScheduleSystem(filename=file_paths['supply_temp'], id="supply_temp")
    
    # Supply water temperature for the heating system (constant hot water)
    supply_water_temp_data = pd.DataFrame({
        'datetime': weather['outdoor_temp']['datetime'],
        'value': np.ones(n_timesteps) * 55.0  # 55°C supply water
    })
    supply_water_temp_file = os.path.join(temp_dir, "supply_water_temp.csv")
    supply_water_temp_data.to_csv(supply_water_temp_file, index=False)
    supply_water_temp_schedule = tb.ScheduleSystem(filename=supply_water_temp_file, id="supply_water_temp")
    
    # =========================================================================
    # Add Components to Model
    # =========================================================================
    model.add_component(building_space)
    model.add_component(valve)
    model.add_component(space_heater)
    model.add_component(pi_controller)
    model.add_component(outdoor_temp_schedule)
    model.add_component(solar_schedule)
    model.add_component(occupancy_schedule)
    model.add_component(setpoint_schedule)
    model.add_component(supply_flow_schedule)
    model.add_component(exhaust_flow_schedule)
    model.add_component(supply_temp_schedule)
    model.add_component(supply_water_temp_schedule)
    
    # =========================================================================
    # Connect Components
    # =========================================================================
    
    # Weather/environment -> Building space
    model.add_connection(outdoor_temp_schedule, building_space, "scheduleValue", "outdoorTemperature")
    model.add_connection(solar_schedule, building_space, "scheduleValue", "globalIrradiation")
    model.add_connection(occupancy_schedule, building_space, "scheduleValue", "numberOfPeople")
    model.add_connection(supply_flow_schedule, building_space, "scheduleValue", "supplyAirFlowRate")
    model.add_connection(exhaust_flow_schedule, building_space, "scheduleValue", "exhaustAirFlowRate")
    model.add_connection(supply_temp_schedule, building_space, "scheduleValue", "supplyAirTemperature")
    
    # Control loop: Building temp -> Controller
    model.add_connection(building_space, pi_controller, "indoorTemperature", "actualValue")
    model.add_connection(setpoint_schedule, pi_controller, "scheduleValue", "setpointValue")
    
    # Control chain: Controller -> Valve -> SpaceHeater -> BuildingSpace
    model.add_connection(pi_controller, valve, "inputSignal", "valvePosition")
    model.add_connection(valve, space_heater, "waterFlowRate", "waterFlowRate")
    model.add_connection(supply_water_temp_schedule, space_heater, "scheduleValue", "supplyWaterTemperature")
    model.add_connection(building_space, space_heater, "indoorTemperature", "indoorTemperature")
    model.add_connection(space_heater, building_space, "Power", "heatGain")
    
    # =========================================================================
    # Load and Simulate
    # =========================================================================
    model.load(
        draw_semantic_model=False,
        draw_simulation_model=False,
        verbose=0,
    )
    
    print("   Running closed-loop simulation...")
    simulator = tb.Simulator(model)
    
    simulator.simulate(
        start_time=start_time,
        end_time=end_time,
        step_size=step_size,
    )
    
    # =========================================================================
    # Extract Results
    # =========================================================================
    timestamps = [start_time + datetime.timedelta(seconds=i * step_size) 
                  for i in range(n_timesteps)]
    
    temperature = building_space.output["indoorTemperature"].history[0].detach().numpy()
    actuator = pi_controller.output["inputSignal"].history[0].detach().numpy()
    heater_power = space_heater.output["Power"].history[0].detach().numpy()
    setpoint = weather['setpoint']['value'].values
    
    # Create output DataFrames (no noise added to outputs - noise only in inputs/weather)
    df_temperature = pd.DataFrame({'datetime': timestamps, 'value': temperature})
    df_setpoint = pd.DataFrame({'datetime': timestamps, 'value': setpoint})
    df_actuator = pd.DataFrame({'datetime': timestamps, 'value': actuator})
    df_heater_power = pd.DataFrame({'datetime': timestamps, 'value': heater_power})
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"   Generated {n_timesteps} timesteps of data")
    print(f"   Temperature range: {temperature.min():.1f}°C to {temperature.max():.1f}°C")
    print(f"   Actuator range: {actuator.min():.2f} to {actuator.max():.2f}")
    print(f"   Heater power range: {heater_power.min():.0f}W to {heater_power.max():.0f}W")
    
    return df_temperature, df_setpoint, df_actuator, df_heater_power, weather, start_time, end_time, {
        'kp': kp,
        'Ti': Ti,
        'ki': kp / Ti,  # Derived Ki
    }


def run_controller_identification_example():
    """
    Run controller identification using the twin4build Estimator.
    
    Phase 1: Generate data with known PI controller
    Phase 2: Identify controller from data
    """
    print("=" * 60)
    print("Controller Identification with twin4build")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # =========================================================================
    # Phase 1: Generate Data with twin4build simulation
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Data Generation with Known Controller")
    print("=" * 60)
    
    step_size = 60  # 1 minute steps
    n_timesteps = 1440  # 24 hours
    
    # True controller parameters
    true_kp = 0.1
    true_Ti = 10
    
    print(f"\n   True controller: PI with Kp={true_kp}, Ti={true_Ti}s")
    print(f"   (Equivalent Ki = Kp/Ti = {true_kp/true_Ti:.6f})")
    
    df_temperature, df_setpoint, df_actuator, df_heater_power, weather, start_time, end_time, true_params = \
        generate_data_with_twin4build(
            n_timesteps=n_timesteps,
            step_size=step_size,
            kp=true_kp,
            Ti=true_Ti,
        )
    
    # =========================================================================
    # Plot Generated Data for Verification
    # =========================================================================
    print("\n   Plotting generated data for verification...")
    
    time_hours = np.arange(n_timesteps) * step_size / 3600
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Plot 1: Temperature and Setpoint
    ax1 = axes[0]
    ax1.plot(time_hours, df_temperature['value'].values, 'b-', linewidth=1, label='Indoor Temperature')
    ax1.plot(time_hours, df_setpoint['value'].values, 'r--', linewidth=2, label='Setpoint')
    ax1.plot(time_hours, weather['outdoor_temp']['value'].values, 'g-', alpha=0.7, linewidth=1, label='Outdoor Temperature')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend(loc='upper right')
    ax1.set_title('Generated Synthetic Data - Closed Loop Simulation\n'
                  f'Controller: PI with Kp={true_kp}, Ti={true_Ti}s (Ki={true_params["ki"]:.6f})')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Controller Output (Actuator Signal)
    ax2 = axes[1]
    ax2.plot(time_hours, df_actuator['value'].values, 'm-', linewidth=1)
    ax2.set_ylabel('Valve Position (0-1)')
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title('Controller Output (Valve Position)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Heater Power
    ax3 = axes[2]
    ax3.plot(time_hours, df_heater_power['value'].values, 'orange', linewidth=1)
    ax3.set_ylabel('Heat Output (W)')
    ax3.set_title('Space Heater Power Output')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: External Disturbances
    ax4 = axes[3]
    ax4.plot(time_hours, weather['solar']['value'].values, 'y-', linewidth=1, label='Solar Radiation (W/m²)')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(time_hours, weather['occupancy']['value'].values, 'c-', linewidth=1, label='Occupancy')
    ax4.set_ylabel('Solar (W/m²)', color='orange')
    ax4_twin.set_ylabel('Occupancy (people)', color='cyan')
    ax4.set_xlabel('Time (hours)')
    ax4.set_title('External Disturbances')
    ax4.grid(True, alpha=0.3)
    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    # Save and show
    synthetic_data_plot = "twin4build/examples/synthetic_data_generated.png"
    plt.savefig(synthetic_data_plot, dpi=150)
    plt.show()
    
    print(f"   Synthetic data plot saved to: {synthetic_data_plot}")
    
    # Save data to CSV for Phase 2
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "temperature.csv")
    sp_file = os.path.join(temp_dir, "setpoint.csv")
    act_file = os.path.join(temp_dir, "actuator.csv")
    
    df_temperature.to_csv(temp_file, index=False)
    df_setpoint.to_csv(sp_file, index=False)
    df_actuator.to_csv(act_file, index=False)
    
    # =========================================================================
    # Phase 2: Controller Identification
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Controller Identification")
    print("=" * 60)
    
    print("\n1. Creating identification model...")
    
    model = tb.Model(id="controller_identification_model")
    
    # Create ScheduleSystem for temperature sensor input
    temperature_schedule = tb.ScheduleSystem(
        filename=temp_file,
        id="temperature_input"
    )
    
    # Create ScheduleSystem for setpoint input  
    setpoint_schedule = tb.ScheduleSystem(
        filename=sp_file,
        id="setpoint_input"
    )
    
    # Create the controller to identify
    # Uses composed candidate controllers (default: P, PI, PID variants)
    controller = tb.ControllerIdentificationTorchSystem(
        n_sensors=1,
        n_setpoints=1,
        n_actuators=1,
        # candidate_controllers=None uses default PIDControllerSystem variants
        isReverse=True,  # Match the true controller
        id="identified_controller",
    )
    
    # Create sensor to measure actuator output
    actuator_sensor = tb.SensorSystem(
        filename=act_file,
        id="actuator_sensor"
    )
    
    # Add components to model
    model.add_component(temperature_schedule)
    model.add_component(setpoint_schedule)
    model.add_component(controller)
    model.add_component(actuator_sensor)
    
    # Create connections
    model.add_connection(temperature_schedule, controller, 
                         "scheduleValue", "sensorValue", input_port_index=0)
    model.add_connection(setpoint_schedule, controller,
                         "scheduleValue", "setpointValue", input_port_index=0)
    model.add_connection(controller, actuator_sensor,
                         "inputSignal", "measuredValue", output_port_index=0)
    
    # Load model
    model.load(
        draw_semantic_model=False,
        draw_simulation_model=True,
        verbose=0,
    )
    
    print(model)
    
    # =========================================================================
    # Setup Estimation
    # =========================================================================
    print("\n2. Setting up estimation...")
    
    simulator = tb.Simulator(model)
    
    # Define parameters to estimate using the controller's helper method
    # This includes alpha weights for each candidate and their parameters
    parameters = controller.get_estimator_parameters()
    
    print(f"   Number of parameters to estimate: {len(parameters)}")
    print("   Parameters:")
    for p in parameters:
        print(f"     - {p[0].id if hasattr(p[0], 'id') else p[0].__class__.__name__}.{p[1]} x0:{p[2]} lb:{p[3]} ub:{p[4]}")
    
    measurements = [(actuator_sensor, 0.02)]
    
    # Run initial simulation
    print("\n3. Running initial simulation...")
    simulator.simulate(
        start_time=start_time,
        end_time=end_time,
        step_size=step_size,
    )
    
    # Store initial predictions for comparison
    initial_predictions = actuator_sensor.input["measuredValue"].history[0].detach().numpy()
    actual_actuator = df_actuator['value'].values
    
    # Run estimation using TWO-STAGE approach
    # Stage 1: Fix selection weights (alpha, beta, gamma) and only optimize controller params
    # Stage 2: (Optional) Fine-tune all parameters together
    
    print("\n4. Running parameter estimation (Stage 1: Controller params only)...")
    print("   Fixing alpha=1, beta=1, gamma=1 to first tune kp, Ti, Td...")
    
    estimator = tb.Estimator(simulator)
    
    # Stage 1: Only controller parameters (skip first 3 which are alpha, beta, gamma)
    # controller_params = parameters[3:]  # kp, Ti, Td only
    
    print(f"   Stage 1 parameters: {len(parameters)}")
    for p in parameters:
        print(f"     - {p[0].id}.{p[1]} x0:{p[2]} lb:{p[3]} ub:{p[4]}")
    
    result = estimator.estimate(
        start_time=start_time,
        end_time=end_time,
        step_size=step_size,
        parameters=parameters,
        measurements=measurements,
        n_warmup=10,
        method=("scipy", "SLSQP", "ad"),
    )
    
    print("\n   Stage 1 complete. Controller parameters optimized.")
    print(f"   kp = {controller.candidate_0_0.kp.get().item():.6f}")
    print(f"   Ti = {controller.candidate_0_0.Ti.get().item():.6f}")
    print(f"   Td = {controller.candidate_0_0.Td.get().item():.6f}")
    
    # Stage 2: (Optional) Fine-tune all parameters including selection weights
    # Uncomment below if you want to also optimize alpha, beta, gamma
    # print("\n5. Running parameter estimation (Stage 2: All parameters)...")
    # gradient_scales = controller.get_gradient_scales(weight_scale=0.01)
    # result = estimator.estimate(
    #     start_time=start_time,
    #     end_time=end_time,
    #     step_size=step_size,
    #     parameters=parameters,
    #     measurements=measurements,
    #     n_warmup=10,
    #     method=("scipy", "L-BFGS-B", "ad"),
    #     gradient_scales=gradient_scales,
    # )
    
    # =========================================================================
    # Results
    # =========================================================================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\n" + controller.summary())
    
    # Get identified structure
    structure = controller.get_identified_structure(threshold=0.5)
    print("\nIdentified Structure:")
    print(f"   Active sensors: {structure['sensors']}")
    print(f"   Active setpoints: {structure['setpoints']}")
    for a, candidates in structure['actuators'].items():
        print(f"   Actuator {a} active controllers: {[c['class'] for c in candidates]}")
    
    # Compare with true parameters
    print("\nTrue Controller Parameters:")
    print(f"   True Kp: {true_params['kp']:.6f}")
    print(f"   True Ti: {true_params['Ti']:.6f}")
    print(f"   True Ki (Kp/Ti): {true_params['ki']:.6f}")
    
    # =========================================================================
    # Plot results
    # =========================================================================
    print("\n5. Generating plots...")
    
    final_predictions = actuator_sensor.input["measuredValue"].history[0].detach().numpy()
    actual_actuator = df_actuator['value'].values
    time_hours = np.arange(n_timesteps) * step_size / 3600
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Temperature and setpoint
    ax1 = axes[0]
    ax1.plot(time_hours, df_temperature['value'].values, 'b-', linewidth=1, label='Temperature')
    ax1.plot(time_hours, df_setpoint['value'].values, 'r--', linewidth=2, label='Setpoint')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend(loc='upper right')
    ax1.set_title('Controller Identification Results (24h Simulation)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Actuator comparison
    ax2 = axes[1]
    ax2.plot(time_hours, actual_actuator, 'g-', alpha=0.7, linewidth=1, label='Actual actuator')
    ax2.plot(time_hours, initial_predictions, 'b--', alpha=0.5, linewidth=1, label='Initial prediction')
    ax2.plot(time_hours, final_predictions, 'm-', linewidth=1, label='Identified prediction')
    ax2.set_ylabel('Actuator signal')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prediction error
    ax3 = axes[2]
    initial_error = np.abs(actual_actuator - initial_predictions)
    final_error = np.abs(actual_actuator - final_predictions)
    ax3.plot(time_hours, initial_error, 'b--', alpha=0.5, label=f'Initial MAE: {initial_error.mean():.4f}')
    ax3.plot(time_hours, final_error, 'm-', label=f'Final MAE: {final_error.mean():.4f}')
    ax3.set_ylabel('Absolute Error')
    ax3.set_xlabel('Time (hours)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save and show
    output_path = "twin4build/examples/control_identification_results.png"
    plt.savefig(output_path, dpi=150)
    plt.show()
    
    print(f"\n   Results saved to: {output_path}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    return controller, result


if __name__ == "__main__":
    controller, result = run_controller_identification_example()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
