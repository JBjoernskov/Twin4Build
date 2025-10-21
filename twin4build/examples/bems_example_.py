#!/usr/bin/env python3
"""
Building Energy Management: Parameter Estimation and Optimal Control
====================================================================

Learning Objectives:
- Understand thermal RC network modeling
- Learn parameter estimation from measurements
- Apply optimal control for energy savings

Physical System: Two-room building
- States: [T_a, T_wa, T_i, T_b, T_wb] (temperatures)
- Inputs: [Q_h, Q_r, T_out] (heating, solar, outdoor temp)
"""

import os
import twin4build as tb
import torch
import numpy as np
import pandas as pd
import datetime
from dateutil import tz


# ============================================================================
# THERMAL PARAMETERS
# ============================================================================

class ThermalParameters:
    """Container for building thermal parameters"""
    
    def __init__(self, R_wa=0.014, R_ia=0.005, R_wao=0.016,
                 R_ib=0.004, R_wb=0.014, R_wbo=0.018,
                 C_a=5e5, C_wa=15e5, C_i=1e5, C_b=1e5, C_wb=3e5):
        # Thermal resistances [K/W]
        self.R_wa = R_wa      # Room A air to wall
        self.R_ia = R_ia      # Room A air to interior wall
        self.R_wao = R_wao    # Room A wall to outdoor
        self.R_ib = R_ib      # Interior wall to Room B air
        self.R_wb = R_wb      # Room B air to wall
        self.R_wbo = R_wbo    # Room B wall to outdoor
        
        # Thermal capacitances [J/K]
        self.C_a = C_a        # Room A air
        self.C_wa = C_wa      # Room A wall
        self.C_i = C_i        # Interior wall
        self.C_b = C_b        # Room B air
        self.C_wb = C_wb      # Room B wall




# ============================================================================
# STEP 1: LOAD MEASUREMENT DATA
# ============================================================================

def load_data():
    """Load building measurement data"""
    print("="*60)
    print("STEP 1: LOADING MEASUREMENT DATA")
    print("="*60)
    
    # Generate synthetic data (simulates real sensor data)
    data_file = "building_measurements.csv"
    if not os.path.exists(data_file):
        _generate_data(data_file)
    
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    print(f"✓ Loaded {len(df)} measurements")
    print(f"✓ Period: {df.index[0]} to {df.index[-1]}")
    print(f"✓ Columns: {list(df.columns)}\n")
    
    # Plot the data
    tb.plot.plot(
        time=df.index,
        entries=[
            tb.plot.Entry(data=df['T_a_measured'].values, 
                         label="Room A Temp", color=tb.plot.Colors.blue, fmt="-", axis=1),
            tb.plot.Entry(data=df['T_b_measured'].values, 
                         label="Room B Temp", color=tb.plot.Colors.red, fmt="--", axis=1),
            tb.plot.Entry(data=df['Q_h'].values/1000, 
                         label="Heating", color=tb.plot.Colors.orange, fmt="-", axis=2),
        ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="Power [kW]",
        title="Building Measurements",
        show=True,
        nticks=11
    )
    
    return df


# ============================================================================
# STEP 2: PARAMETER ESTIMATION
# ============================================================================

def estimate_parameters(df):
    """Estimate thermal parameters from measurements"""
    print("\n" + "="*60)
    print("STEP 2: PARAMETER ESTIMATION")
    print("="*60)
    
    # Initial guess (deliberately inaccurate)
    # initial = ThermalParameters(
    #     R_wa=0.1, R_ia=0.04, R_wao=0.1,
    #     R_ib=0.04, R_wb=0.1, R_wbo=0.1,
    #     C_a=1e6, C_wa=5e5, C_i=1e6, C_b=1e6, C_wb=5e5
    # )
    

    initial = ThermalParameters(
        R_wa=0.01, R_ia=0.002, R_wao=0.01,
        R_ib=0.002, R_wb=0.01, R_wbo=0.01,
        C_a=1e6, C_wa=5e5, C_i=1e6, C_b=1e6, C_wb=5e5
    )

    print("Initial parameter guess:")
    print(f"  R_wa = {initial.R_wa:.3e} K/W")
    print(f"  C_a  = {initial.C_a:.3e} J/K")
    print("  ... (11 parameters total)")
    
    # Build model
    model = _build_model(df, initial)
    
    # Create estimator
    simulator = tb.Simulator(model)


    # Run initial simulation
    simulator.simulate(
        step_size=600,
        start_time=df.index[0],
        end_time=df.index[-1] + pd.Timedelta(seconds=600)
    )

    # Extract results
    thermal_sys = model.components["ThermalSystem"]
    estimated_T_a = thermal_sys.ss_model.output["y"].history[:,0].detach().numpy()
    estimated_T_b = thermal_sys.ss_model.output["y"].history[:,3].detach().numpy()
    
    measured_T_a = df['T_a_measured'].values
    measured_T_b = df['T_b_measured'].values

    # Plot comparison
    tb.plot.plot(
        time=df.index,
        entries=[
            tb.plot.Entry(data=measured_T_a, label=r"Measured $T_a$", 
                         color=tb.plot.Colors.black, fmt=".", axis=1, markersize=2),
            tb.plot.Entry(data=estimated_T_a, label=r"Estimated $T_a$", 
                         color=tb.plot.Colors.red, fmt="-", axis=1),
            tb.plot.Entry(data=measured_T_b, label=r"Measured $T_b$", 
                         color=tb.plot.Colors.blue, fmt=".", axis=1, markersize=2),
            tb.plot.Entry(data=estimated_T_b, label=r"Estimated $T_b$", 
                         color=tb.plot.Colors.orange, fmt="-", axis=1),
        ],
        ylabel_1axis=r"Temperature $[^\circ C]$",
        title="Before calibration: Measured vs Estimated",
        show=False,
        nticks=11
    )

    # save initial simulation results in csv file
    initial_results = pd.DataFrame({
        "time": df.index,
        "reference_T_a": estimated_T_a,
        "reference_T_b": estimated_T_b
    }, index=df.index)
    initial_results.to_csv("bems_example_reference.csv")

    estimator = tb.Estimator(simulator)
    
    # Define what to estimate
    thermal_sys = model.components["ThermalSystem"]
    parameters = [
        (thermal_sys, "R_wa_param", initial.R_wa, 0.001, 0.1),
        (thermal_sys, "R_ia_param", initial.R_ia, 0.001, 0.1),
        (thermal_sys, "R_wao_param", initial.R_wao, 0.001, 0.1),
        (thermal_sys, "R_ib_param", initial.R_ib, 0.001, 0.1),
        (thermal_sys, "R_wb_param", initial.R_wb, 0.001, 0.1),
        (thermal_sys, "R_wbo_param", initial.R_wbo, 0.001, 0.1),
        (thermal_sys, "C_a_param", initial.C_a, 1e5, 5e6),
        (thermal_sys, "C_wa_param", initial.C_wa, 1e5, 2e6),
        (thermal_sys, "C_i_param", initial.C_i, 1e5, 5e6),
        (thermal_sys, "C_b_param", initial.C_b, 1e5, 5e6),
        (thermal_sys, "C_wb_param", initial.C_wb, 1e5, 2e6),
    ]
    
    # Define measurements to match
    temp_a_sensor = model.components["TempASensor"]
    temp_b_sensor = model.components["TempBSensor"]
    measurements = [(temp_a_sensor, 0.05), (temp_b_sensor, 0.05)]
    
    print("\nRunning estimation (this may take a few minutes)...")
    
    # Run estimation
    estimator.estimate(
        start_time=df.index[0],
        end_time=df.index[-1] + pd.Timedelta(seconds=600),
        step_size=600,
        parameters=parameters,
        measurements=measurements,
        n_warmup=20,
        method=("scipy", "SLSQP", "ad"),
        options={"maxiter": 100, "ftol": 1e-6}
    )
    
    # Extract results
    estimated = ThermalParameters(
        R_wa=thermal_sys.R_wa_param.get(),
        R_ia=thermal_sys.R_ia_param.get(),
        R_wao=thermal_sys.R_wao_param.get(),
        R_ib=thermal_sys.R_ib_param.get(),
        R_wb=thermal_sys.R_wb_param.get(),
        R_wbo=thermal_sys.R_wbo_param.get(),
        C_a=thermal_sys.C_a_param.get(),
        C_wa=thermal_sys.C_wa_param.get(),
        C_i=thermal_sys.C_i_param.get(),
        C_b=thermal_sys.C_b_param.get(),
        C_wb=thermal_sys.C_wb_param.get()
    )
    
    print("✓ Estimation complete!")
    print(f"\nEstimated parameters:")
    print(f"  R_wa = {estimated.R_wa:.3e} K/W")
    print(f"  C_a  = {estimated.C_a:.3e} J/K")
    print("  ... (improved accuracy)")
    
    return estimated, model


# ============================================================================
# STEP 3: VALIDATE MODEL
# ============================================================================

def validate_model(df, model):
    """Validate the calibrated model"""
    print("\n" + "="*60)
    print("STEP 3: MODEL VALIDATION")
    print("="*60)
    
    # Simulate with estimated parameters
    simulator = tb.Simulator(model)
    simulator.simulate(
        step_size=600,
        start_time=df.index[0],
        end_time=df.index[-1] + pd.Timedelta(seconds=600)
    )
    
    # Extract results
    thermal_sys = model.components["ThermalSystem"]
    estimated_T_a = thermal_sys.ss_model.output["y"].history[:,0].detach().numpy()
    estimated_T_b = thermal_sys.ss_model.output["y"].history[:,3].detach().numpy()
    
    # Calculate accuracy
    measured_T_a = df['T_a_measured'].values
    measured_T_b = df['T_b_measured'].values
    rmse_a = np.sqrt(np.mean((measured_T_a - estimated_T_a)**2))
    rmse_b = np.sqrt(np.mean((measured_T_b - estimated_T_b)**2))
    
    print(f"✓ RMSE Room A: {rmse_a:.3f} °C")
    print(f"✓ RMSE Room B: {rmse_b:.3f} °C")
    
    # Plot comparison
    tb.plot.plot(
        time=df.index,
        entries=[
            tb.plot.Entry(data=measured_T_a, label=r"Measured $T_a$", 
                         color=tb.plot.Colors.black, fmt=".", axis=1, markersize=2),
            tb.plot.Entry(data=estimated_T_a, label=r"Estimated $T_a$", 
                         color=tb.plot.Colors.red, fmt="-", axis=1),
            tb.plot.Entry(data=measured_T_b, label=r"Measured $T_b$", 
                         color=tb.plot.Colors.blue, fmt=".", axis=1, markersize=2),
            tb.plot.Entry(data=estimated_T_b, label=r"Estimated $T_b$", 
                         color=tb.plot.Colors.orange, fmt="-", axis=1),
        ],
        ylabel_1axis=r"Temperature $[^\circ C]$",
        title="After calibration: Measured vs Estimated",
        show=True,
        nticks=11
    )


# ============================================================================
# STEP 4: OPTIMAL CONTROL
# ============================================================================

def optimize_control(df, model):
    """Optimize heating schedule for energy savings"""
    print("\n" + "="*60)
    print("STEP 4: OPTIMAL CONTROL")
    print("="*60)
    print("Objective: Minimize energy while maintaining comfort\n")
    
    # Run baseline simulation
    print("Running baseline simulation...")
    simulator = tb.Simulator(model)
    simulator.simulate(
        step_size=600,
        start_time=df.index[0],
        end_time=df.index[-1] + pd.Timedelta(seconds=600)
    )
    
    thermal_sys = model.components["ThermalSystem"]
    baseline_T = thermal_sys.ss_model.output["y"].history[:,0].detach().numpy()
    baseline_Q = model.components["RadiatorInput"].output["value"].history.detach().numpy()
    baseline_energy = np.sum(baseline_Q) * 600 / 3600 / 1000  # kWh
    
    print(f"✓ Baseline energy: {baseline_energy:.2f} kWh")
    
    # Create optimizable heating schedule
    schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 1000.0,
            "ruleset_start_hour": [0, 6, 9, 12, 15, 18, 22, 24],
            "ruleset_end_hour": [6, 9, 12, 15, 18, 22, 24, 24],
            "ruleset_value": [400, 1500, 1800, 1600, 1700, 1400, 400, 400],
            "ruleset_start_minute": [0]*8,
            "ruleset_end_minute": [0]*8,
        },
        id="HeatingSchedule"
    )
    
    # Define comfort bounds
    heating_sp = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 18.0,
            "ruleset_start_hour": [0, 6, 8, 17, 22, 0, 0],
            "ruleset_end_hour": [6, 8, 17, 22, 24, 0, 0],
            "ruleset_value": [10.0, 21, 21.0, 10, 10, 10.0, 10.0],
            "ruleset_start_minute": [0]*7,
            "ruleset_end_minute": [0]*7,
        },
        id="HeatingSetpoint"
    )
    
    cooling_sp = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 26.0,
            "ruleset_start_hour": [0, 6, 8, 17, 22, 0, 0],
            "ruleset_end_hour": [6, 8, 17, 22, 24, 0, 0],
            "ruleset_value": [30.0, 26.0, 24.0, 25.0, 26.0, 30.0, 30.0],
            "ruleset_start_minute": [0]*7,
            "ruleset_end_minute": [0]*7,
        },
        id="CoolingSetpoint"
    )
    
    # Replace heating input with schedule
    model.add_component(schedule)
    model.add_component(heating_sp)
    model.add_component(cooling_sp)
    
    radiator_input = model.components["RadiatorInput"]
    model.remove_connection(radiator_input, thermal_sys, "value", "u")
    model.add_connection(schedule, thermal_sys, "scheduleValue", "u", input_port_index=0)
    model.load()
    
    # Run optimization
    print("Running optimization...")
    optimizer = tb.Optimizer(tb.Simulator(model))
    
    temp_sensor = model.components["TempASensor"]
    optimizer.optimize(
        start_time=df.index[0],
        end_time=df.index[-1] + pd.Timedelta(seconds=600),
        step_size=600,
        variables=[(schedule, "scheduleValue", 0, 2000)],
        objectives=[(schedule, "scheduleValue", "min")],
        ineq_cons=[
            (temp_sensor, "measuredValue", "lower", heating_sp),
            (temp_sensor, "measuredValue", "upper", cooling_sp)
        ],
        method="scipy",
        options={"maxiter": 500, "disp": True}
    )
    
    # Extract results
    optimized_T = thermal_sys.ss_model.output["y"].history[:,0].detach().numpy()
    optimized_Q = schedule.output["scheduleValue"].history.detach().numpy()
    optimized_energy = np.sum(optimized_Q) * 600 / 3600 / 1000  # kWh
    savings = (baseline_energy - optimized_energy) / baseline_energy * 100
    
    print(f"✓ Optimized energy: {optimized_energy:.2f} kWh")
    print(f"✓ Energy savings: {savings:.1f}%")
    
    # Plot results
    time_index = pd.date_range(start=df.index[0], periods=len(baseline_T), freq='10min')
    heating_sp_vals = heating_sp.output["scheduleValue"].history.detach().numpy()
    cooling_sp_vals = cooling_sp.output["scheduleValue"].history.detach().numpy()
    

    # --- PLOT 1: Baseline Comparison ---
    tb.plot.plot(
        time=time_index,
        entries=[
            # Temperature bounds (shaded region would be ideal, but use lines)
            tb.plot.Entry(data=heating_sp_vals, label="Heating Setpoint (min)", 
                          color=tb.plot.Colors.red, fmt=":", axis=1, linewidth=2),
            tb.plot.Entry(data=cooling_sp_vals, label="Cooling Setpoint (max)", 
                          color=tb.plot.Colors.blue, fmt=":", axis=1, linewidth=2),
            # Baseline temperature
            tb.plot.Entry(data=baseline_T, label="Baseline Temperature", 
                          color=tb.plot.Colors.black, fmt="-", axis=1, linewidth=1.5),
            # Baseline heating on secondary axis
            tb.plot.Entry(data=baseline_Q/1000, label="Baseline Heating", 
                          color=tb.plot.Colors.orange, fmt="-", axis=2, linewidth=1.5),
        ],
        ylabel_1axis=r"Temperature $[^\circ$ C$]$",
        ylabel_2axis="Heating Power [kW]",
        ylim_1axis=(8, 32),
        ylim_2axis=(0, 6),
        title=f"Baseline Control (Energy: {baseline_energy:.1f} kWh)",
        show=False,
        nticks=13
    )
    
    # --- PLOT 2: Optimized Comparison ---
    tb.plot.plot(
        time=time_index,
        entries=[
            # Temperature bounds
            tb.plot.Entry(data=heating_sp_vals, label="Heating Setpoint (min)", 
                          color=tb.plot.Colors.red, fmt=":", axis=1, linewidth=2),
            tb.plot.Entry(data=cooling_sp_vals, label="Cooling Setpoint (max)", 
                          color=tb.plot.Colors.blue, fmt=":", axis=1, linewidth=2),
            # Optimized temperature
            tb.plot.Entry(data=optimized_T, label="Optimized Temperature", 
                          color=tb.plot.Colors.black, fmt="-", axis=1, linewidth=1.5),
            # Optimized heating on secondary axis
            tb.plot.Entry(data=optimized_Q/1000, label="Optimized Heating", 
                          color=tb.plot.Colors.orange, fmt="-", axis=2, linewidth=1.5),
        ],
        ylabel_1axis=r"Temperature $[^\circ$ C$]$",
        ylabel_2axis="Heating Power [kW]",
        ylim_1axis=(8, 32),
        ylim_2axis=(0, 3.5),
        title=f"Optimized Control Strategy (Energy: {optimized_energy:.1f} kWh, Savings: {savings:.1f}%)",
        show=True,
        nticks=13
    )


# ============================================================================
# HELPER FUNCTIONS (IMPLEMENTATION DETAILS)
# ============================================================================

def _build_model(df, params):
    """Build Twin4Build model (implementation detail)"""
    import tempfile
    import os
    import torch.nn as nn
    import twin4build.utils.types as tps
    
    # Create parametric system class
    class ParametricSystem(tb.core.System, nn.Module):
        def __init__(self, params, **kwargs):
            super().__init__(**kwargs)
            nn.Module.__init__(self)
            
            # Store parameters as learnable tensors
            self.R_wa_param = tps.Parameter(torch.tensor(params.R_wa, dtype=torch.float64))
            self.R_ia_param = tps.Parameter(torch.tensor(params.R_ia, dtype=torch.float64))
            self.R_wao_param = tps.Parameter(torch.tensor(params.R_wao, dtype=torch.float64))
            self.R_ib_param = tps.Parameter(torch.tensor(params.R_ib, dtype=torch.float64))
            self.R_wb_param = tps.Parameter(torch.tensor(params.R_wb, dtype=torch.float64))
            self.R_wbo_param = tps.Parameter(torch.tensor(params.R_wbo, dtype=torch.float64))
            self.C_a_param = tps.Parameter(torch.tensor(params.C_a, dtype=torch.float64))
            self.C_wa_param = tps.Parameter(torch.tensor(params.C_wa, dtype=torch.float64))
            self.C_i_param = tps.Parameter(torch.tensor(params.C_i, dtype=torch.float64))
            self.C_b_param = tps.Parameter(torch.tensor(params.C_b, dtype=torch.float64))
            self.C_wb_param = tps.Parameter(torch.tensor(params.C_wb, dtype=torch.float64))
            
            self.input = {"u": tps.Vector(size=3)}
            self.output = {"y": tps.Vector(size=5)}
            
            self.parameter = {
                "R_wa_param": {"lb": 0.001, "ub": 0.1},
                "R_ia_param": {"lb": 0.001, "ub": 0.1},
                "R_wao_param": {"lb": 0.001, "ub": 0.1},
                "R_ib_param": {"lb": 0.001, "ub": 0.1},
                "R_wb_param": {"lb": 0.001, "ub": 0.1},
                "R_wbo_param": {"lb": 0.001, "ub": 0.1},
                "C_a_param": {"lb": 1e5, "ub": 5e6},
                "C_wa_param": {"lb": 1e5, "ub": 2e6},
                "C_i_param": {"lb": 1e5, "ub": 5e6},
                "C_b_param": {"lb": 1e5, "ub": 5e6},
                "C_wb_param": {"lb": 1e5, "ub": 2e6},
            }
            
            self._config = {"parameters": list(self.parameter.keys())}
            self.INITIALIZED = False
            self._create_ss()
        
        def _build_matrices(self):
            A = torch.zeros((5, 5), dtype=torch.float64)
            A[0, 0] = -(1/(self.R_wa_param.get() * self.C_a_param.get()) + 
                        1/(self.R_ia_param.get() * self.C_a_param.get()))
            A[0, 1] = 1/(self.R_wa_param.get() * self.C_a_param.get())
            A[0, 2] = 1/(self.R_ia_param.get() * self.C_a_param.get())
            A[1, 0] = 1/(self.R_wa_param.get() * self.C_wa_param.get())
            A[1, 1] = -(1/(self.R_wa_param.get() * self.C_wa_param.get()) + 
                        1/(self.R_wao_param.get() * self.C_wa_param.get()))
            A[2, 0] = 1/(self.R_ia_param.get() * self.C_i_param.get())
            A[2, 2] = -(1/(self.R_ia_param.get() * self.C_i_param.get()) + 
                        1/(self.R_ib_param.get() * self.C_i_param.get()))
            A[2, 3] = 1/(self.R_ib_param.get() * self.C_i_param.get())
            A[3, 2] = 1/(self.R_ib_param.get() * self.C_b_param.get())
            A[3, 3] = -(1/(self.R_wb_param.get() * self.C_b_param.get()) + 
                        1/(self.R_ib_param.get() * self.C_b_param.get()))
            A[3, 4] = 1/(self.R_wb_param.get() * self.C_b_param.get())
            A[4, 3] = 1/(self.R_wb_param.get() * self.C_wb_param.get())
            A[4, 4] = -(1/(self.R_wb_param.get() * self.C_wb_param.get()) + 
                        1/(self.R_wbo_param.get() * self.C_wb_param.get()))
            
            B = torch.zeros((5, 3), dtype=torch.float64)
            B[0, 0] = 1/self.C_a_param.get()
            B[3, 1] = 1/self.C_b_param.get()
            B[1, 2] = 1/(self.R_wao_param.get() * self.C_wa_param.get())
            B[4, 2] = 1/(self.R_wbo_param.get() * self.C_wb_param.get())
            
            return A, B
        
        def _create_ss(self):
            A, B = self._build_matrices()
            C = torch.eye(5, dtype=torch.float64)
            D = torch.zeros((5, 3), dtype=torch.float64)
            x0 = torch.tensor([18.0, 12.0, 15.0, 16.0, 11.0], dtype=torch.float64)
            
            self.ss_model = tb.DiscreteStatespaceSystem(
                A=A, B=B, C=C, D=D, x0=x0,
                state_names=["T_a", "T_wa", "T_i", "T_b", "T_wb"],
                sample_time=600.0,
                id=f"ss_model_{self.id}"
            )
        
        @property
        def config(self):
            return self._config
        
        def initialize(self, start_time, end_time, step_size, simulator):
            for port in self.input.values():
                port.initialize(start_time, end_time, step_size, simulator)
            for port in self.output.values():
                port.initialize(start_time, end_time, step_size, simulator)
            
            self._create_ss()
            self.ss_model.initialize(start_time, end_time, step_size, simulator)
            self.INITIALIZED = True
        
        def do_step(self, secondTime=None, dateTime=None, step_size=None, step_index=None):
            self.ss_model.input["u"].set(self.input["u"].get(), step_index=step_index)
            self.ss_model.do_step(secondTime, dateTime, step_size, step_index=step_index)
            self.output["y"].set(self.ss_model.output["y"].get(), step_index)
    
    # Build model
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save measurements
        temp_a_file = os.path.join(temp_dir, "T_a.csv")
        temp_b_file = os.path.join(temp_dir, "T_b.csv")
        pd.DataFrame({'measuredValue': df['T_a_measured']}, index=df.index).to_csv(temp_a_file)
        pd.DataFrame({'measuredValue': df['T_b_measured']}, index=df.index).to_csv(temp_b_file)
        
        # Create model
        model = tb.Model(id="building_model")
        
        thermal_sys = ParametricSystem(params, sample_time=600.0, id="ThermalSystem")
        radiator = tb.TimeSeriesInputSystem(
            df=pd.DataFrame({'radiatorHeat': df['Q_h']}, index=df.index), id="RadiatorInput")
        window = tb.TimeSeriesInputSystem(
            df=pd.DataFrame({'windowHeat': df['Q_r']}, index=df.index), id="WindowInput")
        outdoor = tb.TimeSeriesInputSystem(
            df=pd.DataFrame({'outdoorTemperature': df['T_out']}, index=df.index), id="OutdoorInput")
        temp_a_sensor = tb.SensorSystem(
            df=pd.read_csv(temp_a_file, index_col=0, parse_dates=True), id="TempASensor")
        temp_b_sensor = tb.SensorSystem(
            df=pd.read_csv(temp_b_file, index_col=0, parse_dates=True), id="TempBSensor")
        
        # Connect
        model.add_connection(radiator, thermal_sys, "value", "u", input_port_index=0)
        model.add_connection(window, thermal_sys, "value", "u", input_port_index=1)
        model.add_connection(outdoor, thermal_sys, "value", "u", input_port_index=2)
        model.add_connection(thermal_sys, temp_a_sensor, "y", "measuredValue", output_port_index=0)
        model.add_connection(thermal_sys, temp_b_sensor, "y", "measuredValue", output_port_index=3)
        
        model.load()
        return model


def _generate_data(filename):
    """Generate synthetic measurement data"""
    import os
    
    true_params = ThermalParameters()
    
    start_time = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=tz.gettz("Europe/Copenhagen"))
    end_time = datetime.datetime(2024, 1, 4, 0, 0, 0, tzinfo=tz.gettz("Europe/Copenhagen"))
    time_index = pd.date_range(start=start_time, end=end_time, freq='600s')[:-1]
    time_seconds = np.array([(t - start_time).total_seconds() for t in time_index])
    hour = (time_seconds / 3600) % 24
    day = time_seconds / 3600 / 24
    
    # Heating schedule
    Q_h = np.where((hour >= 6) & (hour < 9), 2000, 0)
    Q_h = np.where((hour >= 9) & (hour < 22), 1000, Q_h)
    Q_h = Q_h * (1 + 0.2 * day / 3)
    
    # Solar gains
    solar = np.where((hour >= 6) & (hour <= 18), 
                     np.sin(np.pi * (hour - 6) / 12), 0)
    Q_r = 600 * solar * (0.3 + 0.7 * day / 3)
    
    # Outdoor temperature
    T_base = 5 + 3*np.sin(2*np.pi*time_seconds/(3600*72))
    T_diurnal = 4*np.sin(2*np.pi*time_seconds/(3600*24) - np.pi/2)
    T_out = T_base + T_diurnal
    
    # Simulate
    model = _build_model(pd.DataFrame({
        'T_a_measured': np.zeros(len(time_index)),
        'T_b_measured': np.zeros(len(time_index)),
        'Q_h': Q_h, 'Q_r': Q_r, 'T_out': T_out
    }, index=time_index), true_params)
    
    simulator = tb.Simulator(model)
    simulator.simulate(step_size=600, start_time=start_time, end_time=end_time)
    
    thermal_sys = model.components["ThermalSystem"]
    states = thermal_sys.ss_model.output["y"].history
    
    # Add noise
    np.random.seed(42)
    T_a_measured = states[:, 0].detach().numpy() + np.random.normal(0, 0.35, len(time_index))
    T_b_measured = states[:, 3].detach().numpy() + np.random.normal(0, 0.35, len(time_index))
    
    # Save
    df = pd.DataFrame({
        'T_a_measured': T_a_measured,
        'T_b_measured': T_b_measured,
        'Q_h': Q_h,
        'Q_r': Q_r,
        'T_out': T_out
    }, index=time_index)
    df.to_csv(filename)


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    """Run the complete building energy management workflow"""
    print("\n" + "="*60)
    print("BUILDING ENERGY MANAGEMENT SYSTEM")
    print("Parameter Estimation and Optimal Control")
    print("="*60 + "\n")
    
    # Step 1: Load measurement data
    df = load_data()
    
    # Step 2: Estimate parameters from measurements
    estimated_params, model = estimate_parameters(df)
    
    # Step 3: Validate the calibrated model
    validate_model(df, model)
    
    # Step 4: Optimize heating control
    optimize_control(df, model)
    
    print("\n" + "="*60)
    print("WORKFLOW COMPLETED!")
    print("="*60)
    print("✓ Parameters estimated from data")
    print("✓ Model validated against measurements")
    print("✓ Optimal control achieved energy savings")
    print("\nKey Takeaways:")
    print("• RC networks model building thermal dynamics")
    print("• Parameter estimation calibrates models to reality")
    print("• Optimal control balances energy and comfort")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run the workflow
    main()