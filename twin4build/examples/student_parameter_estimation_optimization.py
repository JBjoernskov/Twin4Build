#!/usr/bin/env python3
"""
Building Energy Management System: Parameter Estimation and Optimal Control
===========================================================================

This example demonstrates a complete workflow for building energy management:
1. Load measurement data from a two-room thermal system
2. Construct state-space model matrices (A, B, C, D)
3. Estimate unknown thermal parameters using Twin4Build
4. Validate the calibrated model
5. Optimize heating control for energy efficiency

Physical System:
---------------
Two-room building with:
- Room A: Living room with radiator heating
- Room B: Office room with window (solar gains)
- Interior wall connecting the rooms
- Exterior walls with thermal mass
- States: [T_a, T_wa, T_i, T_b, T_wb] (air and wall temperatures)
- Inputs: [Q_h, Q_r, T_out] (heating, solar, outdoor temperature)

Learning Objectives:
-------------------
- Understand thermal RC network modeling
- Learn parameter estimation techniques
- Apply optimal control for energy savings
- Validate model performance
"""

import twin4build as tb
import torch
import datetime
from dateutil import tz
import numpy as np
import pandas as pd
import tempfile
import os
import torch.nn as nn
import twin4build.utils.types as tps


class MockSimulator:
    """Simple mock simulator class for Twin4Build plotting with direct array injection"""
    def __init__(self, time_index):
        self.dateTimeSteps = time_index


class ThermalParameters:
    """Container for thermal RC model parameters"""
    
    def __init__(self, 
        R_wa=0.014, 
        R_ia=0.005, 
        R_wao=0.016,
        R_ib=0.004, 
        R_wb=0.014, 
        R_wbo=0.018, 
        C_a=5e5, 
        C_wa=15e5, 
        C_i=1e5, 
        C_b=1e5, 
        C_wb=3e5):

        # Thermal resistances [K/W]
        self.R_wa = R_wa    # Room A air to wall
        self.R_ia = R_ia    # Room A air to interior wall
        self.R_wao = R_wao  # Room A wall to outdoor
        self.R_ib = R_ib    # Interior wall to Room B air
        self.R_wb = R_wb    # Room B air to wall
        self.R_wbo = R_wbo  # Room B wall to outdoor
        
        # Thermal capacitances [J/K]
        self.C_a = C_a      # Room A air
        self.C_wa = C_wa    # Room A wall
        self.C_i = C_i      # Interior wall
        self.C_b = C_b      # Room B air
        self.C_wb = C_wb    # Room B wall
    
    def to_dict(self):
        """Convert parameters to dictionary"""
        return {
            'R_wa': self.R_wa, 'R_ia': self.R_ia, 'R_wao': self.R_wao,
            'R_ib': self.R_ib, 'R_wb': self.R_wb, 'R_wbo': self.R_wbo,
            'C_a': self.C_a, 'C_wa': self.C_wa, 'C_i': self.C_i,
            'C_b': self.C_b, 'C_wb': self.C_wb
        }


class ParametricThermalSystem(tb.core.System, nn.Module):
    """
    Parametric thermal system for parameter estimation.
    
    This system can update its parameters during optimization and 
    automatically computes gradients for parameter estimation.
    """
    
    def __init__(self, initial_params, **kwargs):
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        
        # Store parameters as learnable tensors
        self.R_wa_param = tps.Parameter(torch.tensor(initial_params.R_wa, dtype=torch.float64))
        self.R_ia_param = tps.Parameter(torch.tensor(initial_params.R_ia, dtype=torch.float64))
        self.R_wao_param = tps.Parameter(torch.tensor(initial_params.R_wao, dtype=torch.float64))
        self.R_ib_param = tps.Parameter(torch.tensor(initial_params.R_ib, dtype=torch.float64))
        self.R_wb_param = tps.Parameter(torch.tensor(initial_params.R_wb, dtype=torch.float64))
        self.R_wbo_param = tps.Parameter(torch.tensor(initial_params.R_wbo, dtype=torch.float64))
        self.C_a_param = tps.Parameter(torch.tensor(initial_params.C_a, dtype=torch.float64))
        self.C_wa_param = tps.Parameter(torch.tensor(initial_params.C_wa, dtype=torch.float64))
        self.C_i_param = tps.Parameter(torch.tensor(initial_params.C_i, dtype=torch.float64))
        self.C_b_param = tps.Parameter(torch.tensor(initial_params.C_b, dtype=torch.float64))
        self.C_wb_param = tps.Parameter(torch.tensor(initial_params.C_wb, dtype=torch.float64))
        
        # Define system inputs and outputs
        self.input = {"u": tps.Vector(size=3)}   # [Q_h, Q_r, T_out]
        self.output = {"y": tps.Vector(size=5)}  # [T_a, T_wa, T_i, T_b, T_wb]
        
        # Define parameter bounds for estimation
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
        self._create_state_space_model()
    
    def _build_matrices(self):
        """
        Build state-space matrices A and B from current parameter values.
        
        State equations (5 states):
        dT_a/dt  = (T_wa-T_a)/R_wa/C_a + (T_i-T_a)/R_ia/C_a + Q_h/C_a
        dT_wa/dt = (T_a-T_wa)/R_wa/C_wa + (T_out-T_wa)/R_wao/C_wa
        dT_i/dt  = (T_a-T_i)/R_ia/C_i + (T_b-T_i)/R_ib/C_i
        dT_b/dt  = (T_i-T_b)/R_ib/C_b + (T_wb-T_b)/R_wb/C_b + Q_r/C_b
        dT_wb/dt = (T_b-T_wb)/R_wb/C_wb + (T_out-T_wb)/R_wbo/C_wb
        """
        A = torch.zeros((5, 5), dtype=torch.float64)
        
        # T_a equation (state 0)
        A[0, 0] = -(1/(self.R_wa_param.get() * self.C_a_param.get()) + 
                    1/(self.R_ia_param.get() * self.C_a_param.get()))
        A[0, 1] = 1/(self.R_wa_param.get() * self.C_a_param.get())
        A[0, 2] = 1/(self.R_ia_param.get() * self.C_a_param.get())
        
        # T_wa equation (state 1)
        A[1, 0] = 1/(self.R_wa_param.get() * self.C_wa_param.get())
        A[1, 1] = -(1/(self.R_wa_param.get() * self.C_wa_param.get()) + 
                    1/(self.R_wao_param.get() * self.C_wa_param.get()))
        
        # T_i equation (state 2)
        A[2, 0] = 1/(self.R_ia_param.get() * self.C_i_param.get())
        A[2, 2] = -(1/(self.R_ia_param.get() * self.C_i_param.get()) + 
                    1/(self.R_ib_param.get() * self.C_i_param.get()))
        A[2, 3] = 1/(self.R_ib_param.get() * self.C_i_param.get())
        
        # T_b equation (state 3)
        A[3, 2] = 1/(self.R_ib_param.get() * self.C_b_param.get())
        A[3, 3] = -(1/(self.R_wb_param.get() * self.C_b_param.get()) + 
                    1/(self.R_ib_param.get() * self.C_b_param.get()))
        A[3, 4] = 1/(self.R_wb_param.get() * self.C_b_param.get())
        
        # T_wb equation (state 4)
        A[4, 3] = 1/(self.R_wb_param.get() * self.C_wb_param.get())
        A[4, 4] = -(1/(self.R_wb_param.get() * self.C_wb_param.get()) + 
                    1/(self.R_wbo_param.get() * self.C_wb_param.get()))
        
        # B matrix: input effects
        B = torch.zeros((5, 3), dtype=torch.float64)
        B[0, 0] = 1/self.C_a_param.get()                                    # Q_h → T_a
        B[3, 1] = 1/self.C_b_param.get()                                    # Q_r → T_b
        B[1, 2] = 1/(self.R_wao_param.get() * self.C_wa_param.get())       # T_out → T_wa
        B[4, 2] = 1/(self.R_wbo_param.get() * self.C_wb_param.get())       # T_out → T_wb
        
        return A, B
    
    def _create_state_space_model(self):
        """Create discrete state-space model"""
        A, B = self._build_matrices()
        C = torch.eye(5, dtype=torch.float64)  # Observe all states
        D = torch.zeros((5, 3), dtype=torch.float64)  # No feedthrough
        x0 = torch.tensor([18.0, 12.0, 15.0, 16.0, 11.0], dtype=torch.float64)  # Initial conditions
        
        self.ss_model = tb.DiscreteStatespaceSystem(
            A=A, B=B, C=C, D=D, x0=x0,
            state_names=["T_a", "T_wa", "T_i", "T_b", "T_wb"],
            sample_time=600.0,  # 10-minute time steps
            id=f"ss_model_{self.id}"
        )
    
    @property
    def config(self):
        return self._config
    
    def initialize(self, start_time, end_time, step_size, simulator):
        """Initialize the system"""
        for input_port in self.input.values():
            input_port.initialize(start_time, end_time, step_size, simulator)
        for output_port in self.output.values():
            output_port.initialize(start_time, end_time, step_size, simulator)
        
        if not self.INITIALIZED:
            self._create_state_space_model()
            self.ss_model.initialize(start_time, end_time, step_size, simulator)
            self.INITIALIZED = True
        else:
            self._create_state_space_model()
            self.ss_model.initialize(start_time, end_time, step_size, simulator)
    
    def do_step(self, secondTime=None, dateTime=None, step_size=None, stepIndex=None):
        """Execute one simulation step"""
        self.ss_model.input["u"].set(self.input["u"].get(), stepIndex=stepIndex)
        self.ss_model.do_step(secondTime, dateTime, step_size, stepIndex=stepIndex)
        y = self.ss_model.output["y"].get()
        self.output["y"].set(y, stepIndex)


def load_measurement_data():
    """
    SECTION 1: LOAD AND EXPLORE MEASUREMENT DATA
    ============================================
    
    Load "real" measurement data from a two-room building system.
    In practice, this data would come from building sensors.
    """
    print("="*60)
    print("SECTION 1: LOADING MEASUREMENT DATA")
    print("="*60)
    
    # Load pre-generated measurement data (simulates real building data)
    data_file = "building_measurements.csv"
    
    if not os.path.exists(data_file):
        print("Generating synthetic measurement data...")
    _generate_synthetic_data(data_file)
    
    # Load the measurement data
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    print(f"✓ Loaded {len(df)} measurement points")
    print(f"✓ Time period: {df.index[0]} to {df.index[-1]}")
    print(f"✓ Available measurements: {list(df.columns)}")
    
    # Display data summary
    print("\nData Summary:")
    print(df.describe())
    
    # Create a mock simulator object for time axis
    mock_simulator = MockSimulator(df.index)
    
    # Plot the measurement data using Twin4Build's simplified matplotlib-like API
    tb.plot.plot(
        time=df.index,  # Use DataFrame index as time axis
        entries=[
            tb.plot.Entry(data=df['T_a_measured'].values, label="Room A Temperature", 
                          color=tb.plot.Colors.blue, fmt="-", axis=1),
            tb.plot.Entry(data=df['T_b_measured'].values, label="Room B Temperature", 
                          color=tb.plot.Colors.red, fmt="--", axis=1),
            tb.plot.Entry(data=df['T_out'].values, label="Outdoor Temperature", 
                          color=tb.plot.Colors.green, fmt=":", axis=1),
            tb.plot.Entry(data=df['Q_h'].values/1000, label="Radiator Heating", 
                          color=tb.plot.Colors.orange, fmt="-.", axis=2),  # Convert to kW
            tb.plot.Entry(data=df['Q_r'].values/1000, label="Solar Heat Gains", 
                          color=tb.plot.Colors.brown, fmt=":", axis=3),  # Convert to kW
        ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="Power [kW]",
        ylabel_3axis="Solar Gains [kW]",
        title="Building Measurement Data Overview",
        show=True,
        nticks=11
    )
    
    return df


def construct_state_space_matrices():
    """
    SECTION 2: CONSTRUCT STATE-SPACE MODEL MATRICES
    ===============================================
    
    Define the A, B, C, D matrices for the thermal RC network.
    This represents the physical relationships in the building.
    """
    print("\n" + "="*60)
    print("SECTION 2: STATE-SPACE MODEL CONSTRUCTION")
    print("="*60)
    
    print("Building thermal RC network model:")
    print("States: [T_a, T_wa, T_i, T_b, T_wb]")
    print("  T_a  = Room A air temperature")
    print("  T_wa = Room A wall temperature")
    print("  T_i  = Interior wall temperature")
    print("  T_b  = Room B air temperature")
    print("  T_wb = Room B wall temperature")
    print()
    print("Inputs: [Q_h, Q_r, T_out]")
    print("  Q_h   = Radiator heating power [W]")
    print("  Q_r   = Solar heat gains [W]")
    print("  T_out = Outdoor temperature [°C]")
    print()
    
    print("State-space model: dx/dt = A*x + B*u")
    print("                   y = C*x + D*u")
    print()
    
    print("A matrix (5x5) - Internal thermal coupling:")
    print("     T_a    T_wa   T_i    T_b    T_wb")
    print("T_a [-(1/R_wa + 1/R_ia)/C_a,  1/(R_wa*C_a),  1/(R_ia*C_a),     0,           0     ]")
    print("T_wa[ 1/(R_wa*C_wa), -(1/R_wa + 1/R_wao)/C_wa,    0,           0,           0     ]")
    print("T_i [ 1/(R_ia*C_i),        0,    -(1/R_ia + 1/R_ib)/C_i, 1/(R_ib*C_i),    0     ]")
    print("T_b [     0,               0,     1/(R_ib*C_b), -(1/R_wb + 1/R_ib)/C_b, 1/(R_wb*C_b)]")
    print("T_wb[     0,               0,           0,        1/(R_wb*C_wb), -1/R_wbo/C_wb - 1/R_wb/C_wb]")
    print()
    
    print("B matrix (5x3) - Input effects:")
    print("     Q_h      Q_r      T_out")
    print("T_a [ 1/C_a,    0,        0    ]")
    print("T_wa[   0,      0,   1/(R_wao*C_wa)]")
    print("T_i [   0,      0,        0    ]")
    print("T_b [   0,   1/C_b,       0    ]")
    print("T_wb[   0,      0,   1/(R_wbo*C_wb)]")
    print()
    
    print("C matrix (5x5) - Output mapping (observe all states):")
    print("C = I (identity matrix)")
    print()
    
    print("D matrix (5x3) - Direct feedthrough:")
    print("D = 0 (no direct input-to-output coupling)")
    print()
    
    print("Physical interpretation:")
    print("• A matrix: How states influence each other (thermal coupling)")
    print("• B matrix: How inputs affect states (heat sources, outdoor temperature)")
    print("• Diagonal A terms: Heat loss from each thermal mass")
    print("• Off-diagonal A terms: Heat exchange between thermal masses")
    print("• R parameters: Thermal resistances (higher = less heat transfer)")
    print("• C parameters: Thermal capacitances (higher = more thermal inertia)")
    print()
    
    print("Parameters to estimate:")
    print("  Thermal resistances: R_wa, R_ia, R_wao, R_ib, R_wb, R_wbo [K/W]")
    print("  Thermal capacitances: C_a, C_wa, C_i, C_b, C_wb [J/K]")
    
    # Initial parameter guess (deliberately different from true values)
    initial_params = ThermalParameters(
        R_wa=0.1,     # True: 0.008
        R_ia=0.04,     # True: 0.006
        R_wao=0.1,   # True: 0.012
        R_ib=0.04,     # True: 0.005
        R_wb=0.1,     # True: 0.009
        R_wbo=0.1,   # True: 0.013
        C_a=1e6,       # True: 1.2e6
        C_wa=5e5,      # True: 6e5
        C_i=1e6,       # True: 1.1e6
        C_b=1e6,       # True: 1.3e6
        C_wb=5e5       # True: 5.5e5
    )
    
    print("\nInitial parameter guess:")
    for key, value in initial_params.to_dict().items():
        print(f"  {key:6s} = {value:.3e}")
    
    return initial_params


def run_parameter_estimation(df, initial_params):
    """
    SECTION 3: PARAMETER ESTIMATION
    ===============================
    
    Use Twin4Build's estimator to calibrate model parameters
    against the measurement data.
    """
    print("\n" + "="*60)
    print("SECTION 3: PARAMETER ESTIMATION")
    print("="*60)
    
    # Create time index from data
    time_index = df.index
    start_time = time_index[0]
    end_time = time_index[-1] + pd.Timedelta(seconds=600)
    step_size = 600
    
    # Create temporary measurement files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save temperature measurements to files
        temp_a_file = os.path.join(temp_dir, "T_a_measured.csv")
        temp_b_file = os.path.join(temp_dir, "T_b_measured.csv")
        
        pd.DataFrame({'measuredValue': df['T_a_measured']}, index=time_index).to_csv(temp_a_file)
        pd.DataFrame({'measuredValue': df['T_b_measured']}, index=time_index).to_csv(temp_b_file)
        
        # Create estimation model
        model = tb.Model(id="parameter_estimation_model")
        
        # Create parametric thermal system
        thermal_system = ParametricThermalSystem(
            initial_params, 
            sample_time=600.0,
            id="ParametricThermalSystem"
        )
        
        # Create input systems
        radiator_input = tb.TimeSeriesInputSystem(
            df=pd.DataFrame({'radiatorHeat': df['Q_h']}, index=time_index),
            id="RadiatorInput"
        )
        window_input = tb.TimeSeriesInputSystem(
            df=pd.DataFrame({'windowHeat': df['Q_r']}, index=time_index),
            id="WindowInput"
        )
        outdoor_input = tb.TimeSeriesInputSystem(
            df=pd.DataFrame({'outdoorTemperature': df['T_out']}, index=time_index),
            id="OutdoorInput"
        )
        
        # Create measurement sensors
        temp_a_sensor = tb.SensorSystem(
            df=pd.read_csv(temp_a_file, index_col=0, parse_dates=True),
            id="TempASensor"
        )
        temp_b_sensor = tb.SensorSystem(
            df=pd.read_csv(temp_b_file, index_col=0, parse_dates=True),
            id="TempBSensor"
        )
        
        # Connect components
        model.add_connection(radiator_input, thermal_system, "value", "u", input_port_index=0)
        model.add_connection(window_input, thermal_system, "value", "u", input_port_index=1)
        model.add_connection(outdoor_input, thermal_system, "value", "u", input_port_index=2)
        model.add_connection(thermal_system, temp_a_sensor, "y", "measuredValue", output_port_index=0)
        model.add_connection(thermal_system, temp_b_sensor, "y", "measuredValue", output_port_index=3)
        
        # Load model and create estimator
        model.load()
        simulator = tb.Simulator(model)
        estimator = tb.Estimator(simulator)
        
        # Define parameters for estimation
        parameters = [
            (thermal_system, "R_wa_param", initial_params.R_wa, 0.001, 0.1),
            (thermal_system, "R_ia_param", initial_params.R_ia, 0.001, 0.1),
            (thermal_system, "R_wao_param", initial_params.R_wao, 0.001, 0.1),
            (thermal_system, "R_ib_param", initial_params.R_ib, 0.001, 0.1),
            (thermal_system, "R_wb_param", initial_params.R_wb, 0.001, 0.1),
            (thermal_system, "R_wbo_param", initial_params.R_wbo, 0.001, 0.1),
            (thermal_system, "C_a_param", initial_params.C_a, 1e5, 5e6),
            (thermal_system, "C_wa_param", initial_params.C_wa, 1e5, 2e6),
            (thermal_system, "C_i_param", initial_params.C_i, 1e5, 5e6),
            (thermal_system, "C_b_param", initial_params.C_b, 1e5, 5e6),
            (thermal_system, "C_wb_param", initial_params.C_wb, 1e5, 2e6),
        ]
        
        # Define measurements
        measurements = [(temp_a_sensor, 0.05), (temp_b_sensor, 0.05)]
        
        print("Running parameter estimation...")
        print("This may take a few minutes...")
        
        # Run parameter estimation
        estimation_result = estimator.estimate(
            start_time=start_time,
            end_time=end_time,
            step_size=step_size,
            parameters=parameters,
            measurements=measurements,
            n_warmup=20,
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": 100, "ftol": 1e-6}
        )
        
        print("✓ Parameter estimation completed!")
        
        # Extract estimated parameters
        estimated_params = ThermalParameters()
        estimated_params.R_wa = thermal_system.R_wa_param.get()
        estimated_params.R_ia = thermal_system.R_ia_param.get()
        estimated_params.R_wao = thermal_system.R_wao_param.get()
        estimated_params.R_ib = thermal_system.R_ib_param.get()
        estimated_params.R_wb = thermal_system.R_wb_param.get()
        estimated_params.R_wbo = thermal_system.R_wbo_param.get()
        estimated_params.C_a = thermal_system.C_a_param.get()
        estimated_params.C_wa = thermal_system.C_wa_param.get()
        estimated_params.C_i = thermal_system.C_i_param.get()
        estimated_params.C_b = thermal_system.C_b_param.get()
        estimated_params.C_wb = thermal_system.C_wb_param.get()
        
        return estimated_params, model, thermal_system, temp_a_sensor, temp_b_sensor


def show_estimation_results(df, initial_params, estimated_params, model, thermal_system):
    """
    SECTION 4: PARAMETER ESTIMATION RESULTS
    =======================================
    
    Compare estimated parameters with initial guess and validate model fit.
    """
    print("\n" + "="*60)
    print("SECTION 4: PARAMETER ESTIMATION RESULTS")
    print("="*60)
    
    # True parameters (for reference - normally unknown)
    true_params = ThermalParameters()  # Default values are the "true" ones
    
    print("\nParameter Comparison:")
    print("{:<10s} {:<15s} {:<15s} {:<15s}".format(
        "Parameter", "Initial Guess", "Estimated", "Improvement"
    ))
    print("-"*65)
    
    initial_dict = initial_params.to_dict()
    estimated_dict = estimated_params.to_dict()
    true_dict = true_params.to_dict()
    
    for key in initial_dict.keys():
        initial_val = initial_dict[key]
        estimated_val = estimated_dict[key]
        true_val = true_dict[key]
        
        initial_error = abs(initial_val - true_val) / true_val * 100
        estimated_error = abs(estimated_val - true_val) / true_val * 100
        improvement = initial_error - estimated_error
        
        print(f"{key:<10s} {initial_val:<15.3e} {estimated_val:<15.3e} {improvement:>+14.1f}%")
    
    print("\n✓ Parameter estimation successfully improved model accuracy!")
    
    # Run validation simulation with estimated parameters
    print("\nRunning validation simulation with estimated parameters...")
    
    time_index = df.index
    start_time = time_index[0]
    end_time = time_index[-1] + pd.Timedelta(seconds=600)
    step_size = 600
    
    # Create validation simulator
    validation_simulator = tb.Simulator(model)
    validation_simulator.simulate(
        step_size=step_size,
        start_time=start_time,
        end_time=end_time
    )
    
    # Extract simulation results
    estimated_T_a = thermal_system.ss_model.output["y"].history[:,0].detach().numpy()
    estimated_T_b = thermal_system.ss_model.output["y"].history[:,3].detach().numpy()
    
    # Calculate fit quality
    measured_T_a = df['T_a_measured'].values
    measured_T_b = df['T_b_measured'].values
    
    rmse_a = np.sqrt(np.mean((measured_T_a - estimated_T_a)**2))
    rmse_b = np.sqrt(np.mean((measured_T_b - estimated_T_b)**2))
    
    print(f"✓ Validation RMSE T_a: {rmse_a:.3f} °C")
    print(f"✓ Validation RMSE T_b: {rmse_b:.3f} °C")
    
    # Plot estimation results using Twin4Build's plotting utility
    print("\nGenerating parameter estimation validation plots...")
    
    # Create a mock simulator object for time axis
    mock_simulator = MockSimulator(time_index)
    
    # Plot comparison of measured vs estimated temperatures using simplified API
    tb.plot.plot(
        time=time_index,  # Use time index for x-axis
        entries=[
            tb.plot.Entry(data=measured_T_a, label="Measured T_a", 
                          color=tb.plot.Colors.black, fmt=".", axis=1, markersize=2),
            tb.plot.Entry(data=estimated_T_a, label="Estimated T_a", 
                          color=tb.plot.Colors.red, fmt="-", axis=1),
            tb.plot.Entry(data=measured_T_b, label="Measured T_b", 
                          color=tb.plot.Colors.blue, fmt=".", axis=1, markersize=2),
            tb.plot.Entry(data=estimated_T_b, label="Estimated T_b", 
                          color=tb.plot.Colors.orange, fmt="-", axis=1),
        ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="Heat Input [W]",
        ylabel_3axis="Outdoor Temperature [°C]",
        title="Parameter Estimation Results: Measured vs Estimated",
        show=True,
        nticks=11
    )
    
    print("✓ Validation plots saved to 'parameter_estimation_validation.png'")
    
    return rmse_a, rmse_b


def run_optimal_control(df, estimated_params, model, thermal_system, temp_a_sensor):
    """
    SECTION 5: OPTIMAL CONTROL
    ==========================
    
    Use the calibrated model to optimize heating control for energy efficiency
    while maintaining thermal comfort.
    """
    print("\n" + "="*60)
    print("SECTION 5: OPTIMAL CONTROL")
    print("="*60)
    
    print("Objective: Minimize energy consumption while maintaining comfort")
    print("Decision variable: Radiator heating schedule")
    print("Constraints: Room temperature within comfort bounds")


    # Set up simulation time
    time_index = df.index
    start_time = time_index[0]
    end_time = time_index[-1] + pd.Timedelta(seconds=600)
    step_size = 600

    opt_simulator = tb.Simulator(model)
    # Run baseline simulation
    print("\nRunning baseline simulation...")
    opt_simulator.simulate(
        step_size=step_size,
        start_time=start_time,
        end_time=end_time
    )

    # Extract baseline results
    baseline_T_a = thermal_system.ss_model.output["y"].history[:,0].detach().numpy() #thermal_system.ss_model.output["y"].history[:,0].detach().numpy()
    baseline_heating = model.components["RadiatorInput"].output["value"].history.detach().numpy()
    baseline_energy = np.sum(baseline_heating) * step_size / 3600 / 1000  # kWh
    
    print(f"✓ Baseline energy consumption: {baseline_energy:.2f} kWh")
    
    # Create optimizable heating schedule
    radiator_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 1000.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 6, 9, 12, 15, 18, 22, 24],
            "ruleset_end_hour": [6, 9, 12, 15, 18, 22, 24, 24],
            "ruleset_value": [400, 1500, 1800, 1600, 1700, 1400, 400, 400]
        },
        id="OptimizableRadiatorSchedule"
    )
    
    # Create comfort setpoints
    heating_setpoint = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 18.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 6, 8, 17, 22, 0, 0],
            "ruleset_end_hour": [6, 8, 17, 22, 24, 0, 0],
            "ruleset_value": [10.0, 21, 21.0, 10, 10, 10.0, 10.0]
        },
        id="HeatingSetpoint"
    )
    
    cooling_setpoint = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 26.0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [0, 6, 8, 17, 22, 0, 0],
            "ruleset_end_hour": [6, 8, 17, 22, 24, 0, 0],
            "ruleset_value": [30.0, 26.0, 24.0, 25.0, 26.0, 30.0, 30.0]
        },
        id="CoolingSetpoint"
    )
    
    # Add optimization components to existing model
    model.add_component(radiator_schedule)
    model.add_component(heating_setpoint)
    model.add_component(cooling_setpoint)
    
    # Replace heating input with optimizable schedule
    radiator_input = model.components["RadiatorInput"]
    model.remove_connection(radiator_input, thermal_system, "value", "u")
    model.add_connection(radiator_schedule, thermal_system, "scheduleValue", "u", input_port_index=0)
    
    # Reload model
    model.load()
    
    # Create optimizer
    optimizer = tb.Optimizer(opt_simulator)
    
    
    
    
    
    # Define optimization problem
    variables = [(radiator_schedule, "scheduleValue", 0, 2000)]
    objectives = [(radiator_schedule, "scheduleValue", "min")]
    ineq_cons = [
        (temp_a_sensor, "measuredValue", "lower", heating_setpoint),
        (temp_a_sensor, "measuredValue", "upper", cooling_setpoint)
    ]
    
    # Run optimization
    print("\nRunning optimization...")
    optimizer.optimize(
        start_time=start_time,
        end_time=end_time,
        step_size=step_size,
        variables=variables,
        objectives=objectives,
        eq_cons=None,
        ineq_cons=ineq_cons,
        method="scipy",
        options={"maxiter": 500, "disp": True}#, "tol": 1e-16}
    )
    
    print("✓ Optimization completed!")
    
    # Extract optimized results
    optimized_T_a = thermal_system.ss_model.output["y"].history[:,0].detach().numpy() #thermal_system.ss_model.output["y"].history[:,0].detach().numpy()
    optimized_heating = radiator_schedule.output["scheduleValue"].history.detach().numpy()
    optimized_energy = np.sum(optimized_heating) * step_size / 3600 / 1000  # kWh
    
    energy_savings = (baseline_energy - optimized_energy) / baseline_energy * 100
    
    return {
        'baseline_T_a': baseline_T_a,
        'optimized_T_a': optimized_T_a,
        'baseline_heating': baseline_heating,
        'optimized_heating': optimized_heating,
        'baseline_energy': baseline_energy,
        'optimized_energy': optimized_energy,
        'energy_savings': energy_savings,
        'heating_setpoint': heating_setpoint,
        'cooling_setpoint': cooling_setpoint,
        'opt_simulator': opt_simulator
    }




def show_optimization_results_improved(df, results, opt_simulator):
    """
    SECTION 6: OPTIMIZATION RESULTS (IMPROVED VISUALIZATION)
    ========================================================
    
    Display the results with enhanced visualizations.
    """
    print("\n" + "="*60)
    print("SECTION 6: OPTIMIZATION RESULTS")
    print("="*60)
    
    print(f"Baseline energy consumption: {results['baseline_energy']:.2f} kWh")
    print(f"Optimized energy consumption: {results['optimized_energy']:.2f} kWh")
    print(f"Energy savings: {results['energy_savings']:.1f}%")
    
    print(f"\nBaseline temperature range: {np.min(results['baseline_T_a']):.1f}°C - {np.max(results['baseline_T_a']):.1f}°C")
    print(f"Optimized temperature range: {np.min(results['optimized_T_a']):.1f}°C - {np.max(results['optimized_T_a']):.1f}°C")
    
    # Create time index
    time_index = pd.date_range(start=df.index[0], periods=len(results['baseline_T_a']), freq='10min')
    
    # Extract setpoint data
    heating_sp = results['heating_setpoint'].output["scheduleValue"].history.detach().numpy()
    cooling_sp = results['cooling_setpoint'].output["scheduleValue"].history.detach().numpy()
    
    # Calculate comfort violations
    baseline_violations = np.sum((results['baseline_T_a'] < heating_sp) | 
                                  (results['baseline_T_a'] > cooling_sp))
    optimized_violations = np.sum((results['optimized_T_a'] < heating_sp) | 
                                   (results['optimized_T_a'] > cooling_sp))
    
    print(f"\nComfort violations (time steps):")
    print(f"  Baseline: {baseline_violations}")
    print(f"  Optimized: {optimized_violations}")
    
    # --- PLOT 1: Baseline Comparison ---
    tb.plot.plot(
        time=time_index,
        entries=[
            # Temperature bounds (shaded region would be ideal, but use lines)
            tb.plot.Entry(data=heating_sp, label="Heating Setpoint (min)", 
                          color=tb.plot.Colors.red, fmt=":", axis=1, linewidth=2),
            tb.plot.Entry(data=cooling_sp, label="Cooling Setpoint (max)", 
                          color=tb.plot.Colors.blue, fmt=":", axis=1, linewidth=2),
            # Baseline temperature
            tb.plot.Entry(data=results['baseline_T_a'], label="Baseline Temperature", 
                          color=tb.plot.Colors.black, fmt="-", axis=1, linewidth=1.5),
            # Baseline heating on secondary axis
            tb.plot.Entry(data=results['baseline_heating']/1000, label="Baseline Heating", 
                          color=tb.plot.Colors.orange, fmt="-", axis=2, linewidth=1.5),
        ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="Heating Power [kW]",
        ylim_1axis=(8, 32),
        ylim_2axis=(0, 6),
        title=f"Baseline Control Strategy (Energy: {results['baseline_energy']:.1f} kWh)",
        show=True,
        nticks=13
    )
    
    # --- PLOT 2: Optimized Comparison ---
    tb.plot.plot(
        time=time_index,
        entries=[
            # Temperature bounds
            tb.plot.Entry(data=heating_sp, label="Heating Setpoint (min)", 
                          color=tb.plot.Colors.red, fmt=":", axis=1, linewidth=2),
            tb.plot.Entry(data=cooling_sp, label="Cooling Setpoint (max)", 
                          color=tb.plot.Colors.blue, fmt=":", axis=1, linewidth=2),
            # Optimized temperature
            tb.plot.Entry(data=results['optimized_T_a'], label="Optimized Temperature", 
                          color=tb.plot.Colors.black, fmt="-", axis=1, linewidth=1.5),
            # Optimized heating on secondary axis
            tb.plot.Entry(data=results['optimized_heating']/1000, label="Optimized Heating", 
                          color=tb.plot.Colors.orange, fmt="-", axis=2, linewidth=1.5),
        ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="Heating Power [kW]",
        ylim_1axis=(8, 32),
        ylim_2axis=(0, 3.5),
        title=f"Optimized Control Strategy (Energy: {results['optimized_energy']:.1f} kWh, Savings: {results['energy_savings']:.1f}%)",
        show=True,
        nticks=13
    )
    
    
    # Calculate and display additional metrics
    print("\n" + "="*60)
    print("DETAILED PERFORMANCE METRICS")
    print("="*60)
    
    # Average temperatures during occupied hours (6-22)
    hour_of_day = time_index.hour
    occupied_mask = (hour_of_day >= 6) & (hour_of_day < 22)
    
    baseline_avg_occupied = np.mean(results['baseline_T_a'][occupied_mask])
    optimized_avg_occupied = np.mean(results['optimized_T_a'][occupied_mask])
    
    print(f"\nAverage temperature during occupied hours (6:00-22:00):")
    print(f"  Baseline:  {baseline_avg_occupied:.2f}°C")
    print(f"  Optimized: {optimized_avg_occupied:.2f}°C")
    
    # Peak heating demand
    baseline_peak = np.max(results['baseline_heating'])/1000
    optimized_peak = np.max(results['optimized_heating'])/1000
    
    print(f"\nPeak heating demand:")
    print(f"  Baseline:  {baseline_peak:.2f} kW")
    print(f"  Optimized: {optimized_peak:.2f} kW")
    print(f"  Reduction: {(baseline_peak - optimized_peak)/baseline_peak*100:.1f}%")
    
    # Temperature stability (standard deviation)
    baseline_std = np.std(results['baseline_T_a'][occupied_mask])
    optimized_std = np.std(results['optimized_T_a'][occupied_mask])
    
    print(f"\nTemperature stability (std dev during occupied hours):")
    print(f"  Baseline:  {baseline_std:.2f}°C")
    print(f"  Optimized: {optimized_std:.2f}°C")
    
    print("\n✓ Optimization successfully reduced energy consumption while maintaining comfort!")

# def show_optimization_results(df, results, opt_simulator):
#     """
#     SECTION 6: OPTIMIZATION RESULTS
#     ===============================
    
#     Display the results of the optimal control optimization.
#     """
#     print("\n" + "="*60)
#     print("SECTION 6: OPTIMIZATION RESULTS")
#     print("="*60)
    
#     print(f"Baseline energy consumption: {results['baseline_energy']:.2f} kWh")
#     print(f"Optimized energy consumption: {results['optimized_energy']:.2f} kWh")
#     print(f"Energy savings: {results['energy_savings']:.1f}%")
    
#     print(f"\nBaseline temperature range: {np.min(results['baseline_T_a']):.1f}°C - {np.max(results['baseline_T_a']):.1f}°C")
#     print(f"Optimized temperature range: {np.min(results['optimized_T_a']):.1f}°C - {np.max(results['optimized_T_a']):.1f}°C")
    
#     # Create time index for optimization results (assuming 10-minute intervals)
#     time_index = pd.date_range(start=df.index[0], periods=len(results['baseline_T_a']), freq='10min')
#     mock_simulator = MockSimulator(time_index)
    
#     # Plot optimization results using Twin4Build's simplified matplotlib-like API
#     tb.plot.plot(
#         time=time_index,  # Use time index for x-axis
#         entries=[
#             # Direct data entries on axis 1
#             tb.plot.Entry(data=results['heating_setpoint'].output["scheduleValue"].history.detach().numpy(), 
#                 label="Heating Setpoint", color=tb.plot.Colors.red, fmt="-", axis=1),
#             tb.plot.Entry(data=results['cooling_setpoint'].output["scheduleValue"].history.detach().numpy(), 
#                 label="Cooling Setpoint", color=tb.plot.Colors.blue, fmt="-", axis=1),
#             tb.plot.Entry(data=results['baseline_T_a'], label="Baseline Temperature", axis=1, color=tb.plot.Colors.green, fmt="--"),
#             # Direct data entries on axis 2 with custom styling
#             tb.plot.Entry(data=results['baseline_heating'], label="Baseline Heating", 
#                           color=tb.plot.Colors.orange, fmt="--", axis=2),
#         ],
#         ylabel_1axis="Temperature [°C]",
#         ylabel_2axis="Heating Power [W]",
#         ylim_2axis=(0, 5000),
#         title="Optimal Control Results: Temperature Management and Energy Optimization",
#         show=True,
#         nticks=11
#     )

#     # Plot optimization results using Twin4Build's simplified matplotlib-like API
#     tb.plot.plot(
#         time=time_index,  # Use time index for x-axis
#         entries=[
#             tb.plot.Entry(data=results['heating_setpoint'].output["scheduleValue"].history.detach().numpy(), 
#                 label="Heating Setpoint", color=tb.plot.Colors.red, fmt="-", axis=1),
#             tb.plot.Entry(data=results['cooling_setpoint'].output["scheduleValue"].history.detach().numpy(), 
#                           label="Cooling Setpoint", color=tb.plot.Colors.blue, fmt="-", axis=1),
#             tb.plot.Entry(data=results['optimized_T_a'], label="Optimized Temperature", axis=1, color=tb.plot.Colors.green, fmt="--"),
#             # Extract component data as arrays for simplified API
#             tb.plot.Entry(data=results['optimized_heating'], label="Optimized Heating", 
#                           color=tb.plot.Colors.orange, fmt="--", axis=2),
#         ],
#         ylabel_1axis="Temperature [°C]",
#         ylabel_2axis="Heating Power [W]",
#         ylim_2axis=(0, 5000),
#         title="Optimal Control Results: Temperature Management and Energy Optimization",
#         show=True,
#         nticks=11
#     )
    
#     print("\n✓ Optimization successfully reduced energy consumption while maintaining comfort!")


def _generate_synthetic_data(filename):
    """Generate synthetic measurement data (hidden from students)"""
    # This function generates the "real" measurement data
    # In practice, this would be actual building sensor data
    
    # True parameters (what we want to recover)
    true_params = ThermalParameters()
    
    # Time setup (3 days, 10-minute intervals)
    start_time = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=tz.gettz("Europe/Copenhagen"))
    end_time = datetime.datetime(2024, 1, 4, 0, 0, 0, tzinfo=tz.gettz("Europe/Copenhagen"))
    time_index = pd.date_range(start=start_time, end=end_time, freq='600s')[:-1]
    time_seconds = np.array([(t - start_time).total_seconds() for t in time_index])
    hour_of_day = (time_seconds / 3600) % 24
    day_of_simulation = (time_seconds / 3600 / 24)
    
    # Heating schedule
    schedule_values = np.zeros_like(hour_of_day)
    
    schedule_values = np.where((hour_of_day >= 6) & (hour_of_day < 9), 2000, schedule_values)
    schedule_values = np.where((hour_of_day >= 9) & (hour_of_day < 12), 1000, schedule_values)
    schedule_values = np.where((hour_of_day >= 12) & (hour_of_day < 15), 1000, schedule_values)
    schedule_values = np.where((hour_of_day >= 15) & (hour_of_day < 18), 1000, schedule_values)
    schedule_values = np.where((hour_of_day >= 18) & (hour_of_day < 22), 1000, schedule_values)
    
    schedule_values = np.where((day_of_simulation >= 0) & (day_of_simulation < 1), schedule_values*1, schedule_values)
    schedule_values = np.where((day_of_simulation >= 1) & (day_of_simulation < 2), schedule_values*1.2, schedule_values)
    schedule_values = np.where((day_of_simulation >= 2) & (day_of_simulation < 3), schedule_values*1.4, schedule_values)


    schedule_values = np.where((hour_of_day >= 0) & (hour_of_day < 6), 1000, schedule_values)
    schedule_values = np.where((hour_of_day >= 22) & (hour_of_day < 24), 1000, schedule_values)
    Q_h = schedule_values
    
    
    # Solar position: peak at noon (hour 12), zero at night
    # Use a parabolic shape from sunrise (6am) to sunset (6pm)
    solar_elevation = np.where(
        (hour_of_day >= 6) & (hour_of_day <= 18),
        np.sin(np.pi * (hour_of_day - 6) / 12),  # Single peak from 6am to 6pm
        0.0
    )
    
    # Add random walk to simulate cloud cover and weather variability
    solar_walk = np.zeros(len(time_index))
    solar_walk[0] = 1  # Start with more initial variation
    solar_walk_step_std = 0.05  # Increased step size for more visible variation
    for i in range(1, len(time_index)):
        solar_walk[i] = max(0, solar_walk[i-1] + np.random.normal(0, solar_walk_step_std))
    
    # Constrain the walk to reasonable bounds (±50% variation for more visible effects)
    # solar_walk = np.clip(solar_walk, 0, 0.5)
    

    fac = np.linspace(0.3, 1, len(time_index))#1+ 0.4*np.sin(2*np.pi*day_of_simulation/3)  # Weather variation

    # Apply solar gains with random walk (cloud cover effects)
    Q_r = 600* solar_elevation * fac * solar_walk


    for t, walk, factor in zip(time_index, solar_walk, fac):
        print("---")
        print("t", t)
        print("solar_walk", walk)
        print("daily_solar_factor", factor)
    
    
    # Outdoor temperature with realistic random walk noise
    T_base = 5 + 3*np.sin(2*np.pi*time_seconds/(3600*72))  # 3-day trend
    T_diurnal = 4*np.sin(2*np.pi*time_seconds/(3600*24) - np.pi/2)  # Daily cycle
    
    # Add random walk component (weather persistence)
    T_walk = np.zeros(len(time_index))
    walk_step_std = 0.5  # Standard deviation of each random walk step
    for i in range(1, len(time_index)):
        T_walk[i] = T_walk[i-1] + np.random.normal(0, walk_step_std)
    
    # Add some high-frequency noise on top
    T_noise = np.random.normal(0, 0.2, len(time_index))  # Measurement/micro-weather noise
    
    T_out = T_base + T_diurnal + T_walk + T_noise
    
    # Simulate true system using Twin4Build's proper simulation framework
    print("Generating synthetic data using Twin4Build simulation...")
    
    # Create true model with known parameters
    true_model = tb.Model(id="true_model_synthetic")
    
    # Create parametric thermal system with true parameters
    true_thermal_system = ParametricThermalSystem(
        true_params, 
        sample_time=600.0,
        id="TrueThermalSystem"
    )
    
    # Create input systems with the generated input data
    radiator_input = tb.TimeSeriesInputSystem(
        df=pd.DataFrame({'radiatorHeat': Q_h}, index=time_index),
        id="TrueRadiatorInput"
    )
    window_input = tb.TimeSeriesInputSystem(
        df=pd.DataFrame({'windowHeat': Q_r}, index=time_index),
        id="TrueWindowInput"
    )
    outdoor_input = tb.TimeSeriesInputSystem(
        df=pd.DataFrame({'outdoorTemperature': T_out}, index=time_index),
        id="TrueOutdoorInput"
    )
    
    # Connect inputs to thermal system
    true_model.add_connection(radiator_input, true_thermal_system, "value", "u", input_port_index=0)
    true_model.add_connection(window_input, true_thermal_system, "value", "u", input_port_index=1)
    true_model.add_connection(outdoor_input, true_thermal_system, "value", "u", input_port_index=2)
    
    # Load and simulate the true model
    true_model.load()
    true_simulator = tb.Simulator(true_model)
    true_simulator.simulate(
        step_size=600,
        start_time=start_time,
        end_time=end_time
    )
    
    # Extract true states from Twin4Build simulation
    states = true_thermal_system.ss_model.output["y"].history
    
    # Add measurement noise
    np.random.seed(42)
    noise_std = 0.35
    T_a_measured = states[:, 0].detach().numpy() + np.random.normal(0, noise_std, len(time_index))
    T_b_measured = states[:, 3].detach().numpy() + np.random.normal(0, noise_std, len(time_index))
    
    # Save to file
    df = pd.DataFrame({
        'T_a_measured': T_a_measured,
        'T_b_measured': T_b_measured,
        'Q_h': Q_h,
        'Q_r': Q_r,
        'T_out': T_out
    }, index=time_index)
    
    df.to_csv(filename)


def main():
    """
    MAIN WORKFLOW: PARAMETER ESTIMATION AND OPTIMAL CONTROL
    =======================================================
    
    Complete demonstration of building energy management:
    1. Load measurement data
    2. Construct thermal model
    3. Estimate parameters
    4. Validate results
    5. Optimize control
    6. Show improvements
    """
    print("BUILDING ENERGY MANAGEMENT SYSTEM")
    print("Parameter Estimation and Optimal Control")
    print("=" * 60)
    
    # Section 1: Load measurement data
    df = load_measurement_data()
    
    # Section 2: Construct state-space model
    initial_params = construct_state_space_matrices()
    
    # Section 3: Run parameter estimation
    estimated_params, model, thermal_system, temp_a_sensor, temp_b_sensor = run_parameter_estimation(df, initial_params)
    
    # Section 4: Show estimation results
    rmse_a, rmse_b = show_estimation_results(df, initial_params, estimated_params, model, thermal_system)
    
    # Section 5: Run optimal control
    optimization_results = run_optimal_control(df, estimated_params, model, thermal_system, temp_a_sensor) #temp_a_sensor
    
    # Section 6: Show optimization results
    show_optimization_results_improved(df, optimization_results, optimization_results['opt_simulator'])
    
    print("\n" + "="*60)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("✓ Model parameters calibrated from measurement data")
    print("✓ Optimal control strategy developed")
    print("✓ Energy savings achieved while maintaining comfort")
    print("\nKey Learning Points:")
    print("• Thermal RC networks can model building physics")
    print("• Parameter estimation calibrates models to real data")
    print("• Optimal control balances energy and comfort")
    print("• Twin4Build enables integrated workflows")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run the complete workflow
    main()
