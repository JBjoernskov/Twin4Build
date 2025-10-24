#!/usr/bin/env python3
"""
Building Energy Management: Parameter Estimation and Optimal Control
====================================================================

STUDENT EXERCISE: Complete the state-space matrices A, B, C, D

Learning Objectives:
- Derive state-space matrices from RC network
- Estimate thermal parameters from measurements
- Apply optimal control for energy savings

Physical System: Two-room building
- States: x = [T_a, T_wa, T_i, T_b, T_wb] (temperatures)
- Inputs: u = [Q_h, Q_r, T_out] (heating, solar, outdoor temp)
- Outputs: y = [T_a, T_wa, T_i, T_b, T_wb] (all temperatures measured)

State-space form: dx/dt = A*x + B*u
                  y = C*x + D*u
"""

import twin4build as tb
import torch
import numpy as np
import pandas as pd
import datetime
import time
from dateutil import tz


# ============================================================================
# STUDENT TASK: COMPLETE THE STATE-SPACE MATRICES
# ============================================================================

def build_state_space_matrices(R_wa, R_ia, R_wao, R_ib, R_wb, R_wbo,
                                C_a, C_wa, C_i, C_b, C_wb):
    """
    TODO: Complete the A, B, C, D matrices based on your exercise solution
    
    Given parameters:
    - Thermal resistances: R_wa, R_ia, R_wao, R_ib, R_wb, R_wbo [K/W]
    - Thermal capacitances: C_a, C_wa, C_i, C_b, C_wb [J/K]
    
    State vector: x = [T_a, T_wa, T_i, T_b, T_wb]
    Input vector: u = [Q_h, Q_r, T_out]
    Output vector: y = [T_a, T_wa, T_i, T_b, T_wb]
    
    Hint: From the RC network in your exercise:
    - dT_a/dt = (T_wa - T_a)/(R_wa*C_a) + (T_i - T_a)/(R_ia*C_a) + Q_h/C_a
    - dT_wa/dt = (T_a - T_wa)/(R_wa*C_wa) + (T_out - T_wa)/(R_wao*C_wa)
    - ... (continue for T_i, T_b, T_wb)
    """
    
    # Initialize matrices
    A = torch.zeros((5, 5), dtype=torch.float64)
    B = torch.zeros((5, 3), dtype=torch.float64)
    C = torch.zeros((5, 5), dtype=torch.float64)
    D = torch.zeros((5, 3), dtype=torch.float64)



    A[0, 0] = -(1/(R_wa * C_a) + 1/(R_ia * C_a))
    A[0, 1] = 1/(R_wa * C_a)
    A[0, 2] = 1/(R_ia * C_a)
    A[1, 0] = 1/(R_wa * C_wa)
    A[1, 1] = -(1/(R_wa * C_wa) + 
                1/(R_wao * C_wa))
    A[2, 0] = 1/(R_ia * C_i)
    A[2, 2] = -(1/(R_ia * C_i) + 
                1/(R_ib * C_i))
    A[2, 3] = 1/(R_ib * C_i)
    A[3, 2] = 1/(R_ib * C_b)
    A[3, 3] = -(1/(R_wb * C_b) + 
                1/(R_ib * C_b))
    A[3, 4] = 1/(R_wb * C_b)
    A[4, 3] = 1/(R_wb * C_wb)
    A[4, 4] = -(1/(R_wb * C_wb) + 
                1/(R_wbo * C_wb))
    
    B[0, 0] = 1/C_a
    B[3, 1] = 1/C_b
    B[1, 2] = 1/(R_wao * C_wa)
    B[4, 2] = 1/(R_wbo * C_wb)

    C = torch.eye(5, dtype=torch.float64)
    
    # ========================================================================
    # TODO: Fill in the A matrix (5x5) - Internal thermal coupling
    # ========================================================================
    # Row 0: dT_a/dt equation
    # A[0, 0] = ???  # Effect of T_a on itself
    # A[0, 1] = ???  # Effect of T_wa on T_a
    # A[0, 2] = ???  # Effect of T_i on T_a
    
    # Row 1: dT_wa/dt equation
    # A[1, 0] = ???  # Effect of T_a on T_wa
    # A[1, 1] = ???  # Effect of T_wa on itself
    
    # Row 2: dT_i/dt equation
    # A[2, 0] = ???  # Effect of T_a on T_i
    # A[2, 2] = ???  # Effect of T_i on itself
    # A[2, 3] = ???  # Effect of T_b on T_i
    
    # Row 3: dT_b/dt equation
    # A[3, 2] = ???  # Effect of T_i on T_b
    # A[3, 3] = ???  # Effect of T_b on itself
    # A[3, 4] = ???  # Effect of T_wb on T_b
    
    # Row 4: dT_wb/dt equation
    # A[4, 3] = ???  # Effect of T_b on T_wb
    # A[4, 4] = ???  # Effect of T_wb on itself
    
    # TODO: Fill in your A matrix based on the RC network equations
    # Example for A[0,0]: A[0, 0] = -(1/(R_wa * C_a) + 1/(R_ia * C_a))
    
    
    # ========================================================================
    # TODO: Fill in the B matrix (5x3) - Input effects
    # ========================================================================
    # Columns: [Q_h, Q_r, T_out]
    
    # B[0, 0] = ???  # Effect of Q_h on T_a
    # B[1, 2] = ???  # Effect of T_out on T_wa
    # B[3, 1] = ???  # Effect of Q_r on T_b
    # B[4, 2] = ???  # Effect of T_out on T_wb
    
    # TODO: Fill in your B matrix
    # Which inputs affect which states?
    
    
    # ========================================================================
    # TODO: Fill in the C matrix (5x5) - Output mapping
    # ========================================================================
    # We observe all states directly, so C should be...?
    
    # TODO: What is C if we observe all states directly?
    
    
    # ========================================================================
    # TODO: Fill in the D matrix (5x3) - Direct feedthrough
    # ========================================================================
    # Is there any direct input-to-output path (bypassing states)?
    
    # TODO: Is there direct input-to-output coupling (no states involved)?
    
    
    return A, B, C, D


# ============================================================================
# VERIFICATION: Check your matrices
# ============================================================================

def verify_matrices():
    """Verify that your matrices have the correct structure"""
    print("="*60)
    print("VERIFYING YOUR STATE-SPACE MATRICES")
    print("="*60)
    
    # Test with example parameters
    test_params = ThermalParameters(
        R_wa=0.014, R_ia=0.005, R_wao=0.016,
        R_ib=0.004, R_wb=0.014, R_wbo=0.018,
        C_a=5e5, C_wa=15e5, C_i=1e5, C_b=1e5, C_wb=3e5
    )
    
    A, B, C, D = build_state_space_matrices(
        test_params.R_wa, test_params.R_ia, test_params.R_wao,
        test_params.R_ib, test_params.R_wb, test_params.R_wbo,
        test_params.C_a, test_params.C_wa, test_params.C_i,
        test_params.C_b, test_params.C_wb
    )
    
    # Check dimensions and content
    checks_passed = 0
    total_checks = 7
    
    print("\nDimension checks:")
    if A.shape == (5, 5):
        print("✓ A matrix is 5x5")
        checks_passed += 1
    else:
        print("✗ A matrix should be 5x5, got", A.shape)
    
    if B.shape == (5, 3):
        print("✓ B matrix is 5x3")
        checks_passed += 1
    else:
        print("✗ B matrix should be 5x3, got", B.shape)
    
    if C.shape == (5, 5):
        print("✓ C matrix is 5x5")
        checks_passed += 1
    else:
        print("✗ C matrix should be 5x5, got", C.shape)
    
    if D.shape == (5, 3):
        print("✓ D matrix is 5x3")
        checks_passed += 1
    else:
        print("✗ D matrix should be 5x3, got", D.shape)
    
    # Check if matrices have been filled in (not all zeros)
    print("\nContent checks:")
    if torch.any(A != 0):
        print("✓ A matrix has been filled in")
        checks_passed += 1
    else:
        print("✗ A matrix appears empty - please fill it in")
    
    if torch.any(B != 0):
        print("✓ B matrix has been filled in")
        checks_passed += 1
    else:
        print("✗ B matrix appears empty - please fill it in")

    if torch.any(C != 0):
        print("✓ C matrix has been filled in")
        checks_passed += 1
    else:
        print("✗ C matrix appears empty - please fill it in")




    
    print(f"\nPassed {checks_passed}/{total_checks} checks")
    
    if checks_passed == total_checks:
        print("✓ All basic checks passed! Your matrices look good.")
        print("\nYour A matrix:")
        print(A)
        print("\nYour B matrix:")
        print(B)
        print("\nYour C matrix:")
        print(C)
        print("\nYour D matrix:")
        print(D)
        return True
    else:
        print("\n⚠ Please complete the matrices before proceeding.")
        print("Edit the build_state_space_matrices() function.")
        return False


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
    print("\n" + "="*60)
    print("STEP 1: LOADING MEASUREMENT DATA")
    print("="*60)
    
    # Generate synthetic data (simulates real sensor data)
    data_file = "building_measurements.csv"
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    print(f"✓ Loaded {len(df)} measurements")
    print(f"✓ Period: {df.index[0]} to {df.index[-1]}")
    print(f"✓ Columns: {list(df.columns)}\n")
    
    # Plot the data
    fig, axes = tb.plot.plot(
        time=df.index,
        entries=[
            tb.plot.Entry(data=df['T_a_measured'].values, 
                         label=r"$T_a$", color=tb.plot.Colors.blue, fmt="-", axis=1),
            tb.plot.Entry(data=df['T_b_measured'].values, 
                         label=r"$T_b$", color=tb.plot.Colors.red, fmt="--", axis=1),
            tb.plot.Entry(data=df['Q_h'].values/1000, 
                         label=r"$\dot{Q}_h$", color=tb.plot.Colors.orange, fmt="-", axis=2),
        ],
        ylabel_1axis=r"Temperature $[^\circ$ C$]$",
        ylabel_2axis="Power [kW]",
        show=False,
        nticks=11
    )
    fig.savefig("bems_room_measurements.png", dpi=300)
    # Plot the data
    fig, axes = tb.plot.plot(
        time=df.index,
        entries=[
            tb.plot.Entry(data=df['T_out'].values, 
                         label=r"$T_{out}$", color=tb.plot.Colors.blue, fmt="-", axis=1),
            tb.plot.Entry(data=df['Q_r'].values/1000, 
                         label=r"$\dot{Q}_r$", color=tb.plot.Colors.orange, fmt="-", axis=2),
        ],
        ylabel_1axis=r"Temperature $[^\circ$ C$]$",
        ylabel_2axis="Solar Gains [kW]",
        show=False,
        align_zero=False,
        nticks=11
    )
    fig.savefig("bems_outdoor_measurements.png", dpi=300)
    
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
    initial = ThermalParameters(
        R_wa=0.01, R_ia=0.002, R_wao=0.01,
        R_ib=0.002, R_wb=0.01, R_wbo=0.01,
        C_a=1e6, C_wa=5e5, C_i=1e6, C_b=1e6, C_wb=5e5
    )

    # R_wa=0.014, R_ia=0.005, R_wao=0.016,
    #              R_ib=0.004, R_wb=0.014, R_wbo=0.018,
    #              C_a=5e5, C_wa=15e5, C_i=1e5, C_b=1e5, C_wb=3e5):
    
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
    estimated_T_a = thermal_sys.ss_model.output["y"].history[:,:,0].detach().numpy()
    estimated_T_b = thermal_sys.ss_model.output["y"].history[:,:,3].detach().numpy()
    estimated_T_a = estimated_T_a.reshape((estimated_T_a.shape[0]*estimated_T_a.shape[1]))
    estimated_T_b = estimated_T_b.reshape((estimated_T_b.shape[0]*estimated_T_b.shape[1]))
    
    measured_T_a = df['T_a_measured'].values
    measured_T_b = df['T_b_measured'].values
    rmse_a = np.sqrt(np.mean((measured_T_a - estimated_T_a)**2))
    rmse_b = np.sqrt(np.mean((measured_T_b - estimated_T_b)**2))
    print(f"Before calibration RMSE Room A: {rmse_a:.3f} °C")
    print(f"Before calibration RMSE Room B: {rmse_b:.3f} °C")

    # Plot comparison
    fig, axes = tb.plot.plot(
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
        ylabel_1axis=r"Temperature $[^\circ$ C$]$",
        title="Before calibration: Measured vs Estimated",
        show=False,
        nticks=11
    )
    fig.savefig("before_calibration_measured_vs_estimated.png", dpi=300)


    
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

    # Parallel simulation
    start_time = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc) # [datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc), datetime.datetime(2024, 1, 2, tzinfo=datetime.timezone.utc), datetime.datetime(2024, 1, 3, tzinfo=datetime.timezone.utc)]
    end_time = datetime.datetime(2024, 1, 4, tzinfo=datetime.timezone.utc) # [datetime.datetime(2024, 1, 2, tzinfo=datetime.timezone.utc), datetime.datetime(2024, 1, 3, tzinfo=datetime.timezone.utc), datetime.datetime(2024, 1, 4, tzinfo=datetime.timezone.utc)]
    step_size = 600
    
    # 🕐 TIMING: Parameter estimation with 3 parallel periods
    print(f"\n🕐 Starting parameter estimation...")
    print(f"   Method: ('scipy', 'SLSQP', 'ad')")
    print(f"   Parameters to estimate: {len(parameters)}")
    
    start_time_estimation = time.time()
    
    # Run estimation
    result = estimator.estimate(
        start_time=start_time,
        end_time=end_time,
        step_size=step_size,
        parameters=parameters,
        measurements=measurements,
        n_warmup=20,
        method=("scipy", "SLSQP", "ad"),
        options={"maxiter": 500, "ftol": 1e-10}
    )
    
    end_time_estimation = time.time()
    estimation_duration = end_time_estimation - start_time_estimation
    
    print(f"\n✅ Parameter estimation completed!")
    print(f"   Total time: {estimation_duration:.2f} seconds ({estimation_duration/60:.2f} minutes)")
    
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
    
    # True values (from ThermalParameters default constructor)
    true = ThermalParameters()  # Uses default true values
    
    # Create comparison table
    print(f"\n{'='*80}")
    print("PARAMETER ESTIMATION RESULTS")
    print(f"{'='*80}")
    print(f"{'Parameter':<12} {'Unit':<8} {'True Value':<12} {'Initial Guess':<14} {'Estimated':<12} {'Error (%)':<10}")
    print(f"{'-'*80}")
    
    # Thermal resistances
    params = [
        ('R_wa', 'K/W', true.R_wa, initial.R_wa, estimated.R_wa),
        ('R_ia', 'K/W', true.R_ia, initial.R_ia, estimated.R_ia),
        ('R_wao', 'K/W', true.R_wao, initial.R_wao, estimated.R_wao),
        ('R_ib', 'K/W', true.R_ib, initial.R_ib, estimated.R_ib),
        ('R_wb', 'K/W', true.R_wb, initial.R_wb, estimated.R_wb),
        ('R_wbo', 'K/W', true.R_wbo, initial.R_wbo, estimated.R_wbo),
        ('C_a', 'J/K', true.C_a, initial.C_a, estimated.C_a),
        ('C_wa', 'J/K', true.C_wa, initial.C_wa, estimated.C_wa),
        ('C_i', 'J/K', true.C_i, initial.C_i, estimated.C_i),
        ('C_b', 'J/K', true.C_b, initial.C_b, estimated.C_b),
        ('C_wb', 'J/K', true.C_wb, initial.C_wb, estimated.C_wb),
    ]
    
    for param_name, unit, true_val, initial_val, estimated_val in params:
        error_pct = abs((estimated_val - true_val) / true_val) * 100
        print(f"{param_name:<12} {unit:<8} {true_val:<12.3e} {initial_val:<14.3e} {estimated_val:<12.3e} {error_pct:<10.1f}")
    
    print(f"{'-'*80}")
    
    # Summary statistics
    errors = [abs((est - true) / true) * 100 for _, _, true, _, est in params]
    print(f"Mean absolute error: {np.mean(errors):.1f}%")
    print(f"Max error: {np.max(errors):.1f}%")
    print(f"Parameters within 10% error: {sum(1 for e in errors if e <= 10)}/{len(errors)}")
    print(f"{'='*80}")
    
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
    estimated_T_a = thermal_sys.ss_model.output["y"].history[0,:,0].detach().numpy()
    estimated_T_b = thermal_sys.ss_model.output["y"].history[0,:,3].detach().numpy()
    
    # Calculate accuracy
    measured_T_a = df['T_a_measured'].values
    measured_T_b = df['T_b_measured'].values
    rmse_a = np.sqrt(np.mean((measured_T_a - estimated_T_a)**2))
    rmse_b = np.sqrt(np.mean((measured_T_b - estimated_T_b)**2))
    
    print(f"After calibration RMSE Room A: {rmse_a:.3f} °C")
    print(f"After calibration RMSE Room B: {rmse_b:.3f} °C")
    
    # Plot comparison
    fig, axes = tb.plot.plot(
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
        ylabel_1axis=r"Temperature $[^\circ$ C$]$",
        title="After calibration: Measured vs Estimated",
        show=False,
        nticks=11
    )
    fig.savefig("after_calibration_measured_vs_estimated.png", dpi=300)


# ============================================================================
# STEP 4: OPTIMAL CONTROL
# ============================================================================

def compare_timing_strategies(df, model):
    """Compare timing: 1 long period (3 days) vs 3 short periods (1 day each)"""
    print("\n" + "="*60)
    print("TIMING COMPARISON: LONG vs SHORT PERIODS")
    print("="*60)
    print("Comparing parameter estimation performance:\n")


    initial = ThermalParameters(
        R_wa=0.01, R_ia=0.002, R_wao=0.01,
        R_ib=0.002, R_wb=0.01, R_wbo=0.01,
        C_a=1e6, C_wa=5e5, C_i=1e6, C_b=1e6, C_wb=5e5
    )
    
    # Setup common parameters
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
    
    temp_a_sensor = model.components["TempASensor"]
    temp_b_sensor = model.components["TempBSensor"]
    measurements = [(temp_a_sensor, 0.05), (temp_b_sensor, 0.05)]
    
    # Strategy 1: One long period (3 days)
    print("📊 STRATEGY 1: Single long period (3 days)")
    estimator1 = tb.Estimator(tb.Simulator(model))
    
    start_time_long = [datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)]
    end_time_long = [datetime.datetime(2024, 1, 4, tzinfo=datetime.timezone.utc)]
    step_size = 600
    
    print(f"   Periods: {len(start_time_long)}")
    print(f"   Duration per period: 3 days")
    print(f"   Total duration: 3 days")
    
    start_time_est1 = time.time()
    result1 = estimator1.estimate(
        start_time=start_time_long,
        end_time=end_time_long,
        step_size=step_size,
        parameters=parameters,
        measurements=measurements,
        n_warmup=20,
        method=("scipy", "SLSQP", "ad"),
        options={"maxiter": 100, "ftol": 1e-8}  # Reduced iterations for comparison
    )
    duration1 = time.time() - start_time_est1
    
    print(f"   ✅ Completed in: {duration1:.2f} seconds ({duration1/60:.2f} minutes)")
    if hasattr(result1, 'nfev'):
        print(f"   Function evaluations: {result1.nfev}")
    
    # Strategy 2: Three short periods (1 day each)
    print(f"\n📊 STRATEGY 2: Multiple short periods (3 × 1 day)")
    estimator2 = tb.Estimator(tb.Simulator(model))
    
    start_time_short = [
        datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
        datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
        datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
    ]
    end_time_short = [
        datetime.datetime(2024, 1, 4, tzinfo=datetime.timezone.utc),
        datetime.datetime(2024, 1, 4, tzinfo=datetime.timezone.utc),
        datetime.datetime(2024, 1, 4, tzinfo=datetime.timezone.utc)
    ]
    
    print(f"   Periods: {len(start_time_short)}")
    print(f"   Duration per period: 1 day")
    print(f"   Total duration: 3 days")
    
    start_time_est2 = time.time()
    result2 = estimator2.estimate(
        start_time=start_time_short,
        end_time=end_time_short,
        step_size=step_size,
        parameters=parameters,
        measurements=measurements,
        n_warmup=20,
        method=("scipy", "SLSQP", "ad"),
        options={"maxiter": 100, "ftol": 1e-8}  # Reduced iterations for comparison
    )
    duration2 = time.time() - start_time_est2
    
    print(f"   ✅ Completed in: {duration2:.2f} seconds ({duration2/60:.2f} minutes)")
    if hasattr(result2, 'nfev'):
        print(f"   Function evaluations: {result2.nfev}")
    
    # Comparison
    print(f"\n📈 PERFORMANCE COMPARISON:")
    print(f"   Strategy 1 (1×3 days): {duration1:.2f}s")
    print(f"   Strategy 2 (3×1 day):  {duration2:.2f}s")
    
    if duration1 > duration2:
        speedup = duration1 / duration2
        print(f"   🚀 Strategy 2 is {speedup:.2f}x FASTER")
    else:
        slowdown = duration2 / duration1
        print(f"   🐌 Strategy 2 is {slowdown:.2f}x slower")
    
    print(f"   Time difference: {abs(duration1 - duration2):.2f} seconds")
    print(f"   Relative difference: {abs(duration1 - duration2)/max(duration1, duration2)*100:.1f}%")


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
    baseline_T = thermal_sys.ss_model.output["y"].history[0,:,0].detach().numpy()
    baseline_Q = model.components["RadiatorInput"].output["value"].history[0,:].detach().numpy()
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
    model.load(verbose=0)
    
    # Run optimization
    print("Running optimization...")
    optimizer = tb.Optimizer(tb.Simulator(model))

    # Parallel simulation
    start_time = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc) # [datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc), datetime.datetime(2024, 1, 2, tzinfo=datetime.timezone.utc), datetime.datetime(2024, 1, 3, tzinfo=datetime.timezone.utc)]
    end_time = datetime.datetime(2024, 1, 4, tzinfo=datetime.timezone.utc) # [datetime.datetime(2024, 1, 2, tzinfo=datetime.timezone.utc), datetime.datetime(2024, 1, 3, tzinfo=datetime.timezone.utc), datetime.datetime(2024, 1, 4, tzinfo=datetime.timezone.utc)]
    step_size = 600
    
    # 🎯 TIMING: Optimization with 3 parallel periods
    temp_sensor = model.components["TempASensor"]
    variables = [(schedule, "scheduleValue", 0, 2000)]
    objectives = [(schedule, "scheduleValue", "min")]
    ineq_cons = [
        (temp_sensor, "measuredValue", "lower", heating_sp),
        (temp_sensor, "measuredValue", "upper", cooling_sp)
    ]
    
    print(f"\n🎯 Starting optimization...")
    print(f"   Variables to optimize: {len(variables)}")
    print(f"   Objectives: {len(objectives)}")
    print(f"   Inequality constraints: {len(ineq_cons)}")
    
    start_time_optimization = time.time()
    
    opt_result = optimizer.optimize(
        start_time=start_time,
        end_time=end_time,
        step_size=step_size,
        variables=variables,
        objectives=objectives,
        ineq_cons=ineq_cons,
        method="scipy",
        options={"maxiter": 500, "disp": True}
    )
    
    end_time_optimization = time.time()
    optimization_duration = end_time_optimization - start_time_optimization
    
    print(f"\n✅ Optimization completed!")
    print(f"   Total time: {optimization_duration:.2f} seconds ({optimization_duration/60:.2f} minutes)")
    
    # Extract results
    optimized_T = thermal_sys.ss_model.output["y"].history[:,:,0].detach().flatten().numpy()
    optimized_Q = schedule.output["scheduleValue"].history[:,:].detach().flatten().numpy()
    optimized_energy = np.sum(optimized_Q) * 600 / 3600 / 1000  # kWh
    savings = (baseline_energy - optimized_energy) / baseline_energy * 100
    
    print(f"✓ Optimized energy: {optimized_energy:.2f} kWh")
    print(f"✓ Energy savings: {savings:.1f}%")
    
    # Plot results
    time_index = pd.date_range(start=df.index[0], periods=len(baseline_T), freq='10min')
    heating_sp_vals = heating_sp.output["scheduleValue"].history[:,:].detach().flatten().numpy()
    cooling_sp_vals = cooling_sp.output["scheduleValue"].history[:,:].detach().flatten().numpy()
    
    tb.plot.plot(
        time=time_index,
        entries=[
            tb.plot.Entry(data=heating_sp_vals, label="Min Temp", 
                         color=tb.plot.Colors.red, fmt=":", axis=1, linewidth=2),
            tb.plot.Entry(data=cooling_sp_vals, label="Max Temp", 
                         color=tb.plot.Colors.blue, fmt=":", axis=1, linewidth=2),
            tb.plot.Entry(data=baseline_T, label="Baseline", 
                         color=tb.plot.Colors.grey, fmt="--", axis=1),
            tb.plot.Entry(data=optimized_T, label="Optimized", 
                         color=tb.plot.Colors.green, fmt="-", axis=1),
            tb.plot.Entry(data=baseline_Q/1000, label="Baseline Heating", 
                         color=tb.plot.Colors.orange, fmt="--", axis=2),
            tb.plot.Entry(data=optimized_Q/1000, label="Optimized Heating", 
                         color=tb.plot.Colors.purple, fmt="-", axis=2),
        ],
        ylabel_1axis=r"Temperature $[^\circ$ C$]$",
        ylabel_2axis="Heating [kW]",
        title=f"Optimal Control (Savings: {savings:.1f}%)",
        show=True,
        nticks=11
    )


# ============================================================================
# HELPER FUNCTIONS (IMPLEMENTATION DETAILS)
# ============================================================================

def _build_model(df, params):
    """Build Twin4Build model (uses your state-space matrices)"""
    import tempfile
    import os
    import torch.nn as nn
    import twin4build.utils.types as tps
    
    # Create parametric system class that uses YOUR matrices
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
        
        def _create_ss(self):
            """Create state-space model using STUDENT's matrices"""
            # Call the student's function to get A, B, C, D
            A, B, C, D = build_state_space_matrices(
                self.R_wa_param.get(), self.R_ia_param.get(), self.R_wao_param.get(),
                self.R_ib_param.get(), self.R_wb_param.get(), self.R_wbo_param.get(),
                self.C_a_param.get(), self.C_wa_param.get(), self.C_i_param.get(),
                self.C_b_param.get(), self.C_wb_param.get()
            )
            
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
        
        def initialize(self, start_time, end_time, step_size):
            _, _, n_timesteps = tb.Simulator.get_simulation_timesteps(start_time, end_time, step_size)
            batch_size = len(start_time)
            for port in self.input.values():
                port.initialize(n_timesteps, batch_size=batch_size)
            for port in self.output.values():
                port.initialize(n_timesteps, batch_size=batch_size)
            
            self._create_ss()
            self.ss_model.initialize(start_time, end_time, step_size)
            self.INITIALIZED = True
        
        def do_step(self, second_time=None, date_time=None, step_size=None, step_index=None):
            self.ss_model.input["u"].set(self.input["u"].get(), step_index=step_index)
            # print("u.shape", self.ss_model.input["u"].tensor.shape)
            # print("u", self.ss_model.input["u"].tensor)
            # aa
            self.ss_model.do_step(second_time, date_time, step_size, step_index=step_index)
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
        
        model.load(verbose=0)
        return model



def sim():
    # Initial guess (deliberately inaccurate)
    initial = ThermalParameters(
        R_wa=0.01, R_ia=0.002, R_wao=0.01,
        R_ib=0.002, R_wb=0.01, R_wbo=0.01,
        C_a=1e6, C_wa=5e5, C_i=1e6, C_b=1e6, C_wb=5e5
    )

    # R_wa=0.014, R_ia=0.005, R_wao=0.016,
    #              R_ib=0.004, R_wb=0.014, R_wbo=0.018,
    #              C_a=5e5, C_wa=15e5, C_i=1e5, C_b=1e5, C_wb=3e5):
    
    print("Initial parameter guess:")
    print(f"  R_wa = {initial.R_wa:.3e} K/W")
    print(f"  C_a  = {initial.C_a:.3e} J/K")
    print("  ... (11 parameters total)")
    

    df = load_data()


    # Build model
    model = _build_model(df, initial)
    
    # Create estimator
    simulator = tb.Simulator(model)

    start_time = [datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc), datetime.datetime(2024, 1, 2, tzinfo=datetime.timezone.utc)]
    end_time = [datetime.datetime(2024, 1, 2, tzinfo=datetime.timezone.utc), datetime.datetime(2024, 1, 3, tzinfo=datetime.timezone.utc)]
    step_size = 600

    # Run initial simulation
    simulator.simulate(
        start_time=start_time,
        end_time=end_time,
        step_size=step_size,
    )
# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    """Run the complete building energy management workflow"""
    print("\n" + "="*60)
    print("BUILDING ENERGY MANAGEMENT SYSTEM")
    print("Student Exercise: State-Space Modeling")
    print("="*60 + "\n")
    
    # First, verify student matrices
    if not verify_matrices():
        print("\n⚠ Please complete the state-space matrices first!")
        print("Edit the build_state_space_matrices() function above.")
        return
    
    print("\n" + "="*60)
    print("MATRICES VERIFIED - STARTING SIMULATION")
    print("="*60)
    
    # Step 1: Load measurement data
    df = load_data()
    
    # Step 2: Estimate parameters from measurements
    estimated_params, model = estimate_parameters(df)
    
    # Step 3: Validate the calibrated model
    validate_model(df, model)
    
    # Step 3.5: Compare timing strategies (optional)
    # compare_timing_strategies(df, model)
    
    # Step 4: Optimize heating control
    optimize_control(df, model)
    
    print("\n" + "="*60)
    print("WORKFLOW COMPLETED!")
    print("="*60)
    print("✓ Your state-space matrices worked correctly!")
    print("✓ Parameters estimated from data")
    print("✓ Model validated against measurements")
    print("✓ Optimal control achieved energy savings")
    print("\nKey Takeaways:")
    print("• RC networks → state-space equations")
    print("• A matrix describes thermal coupling")
    print("• B matrix maps inputs to states")
    print("• Parameter estimation calibrates the model")
    print("• Optimal control minimizes energy use")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run the workflow
    main()
    # sim()