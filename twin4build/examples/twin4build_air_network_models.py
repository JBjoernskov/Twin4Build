#!/usr/bin/env python3
"""
Twin4Build-Compatible Air Network Models
========================================

This module converts the PyTorch air network models to be compatible with the Twin4Build framework.
It includes fan models, damper models, branch models, and a complete air network system that can be
used for parameter estimation and optimal control within Twin4Build.

Key Components:
- FanSystem: Twin4Build-compatible fan model with differentiable fan curves
- DamperSystem: Twin4Build-compatible damper model with variable resistance
- BranchSystem: Twin4Build-compatible branch model for ductwork
- AirNetworkSystem: Complete air network system with equilibrium solving
- EquilibriumSolver: Utility for solving air network equilibrium

Author: Converted from test_air_network_model_torch.py
"""


import pandas as pd
import twin4build as tb
import torch
import torch.nn as nn
import numpy as np
import datetime
from typing import List, Tuple, Optional, Dict, Any
import twin4build.utils.types as tps
import twin4build.core as core


# ============================================================================
# EQUILIBRIUM SOLVER UTILITY
# ============================================================================

def solve_equilibrium_differentiable(model, fan_speed: torch.Tensor, 
                                    supply_damper_pos: torch.Tensor,
                                    exhaust_damper_pos: torch.Tensor, 
                                    n_iterations: int = 50, 
                                    tol: float = 1e-3, 
                                    verbose: bool = False) -> torch.Tensor:
    """
    Solve equilibrium using differentiable fixed-point iteration.
    This maintains the computation graph throughout for gradient flow.
    
    Args:
        model: AirNetworkSystem instance
        fan_speed: Fan speed ratio (0-1) - can be scalar or batch
        supply_damper_pos: Supply damper positions (0-1) - can be scalar or batch
        exhaust_damper_pos: Exhaust damper positions (0-1) - can be scalar or batch
        n_iterations: Maximum iterations
        tol: Convergence tolerance
        verbose: Print convergence info
        
    Returns:
        Equilibrium flow rate
    """
    # Handle both scalar and batch inputs
    if fan_speed.dim() == 0:
        # Scalar inputs - add batch dimension
        fan_speed = fan_speed.unsqueeze(0)
        supply_damper_pos = supply_damper_pos.unsqueeze(0)
        exhaust_damper_pos = exhaust_damper_pos.unsqueeze(0)
        is_scalar = True
    else:
        is_scalar = False
    
    batch_size = fan_speed.shape[0]
    
    # Initialize with a reasonable guess
    flow = fan_speed * model.fan_system.design_flow
    
    prev_error = float('inf')
    stall_count = 0
    
    # Use exponentially decaying step size
    for i in range(n_iterations):
        # Calculate residual
        fan_pressure = model.fan_system.get_pressure_rise(flow, fan_speed)
        system_pressure = model.calculate_system_pressure(flow, supply_damper_pos, exhaust_damper_pos)
        residual = fan_pressure - system_pressure
        
        # Check convergence
        max_error = torch.abs(residual).max().item()
        
        if verbose and i % 10 == 0:
            print(f"Iteration {i}: max error = {max_error:.6f} Pa, flow = {flow.mean().item():.4f} m³/s")
        
        # Check if converged
        if max_error < tol:
            if verbose:
                print(f"Converged in {i} iterations, error = {max_error:.6f} Pa")
            break
        
        # Check if stalled (not making progress)
        if abs(max_error - prev_error) < 1e-6:
            stall_count += 1
            if stall_count > 10:
                if verbose:
                    print(f"Stalled at iteration {i}, error = {max_error:.6f} Pa (good enough)")
                break
        else:
            stall_count = 0
        
        prev_error = max_error
        
        # Adaptive step size: start large, decay over time
        alpha = 0.5 * (0.95 ** i)  # Decays from 0.5 to ~0.15 over 50 iterations
        alpha = max(alpha, 0.05)  # Don't go below 0.05
        
        # Simple update with proper scaling
        flow_scale = model.fan_system.design_flow / (model.fan_system.design_pressure + 1e-6)
        flow_update = alpha * residual * flow_scale
        
        flow = flow + flow_update
        flow = torch.clamp(flow, min=0.01, max=model.fan_system.design_flow * 1.5)
    
    if verbose and i == n_iterations - 1:
        print(f"Reached max iterations ({n_iterations}), error = {max_error:.6f} Pa")
    
    # Return scalar if input was scalar
    if is_scalar:
        return flow.squeeze(0)
    else:
        return flow


# ============================================================================
# FAN SYSTEM
# ============================================================================

class FanSystem(core.System, nn.Module):
    """
    Twin4Build-compatible differentiable fan model.
    
    This system models a centrifugal fan with a quadratic pressure-flow characteristic.
    The fan curve is parameterized as: P = a + b*Q + c*Q² where P is pressure rise,
    Q is flow rate, and a, b, c are learnable parameters.
    
    Inputs:
        - fanSpeed: Fan speed ratio (0-1)
        - flowRate: Air flow rate [m³/s]
        
    Outputs:
        - pressureRise: Fan pressure rise [Pa]
        - fanPower: Fan power consumption [W]
        
    Parameters:
        - design_pressure: Design pressure rise [Pa]
        - design_flow: Design flow rate [m³/s]
        - motor_efficiency: Motor efficiency (0-1)
        - a, b, c: Fan curve coefficients (learnable)
    """
    
    def __init__(self, design_pressure: float = 600.0, design_flow: float = 5.0, 
                 max_pressure: float = 900.0, motor_efficiency: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        
        # Store design parameters
        self.design_pressure = design_pressure
        self.design_flow = design_flow
        self.motor_efficiency = motor_efficiency
        
        # Initialize fan curve coefficients
        max_flow = design_flow * 1.3
        a_init = max_pressure
        c_init = -max_pressure / (max_flow ** 2)
        b_init = (design_pressure - a_init - c_init * design_flow**2) / design_flow
        
        # Learnable parameters
        self.a_param = tps.Parameter(torch.tensor(a_init/1000, dtype=torch.float64))  # kPa
        self.b_param = tps.Parameter(torch.tensor(b_init, dtype=torch.float64))
        self.c_param = tps.Parameter(torch.tensor(c_init, dtype=torch.float64))
        
        # Define inputs and outputs
        self.input = {
            "fanSpeed": tps.Scalar(),
            "flowRate": tps.Scalar()
        }
        self.output = {
            "pressureRise": tps.Scalar(),
            "fanPower": tps.Scalar()
        }
        
        # Parameter bounds for estimation
        self.parameter = {
            "a_param": {"lb": 0.1, "ub": 2.0},
            "b_param": {"lb": -500.0, "ub": 500.0},
            "c_param": {"lb": -1000.0, "ub": -10.0}
        }
        
        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = False
    
    @property
    def config(self):
        return self._config
    
    def get_pressure_rise(self, flow: torch.Tensor, speed_ratio: torch.Tensor) -> torch.Tensor:
        """Calculate fan pressure rise (differentiable)"""
        flow_norm = flow / (speed_ratio + 1e-6)
        pressure_design = self.a_param.get()*1000 + self.b_param.get() * flow_norm + self.c_param.get() * flow_norm**2
        pressure_rise = pressure_design * speed_ratio**2
        return torch.clamp(pressure_rise, min=0.0)
    
    def get_power(self, flow: torch.Tensor, pressure_rise: torch.Tensor) -> torch.Tensor:
        """Calculate fan power consumption"""
        air_power = flow * pressure_rise
        flow_ratio = flow / (self.design_flow + 1e-6)
        pressure_ratio = pressure_rise / (self.design_pressure + 1e-6)
        eta_fan = 0.65 * (1.0 - 0.3 * (flow_ratio - 1.0)**2 - 0.2 * (pressure_ratio - 1.0)**2)
        eta_fan = torch.clamp(eta_fan, min=0.3, max=0.75)
        eta_total = eta_fan * self.motor_efficiency
        return air_power / (eta_total + 1e-6)
    
    def initialize(self, start_time: datetime.datetime, end_time: datetime.datetime, 
                   step_size: int, simulator: core.Simulator) -> None:
        """Initialize the fan system"""
        for port in self.input.values():
            port.initialize(start_time, end_time, step_size, simulator)
        for port in self.output.values():
            port.initialize(start_time, end_time, step_size, simulator)
        self.INITIALIZED = True
    
    def do_step(self, second_time: float = None, date_time: datetime.datetime = None, 
                step_size: int = None, step_index: int = None) -> None:
        """Execute one simulation step"""
        # Get inputs
        fan_speed = self.input["fanSpeed"].get()
        flow_rate = self.input["flowRate"].get()
        
        # Calculate outputs
        pressure_rise = self.get_pressure_rise(flow_rate, fan_speed)
        fan_power = self.get_power(flow_rate, pressure_rise)
        
        # Set outputs
        self.output["pressureRise"].set(pressure_rise, step_index)
        self.output["fanPower"].set(fan_power, step_index)


# ============================================================================
# DAMPER SYSTEM
# ============================================================================

class DamperSystem(core.System, nn.Module):
    """
    Twin4Build-compatible differentiable damper model.
    
    This system models a variable air damper with exponential resistance characteristic.
    The resistance varies exponentially with damper position to model the non-linear
    flow control behavior of real dampers.
    
    Inputs:
        - damperPosition: Damper position (0=closed, 1=fully open)
        - flowRate: Air flow rate [m³/s]
        
    Outputs:
        - pressureDrop: Pressure drop across damper [Pa]
        - resistance: Damper flow resistance [Pa·s²/m⁶]
        
    Parameters:
        - fully_open_resistance: Resistance when fully open [Pa·s²/m⁶]
        - fully_closed_resistance: Resistance when fully closed [Pa·s²/m⁶]
    """
    
    def __init__(self, fully_open_resistance: float = 10.0, 
                 fully_closed_resistance: float = 1e5, **kwargs):
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        
        # Learnable parameters (log scale for numerical stability)
        self.log_R_open_param = tps.Parameter(torch.tensor(np.log(fully_open_resistance), dtype=torch.float64))
        self.k_param = tps.Parameter(torch.tensor(np.log(fully_closed_resistance / fully_open_resistance), dtype=torch.float64))
        
        # Define inputs and outputs
        self.input = {
            "damperPosition": tps.Scalar(),
            "flowRate": tps.Scalar()
        }
        self.output = {
            "pressureDrop": tps.Scalar(),
            "resistance": tps.Scalar()
        }
        
        # Parameter bounds for estimation
        self.parameter = {
            "log_R_open_param": {"lb": np.log(1.0), "ub": np.log(100.0)},
            "k_param": {"lb": np.log(10.0), "ub": np.log(1000.0)}
        }
        
        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = False
    
    @property
    def config(self):
        return self._config
    
    def get_resistance(self, position: torch.Tensor) -> torch.Tensor:
        """Calculate damper resistance based on position"""
        R_open = torch.exp(self.log_R_open_param.get())
        resistance = R_open * torch.exp(self.k_param.get() * (1.0 - position))
        return resistance
    
    def initialize(self, start_time: datetime.datetime, end_time: datetime.datetime, 
                   step_size: int, simulator: core.Simulator) -> None:
        """Initialize the damper system"""
        for port in self.input.values():
            port.initialize(start_time, end_time, step_size, simulator)
        for port in self.output.values():
            port.initialize(start_time, end_time, step_size, simulator)
        self.INITIALIZED = True
    
    def do_step(self, second_time: float = None, date_time: datetime.datetime = None, 
                step_size: int = None, step_index: int = None) -> None:
        """Execute one simulation step"""
        # Get inputs
        damper_position = self.input["damperPosition"].get()
        flow_rate = self.input["flowRate"].get()
        
        # Calculate resistance and pressure drop
        resistance = self.get_resistance(damper_position)
        abs_flow = torch.abs(flow_rate)
        pressure_drop = resistance * abs_flow**2
        
        # Set outputs
        self.output["resistance"].set(resistance, step_index)
        self.output["pressureDrop"].set(pressure_drop, step_index)


# ============================================================================
# BRANCH SYSTEM
# ============================================================================

class BranchSystem(core.System, nn.Module):
    """
    Twin4Build-compatible differentiable branch model.
    
    This system models a ductwork branch with both duct resistance and optional damper.
    The pressure drop includes both linear and non-linear components to capture
    realistic ductwork behavior.
    
    Inputs:
        - flowRate: Air flow rate [m³/s]
        - damperPosition: Damper position (0-1) [optional]
        
    Outputs:
        - pressureDrop: Total pressure drop across branch [Pa]
        - ductPressureDrop: Pressure drop due to duct resistance [Pa]
        - damperPressureDrop: Pressure drop due to damper [Pa] [optional]
        
    Parameters:
        - resistance_coeff: Duct resistance coefficient [Pa·s²/m⁶]
        - flow_exponent: Flow exponent (typically 2.0)
        - linear_resistance: Linear resistance component [Pa·s/m³]
        - has_damper: Whether branch includes a damper
    """
    
    def __init__(self, resistance_coeff: float = 15.0, flow_exponent: float = 2.0,
                 linear_resistance: float = 0.0, has_damper: bool = True,
                 damper_R_open: float = 10.0, damper_R_closed: float = 1e5, **kwargs):
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        
        self.flow_exponent = flow_exponent
        self.has_damper = has_damper
        
        # Learnable parameters
        self.log_R_duct_param = tps.Parameter(torch.tensor(np.log(resistance_coeff), dtype=torch.float64))
        self.R_linear_param = tps.Parameter(torch.tensor(linear_resistance, dtype=torch.float64))
        
        # Define inputs and outputs
        self.input = {"flowRate": tps.Scalar()}
        self.output = {
            "pressureDrop": tps.Scalar(),
            "ductPressureDrop": tps.Scalar()
        }
        
        # Add damper if specified
        if has_damper:
            self.damper = DamperSystem(damper_R_open, damper_R_closed, id=f"damper_{self.id}")
            self.input["damperPosition"] = tps.Scalar()
            self.output["damperPressureDrop"] = tps.Scalar()
        
        # Parameter bounds for estimation
        self.parameter = {
            "log_R_duct_param": {"lb": np.log(1.0), "ub": np.log(1000.0)},
            "R_linear_param": {"lb": 0.0, "ub": 100.0}
        }
        
        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = False
    
    @property
    def config(self):
        return self._config
    
    def get_pressure_drop(self, flow: torch.Tensor, damper_position: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate branch pressure drop"""
        abs_flow = torch.abs(flow)
        R_duct = torch.exp(self.log_R_duct_param.get())
        dp_duct = R_duct * abs_flow**self.flow_exponent + torch.abs(self.R_linear_param.get()) * abs_flow
        
        if self.has_damper and damper_position is not None:
            R_damper = self.damper.get_resistance(damper_position)
            dp_damper = R_damper * abs_flow**2
            return dp_duct + dp_damper, dp_duct, dp_damper
        
        return dp_duct, dp_duct, torch.zeros_like(dp_duct)
    
    def initialize(self, start_time: datetime.datetime, end_time: datetime.datetime, 
                   step_size: int, simulator: core.Simulator) -> None:
        """Initialize the branch system"""
        for port in self.input.values():
            port.initialize(start_time, end_time, step_size, simulator)
        for port in self.output.values():
            port.initialize(start_time, end_time, step_size, simulator)
        
        if self.has_damper:
            self.damper.initialize(start_time, end_time, step_size, simulator)
        
        self.INITIALIZED = True
    
    def do_step(self, second_time: float = None, date_time: datetime.datetime = None, 
                step_size: int = None, step_index: int = None) -> None:
        """Execute one simulation step"""
        # Get inputs
        flow_rate = self.input["flowRate"].get()
        damper_position = self.input.get("damperPosition", None)
        if damper_position is not None:
            damper_position = damper_position.get()
        
        # Calculate pressure drops
        total_dp, duct_dp, damper_dp = self.get_pressure_drop(flow_rate, damper_position)
        
        # Set outputs
        self.output["pressureDrop"].set(total_dp, step_index)
        self.output["ductPressureDrop"].set(duct_dp, step_index)
        if self.has_damper:
            self.output["damperPressureDrop"].set(damper_dp, step_index)


# ============================================================================
# AIR NETWORK SYSTEM
# ============================================================================

class AirNetworkSystem(core.System, nn.Module):
    """
    Twin4Build-compatible differentiable air network system.
    
    This system models a complete air network with fan, supply branches, and exhaust branches.
    It solves the equilibrium flow using differentiable fixed-point iteration, maintaining
    gradients throughout for parameter estimation and optimization.
    
    Inputs:
        - fanSpeed: Fan speed ratio (0-1)
        - supplyDamperPositions: Supply damper positions [Vector]
        - exhaustDamperPositions: Exhaust damper positions [Vector]
        
    Outputs:
        - totalFlow: Total air flow rate [m³/s]
        - fanPressure: Fan pressure rise [Pa]
        - systemPressure: System pressure drop [Pa]
        - fanPower: Fan power consumption [W]
        - supplyFlows: Individual supply branch flows [Vector]
        - exhaustFlows: Individual exhaust branch flows [Vector]
        
    Components:
        - fan_system: FanSystem instance
        - supply_branches: List of BranchSystem instances
        - exhaust_branches: List of BranchSystem instances
    """
    
    def __init__(self, fan_config: Dict[str, Any] = None, 
                 supply_branch_configs: List[Dict[str, Any]] = None,
                 exhaust_branch_configs: List[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        
        # Default configurations
        if fan_config is None:
            fan_config = {"design_pressure": 600, "design_flow": 5.0, "max_pressure": 900}
        if supply_branch_configs is None:
            supply_branch_configs = [
                {"resistance_coeff": 15}, {"resistance_coeff": 20}, {"resistance_coeff": 25}
            ]
        if exhaust_branch_configs is None:
            exhaust_branch_configs = [
                {"resistance_coeff": 12}, {"resistance_coeff": 15}, {"resistance_coeff": 18}
            ]
        
        # Create subsystems
        self.fan_system = FanSystem(id=f"fan_{self.id}", **fan_config)
        
        self.supply_branches = nn.ModuleList([
            BranchSystem(id=f"supply_branch_{i}_{self.id}", **config)
            for i, config in enumerate(supply_branch_configs)
        ])
        
        self.exhaust_branches = nn.ModuleList([
            BranchSystem(id=f"exhaust_branch_{i}_{self.id}", **config)
            for i, config in enumerate(exhaust_branch_configs)
        ])
        
        self.n_supply = len(self.supply_branches)
        self.n_exhaust = len(self.exhaust_branches)
        
        # Define inputs and outputs
        self.input = {
            "fanSpeed": tps.Scalar(),
            "supplyDamperPositions": tps.Vector(size=self.n_supply),
            "exhaustDamperPositions": tps.Vector(size=self.n_exhaust)
        }
        
        self.output = {
            "totalFlow": tps.Scalar(),
            "fanPressure": tps.Scalar(),
            "systemPressure": tps.Scalar(),
            "fanPower": tps.Scalar(),
            "supplyFlows": tps.Vector(size=self.n_supply),
            "exhaustFlows": tps.Vector(size=self.n_exhaust)
        }
        
        self._config = {"parameters": []}
        self.INITIALIZED = False
    
    @property
    def config(self):
        return self._config
    
    def distribute_flow_parallel(self, total_flow: torch.Tensor, branches: nn.ModuleList,
                                 damper_positions: torch.Tensor, n_iterations: int = 15) -> Tuple[torch.Tensor, torch.Tensor]:
        """Distribute flow among parallel branches (differentiable)"""
        # Handle different tensor shapes from Twin4Build's types.py
        # total_flow might come as (1,1) from _convert_to_2D_scalar_tensor
        # damper_positions might come as (1,n) from _convert_to_2D_tensor
        
        # Ensure total_flow is 1D
        if total_flow.dim() == 2 and total_flow.shape[0] == 1:
            total_flow = total_flow.squeeze(0)  # (1,1) -> (1,) or (1,n) -> (n,)
        if total_flow.dim() == 0:
            total_flow = total_flow.unsqueeze(0)  # scalar -> (1,)
            
        # Ensure damper_positions is 2D with proper batch dimension
        if damper_positions.dim() == 1:
            damper_positions = damper_positions.unsqueeze(0)  # (n,) -> (1,n)
        elif damper_positions.dim() == 2 and damper_positions.shape[0] == 1:
            pass  # Already (1,n) - this is correct
        elif damper_positions.dim() == 0:
            damper_positions = damper_positions.unsqueeze(0).unsqueeze(0)  # scalar -> (1,1)
            
        batch_size = max(total_flow.shape[0], damper_positions.shape[0])
        n_branches = damper_positions.shape[1] if damper_positions.dim() == 2 else len(branches)
        
        # Ensure total_flow has batch dimension
        if total_flow.shape[0] == 1 and batch_size > 1:
            total_flow = total_flow.expand(batch_size)
        elif total_flow.dim() == 1 and total_flow.shape[0] != batch_size:
            total_flow = total_flow.expand(batch_size)
            
        # Initialize flows equally - total_flow should be (batch_size,)
        flows = total_flow.unsqueeze(1).expand(-1, n_branches) / n_branches
        
        # Iterative distribution (fixed iterations for differentiability)
        for _ in range(n_iterations):
            pressures = torch.stack([
                branches[i].get_pressure_drop(flows[:, i], damper_positions[:, i])[0]
                for i in range(n_branches)
            ], dim=1)
            
            target_pressure = pressures.mean(dim=1, keepdim=True)
            R_eq = pressures / (flows**2 + 1e-6)
            new_flows = torch.sqrt(torch.clamp(target_pressure / (R_eq + 1e-6), min=0))
            flow_sum = new_flows.sum(dim=1, keepdim=True)
            flows = new_flows * total_flow.unsqueeze(1) / (flow_sum + 1e-6)
        
        pressures = torch.stack([
            branches[i].get_pressure_drop(flows[:, i], damper_positions[:, i])[0]
            for i in range(n_branches)
        ], dim=1)
        
        return flows, pressures
    
    def calculate_system_pressure(self, total_flow: torch.Tensor, 
                                  supply_damper_pos: torch.Tensor,
                                  exhaust_damper_pos: torch.Tensor) -> torch.Tensor:
        """Calculate system pressure drop (differentiable)"""
        _, supply_pressures = self.distribute_flow_parallel(
            total_flow, self.supply_branches, supply_damper_pos
        )
        _, exhaust_pressures = self.distribute_flow_parallel(
            total_flow, self.exhaust_branches, exhaust_damper_pos
        )
        system_pressure = supply_pressures[:, 0] + exhaust_pressures[:, 0]
        return system_pressure
    
    def solve_network_equilibrium(self, fan_speed: torch.Tensor, 
                                  supply_damper_pos: torch.Tensor,
                                  exhaust_damper_pos: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Solve network equilibrium and return all results
        """
        # Solve for equilibrium flow (maintains computation graph)
        flow_solution = solve_equilibrium_differentiable(
            self, fan_speed, supply_damper_pos, exhaust_damper_pos, n_iterations=50
        )
        
        # Calculate all outputs at the solution point
        flow_pos = torch.clamp(flow_solution, min=0.0)
        fan_pressure = self.fan_system.get_pressure_rise(flow_pos, fan_speed)
        system_pressure = self.calculate_system_pressure(flow_pos, supply_damper_pos, exhaust_damper_pos)
        fan_power = self.fan_system.get_power(flow_pos, fan_pressure)
        
        supply_flows, supply_pressures = self.distribute_flow_parallel(
            flow_pos, self.supply_branches, supply_damper_pos
        )
        exhaust_flows, exhaust_pressures = self.distribute_flow_parallel(
            flow_pos, self.exhaust_branches, exhaust_damper_pos
        )
        
        return {
            'total_flow': flow_pos,
            'fan_pressure': fan_pressure,
            'system_pressure': system_pressure,
            'fan_power': fan_power,
            'supply_flows': supply_flows,
            'supply_pressures': supply_pressures,
            'exhaust_flows': exhaust_flows,
            'exhaust_pressures': exhaust_pressures
        }
    
    def initialize(self, start_time: datetime.datetime, end_time: datetime.datetime, 
                   step_size: int, simulator: core.Simulator) -> None:
        """Initialize the air network system"""
        for port in self.input.values():
            port.initialize(start_time, end_time, step_size, simulator)
        for port in self.output.values():
            port.initialize(start_time, end_time, step_size, simulator)
        
        # Initialize subsystems
        self.fan_system.initialize(start_time, end_time, step_size, simulator)
        for branch in self.supply_branches:
            branch.initialize(start_time, end_time, step_size, simulator)
        for branch in self.exhaust_branches:
            branch.initialize(start_time, end_time, step_size, simulator)
        
        self.INITIALIZED = True
    
    def do_step(self, second_time: float = None, date_time: datetime.datetime = None, 
                step_size: int = None, step_index: int = None) -> None:
        """Execute one simulation step"""
        # Get inputs
        fan_speed = self.input["fanSpeed"].get()
        supply_damper_pos = self.input["supplyDamperPositions"].get()
        exhaust_damper_pos = self.input["exhaustDamperPositions"].get()
        
        # Handle tensor shapes from Twin4Build's updated types.py
        # Convert to proper shapes for our equilibrium solver
        if fan_speed.dim() == 2:
            fan_speed = fan_speed.squeeze()  # Remove batch dimensions if present
        if supply_damper_pos.dim() == 2 and supply_damper_pos.shape[0] == 1:
            supply_damper_pos = supply_damper_pos.squeeze(0)  # (1,n) -> (n,)
        if exhaust_damper_pos.dim() == 2 and exhaust_damper_pos.shape[0] == 1:
            exhaust_damper_pos = exhaust_damper_pos.squeeze(0)  # (1,n) -> (n,)
        
        # Solve network equilibrium
        results = self.solve_network_equilibrium(fan_speed, supply_damper_pos, exhaust_damper_pos)
        
        # Set outputs
        self.output["totalFlow"].set(results['total_flow'], step_index)
        self.output["fanPressure"].set(results['fan_pressure'], step_index)
        self.output["systemPressure"].set(results['system_pressure'], step_index)
        self.output["fanPower"].set(results['fan_power'], step_index)
        self.output["supplyFlows"].set(results['supply_flows'], step_index)
        self.output["exhaustFlows"].set(results['exhaust_flows'], step_index)


# ============================================================================
# MODEL BUILDER UTILITIES
# ============================================================================

def create_air_network_system(fan_config: Dict[str, Any] = None,
                              supply_configs: List[Dict[str, Any]] = None,
                              exhaust_configs: List[Dict[str, Any]] = None,
                              system_id: str = "AirNetwork") -> AirNetworkSystem:
    """
    Create a standalone air network system component.
    
    This is useful when you want to integrate the air network into a larger model
    or when you need to manually control inputs programmatically.
    
    Args:
        fan_config: Fan configuration parameters
        supply_configs: List of supply branch configurations  
        exhaust_configs: List of exhaust branch configurations
        system_id: System identifier
        
    Returns:
        AirNetworkSystem component ready to be added to a model
        
    Example:
        >>> air_network = create_air_network_system()
        >>> model = tb.Model(id="my_model")
        >>> model.add_component(air_network)
        >>> # Manually set inputs during simulation
        >>> air_network.input["fanSpeed"].set(torch.tensor(0.8))
    """
    return AirNetworkSystem(
        fan_config=fan_config,
        supply_branch_configs=supply_configs,
        exhaust_branch_configs=exhaust_configs,
        id=system_id
    )


def create_air_network_model(fan_config: Dict[str, Any] = None,
                             supply_configs: List[Dict[str, Any]] = None,
                             exhaust_configs: List[Dict[str, Any]] = None,
                             input_data: Dict[str, Any] = None,
                             model_id: str = "air_network_model") -> tb.Model:
    """
    Create a complete Twin4Build model with air network system and inputs.
    
    This is the main function for creating a ready-to-simulate air network model.
    You should provide input_data for realistic simulations.
    
    Args:
        fan_config: Fan configuration parameters
        supply_configs: List of supply branch configurations
        exhaust_configs: List of exhaust branch configurations
        input_data: Dictionary with time series data or pandas DataFrames:
                   - 'fan_speed': Fan speed time series (0-1)
                   - 'supply_dampers': Supply damper positions (0-1) 
                   - 'exhaust_dampers': Exhaust damper positions (0-1)
        model_id: Model identifier
        
    Returns:
        Configured Twin4Build model ready for simulation
        
    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # Create input data
        >>> time_index = pd.date_range('2024-01-01', periods=100, freq='10min')
        >>> input_data = {
        >>>     'fan_speed': pd.Series(np.random.uniform(0.6, 1.0, 100), index=time_index),
        >>>     'supply_dampers': pd.DataFrame({
        >>>         'damper_0': np.random.uniform(0.7, 1.0, 100),
        >>>         'damper_1': np.random.uniform(0.7, 1.0, 100),
        >>>         'damper_2': np.random.uniform(0.7, 1.0, 100)
        >>>     }, index=time_index),
        >>>     'exhaust_dampers': pd.DataFrame({
        >>>         'damper_0': np.random.uniform(0.7, 1.0, 100),
        >>>         'damper_1': np.random.uniform(0.7, 1.0, 100),
        >>>         'damper_2': np.random.uniform(0.7, 1.0, 100)
        >>>     }, index=time_index)
        >>> }
        >>> 
        >>> model = create_air_network_model(input_data=input_data)
        >>> simulator = tb.Simulator(model)
    """
    import pandas as pd
    
    # Create model
    model = tb.Model(id=model_id)
    
    # Create air network system
    air_network = AirNetworkSystem(
        fan_config=fan_config,
        supply_branch_configs=supply_configs,
        exhaust_branch_configs=exhaust_configs,
        id="AirNetwork"
    )
    
    # Create input systems if data provided
    if input_data is not None:
        # Fan speed input
        if 'fan_speed' in input_data:
            fan_data = input_data['fan_speed']
            if isinstance(fan_data, pd.Series):
                df = pd.DataFrame({'fanSpeed': fan_data})
            else:
                df = pd.DataFrame({'fanSpeed': fan_data})
            
            fan_input = tb.TimeSeriesInputSystem(df=df, id="FanSpeedInput")
            model.add_connection(fan_input, air_network, "value", "fanSpeed")
        
        # Supply damper inputs - need individual connections for each damper
        if 'supply_dampers' in input_data:
            supply_data = input_data['supply_dampers']
            if isinstance(supply_data, pd.DataFrame):
                df = supply_data
            else:
                df = pd.DataFrame(supply_data)
            
            # Create individual input systems for each damper column
            for i, col in enumerate(df.columns):
                damper_input = tb.TimeSeriesInputSystem(
                    df=pd.DataFrame({col: df[col]}), 
                    id=f"SupplyDamper{i}Input"
                )
                model.add_connection(damper_input, air_network, "value", "supplyDamperPositions", input_port_index=i)
        
        # Exhaust damper inputs - need individual connections for each damper
        if 'exhaust_dampers' in input_data:
            exhaust_data = input_data['exhaust_dampers']
            if isinstance(exhaust_data, pd.DataFrame):
                df = exhaust_data
            else:
                df = pd.DataFrame(exhaust_data)
            
            # Create individual input systems for each damper column
            for i, col in enumerate(df.columns):
                damper_input = tb.TimeSeriesInputSystem(
                    df=pd.DataFrame({col: df[col]}), 
                    id=f"ExhaustDamper{i}Input"
                )
                model.add_connection(damper_input, air_network, "value", "exhaustDamperPositions", input_port_index=i)
    else:
        # If no input data provided, just add the air network component
        # This is mainly useful for parameter estimation where inputs come from measurements
        model.add_component(air_network)
        print("Warning: No input data provided. You'll need to connect inputs manually or use for parameter estimation.")
    
    # Load model
    model.load()
    
    return model


def create_air_network_for_estimation(fan_config: Dict[str, Any] = None,
                                      supply_configs: List[Dict[str, Any]] = None,
                                      exhaust_configs: List[Dict[str, Any]] = None,
                                      measurement_data: Dict[str, pd.DataFrame] = None,
                                      model_id: str = "air_network_estimation") -> tb.Model:
    """
    Create an air network model specifically configured for parameter estimation.
    
    This function creates a model with both input time series and sensor systems
    for comparing predictions with measurements during parameter estimation.
    
    Args:
        fan_config: Fan configuration parameters
        supply_configs: List of supply branch configurations
        exhaust_configs: List of exhaust branch configurations  
        measurement_data: Dictionary containing:
                         - 'inputs': DataFrame with fan_speed, supply_dampers, exhaust_dampers
                         - 'measurements': DataFrame with measured outputs (total_flow, fan_power, etc.)
        model_id: Model identifier
        
    Returns:
        Twin4Build model ready for parameter estimation
        
    Example:
        >>> # Prepare data for estimation
        >>> measurement_data = {
        >>>     'inputs': pd.DataFrame({
        >>>         'fan_speed': [...],
        >>>         'supply_damper_0': [...],
        >>>         'supply_damper_1': [...],
        >>>         # ... more dampers
        >>>     }),
        >>>     'measurements': pd.DataFrame({
        >>>         'measured_flow': [...],
        >>>         'measured_power': [...],
        >>>     })
        >>> }
        >>> 
        >>> model = create_air_network_for_estimation(measurement_data=measurement_data)
        >>> estimator = tb.Estimator(tb.Simulator(model))
    """
    import pandas as pd
    import tempfile
    import os
    
    if measurement_data is None:
        raise ValueError("measurement_data is required for estimation model")
    
    # Create model
    model = tb.Model(id=model_id)
    
    # Create air network system  
    air_network = AirNetworkSystem(
        fan_config=fan_config,
        supply_branch_configs=supply_configs,
        exhaust_branch_configs=exhaust_configs,
        id="AirNetwork"
    )
    
    inputs_df = measurement_data['inputs']
    measurements_df = measurement_data['measurements']
    
    # Create input systems from measurement data
    if 'fan_speed' in inputs_df.columns:
        fan_input = tb.TimeSeriesInputSystem(
            df=pd.DataFrame({'fanSpeed': inputs_df['fan_speed']}),
            id="FanSpeedInput"
        )
        model.add_connection(fan_input, air_network, "value", "fanSpeed")
    
    # Handle supply dampers (assuming columns like 'supply_damper_0', 'supply_damper_1', etc.)
    supply_cols = [col for col in inputs_df.columns if col.startswith('supply_damper')]
    if supply_cols:
        supply_df = inputs_df[supply_cols]
        # Create individual input systems for each damper column
        for i, col in enumerate(supply_cols):
            damper_input = tb.TimeSeriesInputSystem(
                df=pd.DataFrame({col: supply_df[col]}), 
                id=f"SupplyDamper{i}Input"
            )
            model.add_connection(damper_input, air_network, "value", "supplyDamperPositions", input_port_index=i)
    
    # Handle exhaust dampers
    exhaust_cols = [col for col in inputs_df.columns if col.startswith('exhaust_damper')]
    if exhaust_cols:
        exhaust_df = inputs_df[exhaust_cols]
        # Create individual input systems for each damper column
        for i, col in enumerate(exhaust_cols):
            damper_input = tb.TimeSeriesInputSystem(
                df=pd.DataFrame({col: exhaust_df[col]}), 
                id=f"ExhaustDamper{i}Input"
            )
            model.add_connection(damper_input, air_network, "value", "exhaustDamperPositions", input_port_index=i)
    
    # Create sensor systems for measurements
    with tempfile.TemporaryDirectory() as temp_dir:
        # Flow sensor
        if 'measured_flow' in measurements_df.columns:
            flow_file = os.path.join(temp_dir, "flow_measurements.csv")
            pd.DataFrame({'measuredValue': measurements_df['measured_flow']}).to_csv(flow_file)
            flow_sensor = tb.SensorSystem(
                df=pd.read_csv(flow_file, index_col=0, parse_dates=True),
                id="FlowSensor"
            )
            model.add_connection(air_network, flow_sensor, "totalFlow", "measuredValue")
        
        # Power sensor
        if 'measured_power' in measurements_df.columns:
            power_file = os.path.join(temp_dir, "power_measurements.csv")
            pd.DataFrame({'measuredValue': measurements_df['measured_power']}).to_csv(power_file)
            power_sensor = tb.SensorSystem(
                df=pd.read_csv(power_file, index_col=0, parse_dates=True),
                id="PowerSensor"
            )
            model.add_connection(air_network, power_sensor, "fanPower", "measuredValue")
    
    # Load model
    model.load()
    
    return model


# ============================================================================
# PARAMETER ESTIMATION EXAMPLE
# ============================================================================

def generate_synthetic_data(n_samples: int = 300) -> Dict[str, Any]:
    """
    Generate synthetic measurement data for parameter estimation.
    This replicates the data generation from test_air_network_model_torch.py
    """
    import pandas as pd
    import numpy as np
    
    print("Generating synthetic measurement data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create time index with timezone (required by Twin4Build)
    time_index = pd.date_range('2024-01-01', periods=n_samples, freq='10min', tz='UTC')
    
    # Generate input data (what we control/measure)
    fan_speeds = np.random.uniform(0.5, 1.0, n_samples)
    supply_dampers_data = np.random.uniform(0.5, 1.0, (n_samples, 3))
    exhaust_dampers_data = np.random.uniform(0.5, 1.0, (n_samples, 3))
    
    # True model parameters (unknown to estimator)
    true_fan_config = {
        "design_pressure": 620,  # Different from initial guess
        "design_flow": 4.8,      # Different from initial guess  
        "max_pressure": 1500     # Different from initial guess
    }
    
    true_supply_configs = [
        {"resistance_coeff": 18},  # Different from initial guess
        {"resistance_coeff": 22},  # Different from initial guess
        {"resistance_coeff": 28}   # Different from initial guess
    ]
    
    true_exhaust_configs = [
        {"resistance_coeff": 14},  # Different from initial guess
        {"resistance_coeff": 17},  # Different from initial guess
        {"resistance_coeff": 20}   # Different from initial guess
    ]
    
    # Create true model to generate measurements
    true_air_network = AirNetworkSystem(
        fan_config=true_fan_config,
        supply_branch_configs=true_supply_configs,
        exhaust_branch_configs=true_exhaust_configs,
        id="TrueAirNetwork"
    )
    
    # Generate measurements by running true model
    measured_flows = []
    measured_powers = []
    
    for i in range(n_samples):
        fan_speed = torch.tensor(fan_speeds[i], dtype=torch.float64)
        supply_dampers = torch.tensor(supply_dampers_data[i], dtype=torch.float64)
        exhaust_dampers = torch.tensor(exhaust_dampers_data[i], dtype=torch.float64)
        
        # Solve equilibrium for true model
        with torch.no_grad():
            results = true_air_network.solve_network_equilibrium(
                fan_speed, supply_dampers, exhaust_dampers
            )
            measured_flows.append(results['total_flow'].item())
            measured_powers.append(results['fan_power'].item())
    
    # Add measurement noise (optional)
    measured_flows = np.array(measured_flows)
    measured_powers = np.array(measured_powers)
    # measured_flows += np.random.normal(0, 0.05, n_samples)  # 5% noise
    # measured_powers += np.random.normal(0, 50, n_samples)   # 50W noise
    
    # Create DataFrames
    inputs_df = pd.DataFrame({
        'fan_speed': fan_speeds,
        'supply_damper_0': supply_dampers_data[:, 0],
        'supply_damper_1': supply_dampers_data[:, 1], 
        'supply_damper_2': supply_dampers_data[:, 2],
        'exhaust_damper_0': exhaust_dampers_data[:, 0],
        'exhaust_damper_1': exhaust_dampers_data[:, 1],
        'exhaust_damper_2': exhaust_dampers_data[:, 2]
    }, index=time_index)
    
    measurements_df = pd.DataFrame({
        'measured_flow': measured_flows,
        'measured_power': measured_powers
    }, index=time_index)
    
    print(f"✓ Generated {n_samples} synthetic measurements")
    print(f"✓ Flow range: {measured_flows.min():.2f} - {measured_flows.max():.2f} m³/s")
    print(f"✓ Power range: {measured_powers.min():.0f} - {measured_powers.max():.0f} W")
    
    return {
        'inputs': inputs_df,
        'measurements': measurements_df,
        'true_parameters': {
            'fan_config': true_fan_config,
            'supply_configs': true_supply_configs,
            'exhaust_configs': true_exhaust_configs
        }
    }


def run_parameter_estimation(measurement_data: Dict[str, Any]) -> tb.Model:
    """
    Run parameter estimation using Twin4Build's estimator.
    This replicates the parameter estimation from test_air_network_model_torch.py
    """
    print("\n" + "="*60)
    print("AIR NETWORK PARAMETER ESTIMATION")
    print("="*60)
    
    # Initial guess (deliberately inaccurate)
    initial_fan_config = {
        "design_pressure": 600,
        "design_flow": 5.0,
        "max_pressure": 900
    }
    
    initial_supply_configs = [
        {"resistance_coeff": 15},
        {"resistance_coeff": 20},
        {"resistance_coeff": 25}
    ]
    
    initial_exhaust_configs = [
        {"resistance_coeff": 12},
        {"resistance_coeff": 15},
        {"resistance_coeff": 18}
    ]
    
    print("Initial parameter guess:")
    print(f"  Fan: P_design={initial_fan_config['design_pressure']} Pa, Q_design={initial_fan_config['design_flow']} m³/s")
    print(f"  Supply resistances: {[c['resistance_coeff'] for c in initial_supply_configs]}")
    print(f"  Exhaust resistances: {[c['resistance_coeff'] for c in initial_exhaust_configs]}")
    
    # Create model for estimation
    model = create_air_network_for_estimation(
        fan_config=initial_fan_config,
        supply_configs=initial_supply_configs,
        exhaust_configs=initial_exhaust_configs,
        measurement_data=measurement_data,
        model_id="air_network_estimation"
    )
    
    # Create simulator and estimator
    simulator = tb.Simulator(model)
    estimator = tb.Estimator(simulator)
    
    # Run initial simulation to check fit before estimation
    simulator.simulate(
        step_size=600,
        start_time=measurement_data['inputs'].index[0],
        end_time=measurement_data['inputs'].index[-1] + pd.Timedelta(seconds=600)
    )
    
    # Calculate initial RMSE
    air_network = model.components["AirNetwork"]
    initial_flow = air_network.output["totalFlow"].history.detach().numpy()
    initial_power = air_network.output["fanPower"].history.detach().numpy()
    
    measured_flow = measurement_data['measurements']['measured_flow'].values
    measured_power = measurement_data['measurements']['measured_power'].values
    
    initial_rmse_flow = np.sqrt(np.mean((measured_flow - initial_flow)**2))
    initial_rmse_power = np.sqrt(np.mean((measured_power - initial_power)**2))
    
    print(f"\nBefore estimation RMSE:")
    print(f"  Flow: {initial_rmse_flow:.3f} m³/s")
    print(f"  Power: {initial_rmse_power:.0f} W")
    
    # Define parameters to estimate
    parameters = [
        # Fan parameters
        (air_network.fan_system, "a_param", initial_fan_config["max_pressure"]/1000, 0.1, 2.0),
        (air_network.fan_system, "b_param", 0.0, -500.0, 500.0),
        (air_network.fan_system, "c_param", -100.0, -1000.0, -10.0),
        
        # Supply branch resistances
        (air_network.supply_branches[0], "log_R_duct_param", np.log(15), np.log(1.0), np.log(1000.0)),
        (air_network.supply_branches[1], "log_R_duct_param", np.log(20), np.log(1.0), np.log(1000.0)),
        (air_network.supply_branches[2], "log_R_duct_param", np.log(25), np.log(1.0), np.log(1000.0)),
        
        # Exhaust branch resistances  
        (air_network.exhaust_branches[0], "log_R_duct_param", np.log(12), np.log(1.0), np.log(1000.0)),
        (air_network.exhaust_branches[1], "log_R_duct_param", np.log(15), np.log(1.0), np.log(1000.0)),
        (air_network.exhaust_branches[2], "log_R_duct_param", np.log(18), np.log(1.0), np.log(1000.0)),
    ]
    
    # Define measurements to match
    flow_sensor = model.components["FlowSensor"]
    power_sensor = model.components["PowerSensor"]
    measurements = [
        (flow_sensor, 0.1),   # Flow sensor with 0.1 m³/s uncertainty
        (power_sensor, 100.0) # Power sensor with 100W uncertainty
    ]
    
    print(f"\nEstimating {len(parameters)} parameters using {len(measurements)} measurement types...")
    print("This may take several minutes...")
    
    # Run estimation
    estimator.estimate(
        start_time=measurement_data['inputs'].index[0],
        end_time=measurement_data['inputs'].index[-1] + pd.Timedelta(seconds=600),
        step_size=600,
        parameters=parameters,
        measurements=measurements,
        n_warmup=20,
        method=("scipy", "SLSQP", "ad"),
        options={"maxiter": 200, "ftol": 1e-6}
    )
    
    print("✓ Parameter estimation complete!")
    
    # Extract estimated parameters
    estimated_fan_a = air_network.fan_system.a_param.get().item()
    estimated_fan_b = air_network.fan_system.b_param.get().item()
    estimated_fan_c = air_network.fan_system.c_param.get().item()
    
    estimated_supply_resistances = [
        torch.exp(branch.log_R_duct_param.get()).item() 
        for branch in air_network.supply_branches
    ]
    
    estimated_exhaust_resistances = [
        torch.exp(branch.log_R_duct_param.get()).item()
        for branch in air_network.exhaust_branches  
    ]
    
    print(f"\nEstimated parameters:")
    print(f"  Fan curve: a={estimated_fan_a:.3f}, b={estimated_fan_b:.1f}, c={estimated_fan_c:.1f}")
    print(f"  Supply resistances: {[f'{r:.1f}' for r in estimated_supply_resistances]}")
    print(f"  Exhaust resistances: {[f'{r:.1f}' for r in estimated_exhaust_resistances]}")
    
    # Compare with true parameters
    true_params = measurement_data['true_parameters']
    print(f"\nTrue parameters:")
    print(f"  Supply resistances: {[c['resistance_coeff'] for c in true_params['supply_configs']]}")
    print(f"  Exhaust resistances: {[c['resistance_coeff'] for c in true_params['exhaust_configs']]}")
    
    return model


def validate_estimation(model: tb.Model, measurement_data: Dict[str, Any]):
    """
    Validate the estimated model against measurements.
    """
    print("\n" + "="*60)
    print("MODEL VALIDATION")
    print("="*60)
    
    # Run simulation with estimated parameters
    simulator = tb.Simulator(model)
    simulator.simulate(
        step_size=600,
        start_time=measurement_data['inputs'].index[0],
        end_time=measurement_data['inputs'].index[-1] + pd.Timedelta(seconds=600)
    )
    
    # Extract results
    air_network = model.components["AirNetwork"]
    estimated_flow = air_network.output["totalFlow"].history.detach().numpy()
    estimated_power = air_network.output["fanPower"].history.detach().numpy()
    
    # Calculate final accuracy
    measured_flow = measurement_data['measurements']['measured_flow'].values
    measured_power = measurement_data['measurements']['measured_power'].values
    
    final_rmse_flow = np.sqrt(np.mean((measured_flow - estimated_flow)**2))
    final_rmse_power = np.sqrt(np.mean((measured_power - estimated_power)**2))
    
    print(f"After estimation RMSE:")
    print(f"  Flow: {final_rmse_flow:.3f} m³/s")
    print(f"  Power: {final_rmse_power:.0f} W")
    
    # Plot comparison if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Flow prediction
        axes[0, 0].scatter(measured_flow, estimated_flow, alpha=0.6, s=20)
        axes[0, 0].plot([measured_flow.min(), measured_flow.max()], 
                       [measured_flow.min(), measured_flow.max()], 'r--', label='Perfect')
        axes[0, 0].set_xlabel('Measured Flow (m³/s)')
        axes[0, 0].set_ylabel('Estimated Flow (m³/s)')
        axes[0, 0].set_title('Flow Prediction')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Power prediction
        axes[0, 1].scatter(measured_power, estimated_power, alpha=0.6, s=20, color='orange')
        axes[0, 1].plot([measured_power.min(), measured_power.max()],
                       [measured_power.min(), measured_power.max()], 'r--', label='Perfect')
        axes[0, 1].set_xlabel('Measured Power (W)')
        axes[0, 1].set_ylabel('Estimated Power (W)')
        axes[0, 1].set_title('Power Prediction')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Time series comparison
        time_subset = measurement_data['inputs'].index[:100]  # First 100 points
        axes[1, 0].plot(time_subset, measured_flow[:100], 'k.', label='Measured', markersize=3)
        axes[1, 0].plot(time_subset, estimated_flow[:100], 'r-', label='Estimated', linewidth=1)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Flow (m³/s)')
        axes[1, 0].set_title('Flow Time Series (First 100 Points)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals
        flow_residuals = estimated_flow - measured_flow
        axes[1, 1].scatter(measured_flow, flow_residuals, alpha=0.6, s=20)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Measured Flow (m³/s)')
        axes[1, 1].set_ylabel('Flow Residual (m³/s)')
        axes[1, 1].set_title('Flow Prediction Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('air_network_estimation_results.png', dpi=150, bbox_inches='tight')
        print("\n✓ Results saved as 'air_network_estimation_results.png'")
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available - skipping plots")
    
    print(f"\n✓ Model validation complete!")
    print(f"✓ Final flow RMSE: {final_rmse_flow:.3f} m³/s")
    print(f"✓ Final power RMSE: {final_rmse_power:.0f} W")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_air_network_simulation():
    """
    Example of how to use the Twin4Build-compatible air network models.
    """
    import pandas as pd
    import numpy as np
    
    print("Twin4Build Air Network Models Example")
    print("=" * 50)
    
    # Define model configuration
    fan_config = {
        "design_pressure": 600,
        "design_flow": 5.0,
        "max_pressure": 900,
        "motor_efficiency": 0.9
    }
    
    supply_configs = [
        {"resistance_coeff": 15, "damper_R_open": 10.0},
        {"resistance_coeff": 20, "damper_R_open": 10.0},
        {"resistance_coeff": 25, "damper_R_open": 10.0}
    ]
    
    exhaust_configs = [
        {"resistance_coeff": 12, "damper_R_open": 10.0},
        {"resistance_coeff": 15, "damper_R_open": 10.0},
        {"resistance_coeff": 18, "damper_R_open": 10.0}
    ]
    
    # Create sample input data for simulation
    time_index = pd.date_range('2024-01-01', periods=100, freq='10min', tz='UTC')
    input_data = {
        'fan_speed': pd.Series(np.random.uniform(0.6, 1.0, 100), index=time_index),
        'supply_dampers': pd.DataFrame({
            'damper_0': np.random.uniform(0.7, 1.0, 100),
            'damper_1': np.random.uniform(0.7, 1.0, 100),
            'damper_2': np.random.uniform(0.7, 1.0, 100)
        }, index=time_index),
        'exhaust_dampers': pd.DataFrame({
            'damper_0': np.random.uniform(0.7, 1.0, 100),
            'damper_1': np.random.uniform(0.7, 1.0, 100),
            'damper_2': np.random.uniform(0.7, 1.0, 100)
        }, index=time_index)
    }
    
    # Create complete model with inputs
    model = create_air_network_model(
        fan_config=fan_config,
        supply_configs=supply_configs,
        exhaust_configs=exhaust_configs,
        input_data=input_data
    )
    
    print(f"✓ Created model with {len(model.components)} components")
    print(f"✓ Air network has {len(model.components['AirNetwork'].supply_branches)} supply branches")
    print(f"✓ Air network has {len(model.components['AirNetwork'].exhaust_branches)} exhaust branches")
    
    # Create simulator
    simulator = tb.Simulator(model)
    
    print("✓ Model ready for simulation!")
    print("✓ Model includes time series inputs for realistic simulation")
    print("✓ Can also be used for parameter estimation and optimization!")
    
    return model, simulator


def example_parameter_estimation():
    """
    Complete example of parameter estimation using Twin4Build's estimator.
    This replicates the functionality from test_air_network_model_torch.py
    """
    print("Twin4Build Air Network Parameter Estimation Example")
    print("=" * 60)
    print("This example demonstrates:")
    print("• Synthetic data generation")
    print("• Parameter estimation using Twin4Build's estimator")
    print("• Model validation and visualization")
    print("=" * 60)
    
    # Step 1: Generate synthetic measurement data
    measurement_data = generate_synthetic_data(n_samples=300)
    
    # Step 2: Run parameter estimation
    estimated_model = run_parameter_estimation(measurement_data)
    
    # Step 3: Validate estimated model
    validate_estimation(estimated_model, measurement_data)
    
    print("\n" + "="*60)
    print("PARAMETER ESTIMATION COMPLETE!")
    print("="*60)
    print("✓ Successfully estimated air network parameters")
    print("✓ Model validated against synthetic measurements")
    print("✓ Results demonstrate Twin4Build's estimation capabilities")
    print("✓ Ready for real-world parameter estimation!")
    
    return estimated_model, measurement_data


if __name__ == "__main__":
    
    estimated_model, measurement_data = example_parameter_estimation()
        
    # print("\nModel components:")
    # for comp_id, comp in model.components.items():
    #     print(f"  - {comp_id}: {type(comp).__name__}")
    
    # print("\nTo run parameter estimation example, use:")
    # print("python twin4build_air_network_models.py estimation")
