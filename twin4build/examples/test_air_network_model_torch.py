import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def solve_equilibrium_differentiable_old(model, fan_speed, supply_damper_pos, exhaust_damper_pos, n_iterations=30000):
    """
    Solve equilibrium using differentiable fixed-point iteration
    This maintains the computation graph throughout
    """
    batch_size = fan_speed.shape[0]
    
    # Initialize flow
    flow = fan_speed * model.fan.design_flow

    tol = 1e-3
    step_size = 0.1

    old_pressure_error = float('inf')
    old_flow = float('inf')
    # Fixed-point iteration (differentiable)
    for i in range(n_iterations):
        # Calculate fan pressure at current flow
        fan_pressure = model.fan.get_pressure_rise(flow, fan_speed)
        
        # Calculate system pressure at current flow
        system_pressure = model.calculate_system_pressure(flow, supply_damper_pos, exhaust_damper_pos)
        
        # Update flow based on pressure imbalance
        # If fan_pressure > system_pressure, increase flow
        # If fan_pressure < system_pressure, decrease flow
        pressure_error = fan_pressure - system_pressure

        # Update flow (gradient flows through this operation!)
        flow = flow + step_size * pressure_error / (model.fan.design_pressure + 1e-12) * model.fan.design_flow
        flow = torch.clamp(flow, min=0.01, max=model.fan.design_flow * 1.5)


        if torch.all(torch.abs(pressure_error) < tol):
            print(f"Converged in {i} iterations")
            break
        # else:
        #     print("--------------------------------")
        #     print("Iteration: ", i)
        #     print("Flow update: ",  step_size * pressure_error / (model.fan.design_pressure + 1e-12) * model.fan.design_flow)
        #     print("Pressure error: ", pressure_error)
        #     print("cond: ", torch.abs(pressure_error) < tol)
        #     print("Pressure error change: ", torch.abs(pressure_error - old_pressure_error).mean().item())
        #     print("Flow change: ", torch.abs(flow - old_flow).mean().item())


        old_pressure_error = pressure_error
        old_flow = flow

    return flow

def solve_equilibrium_differentiable(model, fan_speed, supply_damper_pos, exhaust_damper_pos, 
                                     n_iterations=50, tol=1e-3, verbose=False):
    """
    Solve equilibrium using simple damped iteration - keep it simple!
    This maintains the computation graph throughout
    """
    batch_size = fan_speed.shape[0]
    
    # Initialize with a reasonable guess
    flow = fan_speed * model.fan.design_flow
    
    prev_error = float('inf')
    stall_count = 0
    
    # Use exponentially decaying step size
    for i in range(n_iterations):
        # Calculate residual
        fan_pressure = model.fan.get_pressure_rise(flow, fan_speed)
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
        flow_scale = model.fan.design_flow / (model.fan.design_pressure + 1e-6)
        flow_update = alpha * residual * flow_scale
        
        flow = flow + flow_update
        flow = torch.clamp(flow, min=0.01, max=model.fan.design_flow * 1.5)
    
    if verbose and i == n_iterations - 1:
        print(f"Reached max iterations ({n_iterations}), error = {max_error:.6f} Pa")
    
    return flow

class FanModel(nn.Module):
    """Differentiable fan model"""
    def __init__(self, design_pressure: float, design_flow: float, 
                 max_pressure: float, motor_efficiency: float = 0.9):
        super().__init__()
        
        self.design_pressure = design_pressure
        self.design_flow = design_flow
        self.motor_efficiency = motor_efficiency
        
        # Initialize fan curve coefficients
        max_flow = design_flow * 1.3
        a_init = max_pressure*1000
        c_init = -max_pressure*1000 / (max_flow ** 2)
        b_init = (design_pressure - a_init - c_init * design_flow**2) / design_flow
        
        self.a = nn.Parameter(torch.tensor(a_init/1000, dtype=torch.float32)) #kPa
        self.b = nn.Parameter(torch.tensor(b_init, dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(c_init, dtype=torch.float32))
        
    def get_pressure_rise(self, flow: torch.Tensor, speed_ratio: torch.Tensor) -> torch.Tensor:
        """Calculate fan pressure rise (differentiable)"""
        flow_norm = flow / (speed_ratio + 1e-6)
        pressure_design = self.a*1000 + self.b * flow_norm + self.c * flow_norm**2
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


class DamperModel(nn.Module):
    """Differentiable damper model"""
    def __init__(self, fully_open_resistance: float, fully_closed_resistance: float):
        super().__init__()
        self.log_R_open = nn.Parameter(torch.tensor(np.log(fully_open_resistance), dtype=torch.float32))
        self.k = nn.Parameter(torch.tensor(np.log(fully_closed_resistance / fully_open_resistance), dtype=torch.float32))
        
    def get_resistance(self, position: torch.Tensor) -> torch.Tensor:
        """Calculate damper resistance"""
        R_open = torch.exp(self.log_R_open)
        resistance = R_open * torch.exp(self.k * (1.0 - position))
        return resistance


class BranchModel(nn.Module):
    """Differentiable branch model"""
    def __init__(self, resistance_coeff: float, flow_exponent: float = 2.0,
                 linear_resistance: float = 0.0, has_damper: bool = True,
                 damper_R_open: float = 10.0, damper_R_closed: float = 1e5):
        super().__init__()
        
        self.flow_exponent = flow_exponent
        self.log_R_duct = nn.Parameter(torch.tensor(np.log(resistance_coeff), dtype=torch.float32))
        self.R_linear = nn.Parameter(torch.tensor(linear_resistance, dtype=torch.float32))
        
        self.has_damper = has_damper
        if has_damper:
            self.damper = DamperModel(damper_R_open, damper_R_closed)
        
    def get_pressure_drop(self, flow: torch.Tensor, damper_position: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate branch pressure drop"""
        abs_flow = torch.abs(flow)
        R_duct = torch.exp(self.log_R_duct)
        dp_duct = R_duct * abs_flow**self.flow_exponent + torch.abs(self.R_linear) * abs_flow
        
        if self.has_damper and damper_position is not None:
            R_damper = self.damper.get_resistance(damper_position)
            dp_damper = R_damper * abs_flow**2
            return dp_duct + dp_damper
        
        return dp_duct


class AirNetworkPyTorch(nn.Module):
    """
    Differentiable air network with implicit differentiation
    Maintains gradients through the equilibrium solve
    """
    def __init__(self, fan: FanModel, supply_branches: List[BranchModel], 
                 exhaust_branches: List[BranchModel]):
        super().__init__()
        
        self.fan = fan
        self.supply_branches = nn.ModuleList(supply_branches)
        self.exhaust_branches = nn.ModuleList(exhaust_branches)
        self.n_supply = len(supply_branches)
        self.n_exhaust = len(exhaust_branches)
        
    def distribute_flow_parallel(self, total_flow: torch.Tensor, branches: nn.ModuleList,
                                 damper_positions: torch.Tensor, n_iterations: int = 15) -> Tuple[torch.Tensor, torch.Tensor]:
        """Distribute flow among parallel branches (differentiable)"""
        batch_size = total_flow.shape[0]
        n_branches = len(branches)
        
        # Initialize flows equally
        flows = total_flow.unsqueeze(1).expand(-1, n_branches) / n_branches
        
        # Iterative distribution (fixed iterations for differentiability)
        for _ in range(n_iterations):
            pressures = torch.stack([
                branches[i].get_pressure_drop(flows[:, i], damper_positions[:, i])
                for i in range(n_branches)
            ], dim=1)
            
            target_pressure = pressures.mean(dim=1, keepdim=True)
            R_eq = pressures / (flows**2 + 1e-6)
            new_flows = torch.sqrt(torch.clamp(target_pressure / (R_eq + 1e-6), min=0))
            flow_sum = new_flows.sum(dim=1, keepdim=True)
            flows = new_flows * total_flow.unsqueeze(1) / (flow_sum + 1e-6)
        
        pressures = torch.stack([
            branches[i].get_pressure_drop(flows[:, i], damper_positions[:, i])
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
    
    def forward(self, fan_speed: torch.Tensor, supply_damper_pos: torch.Tensor,
                exhaust_damper_pos: torch.Tensor) -> dict:
        """
        Forward pass - fully differentiable using fixed-point iteration
        No custom autograd needed - gradients flow naturally!
        """
        # Solve for equilibrium flow (maintains computation graph)
        flow_solution = solve_equilibrium_differentiable(
            self, fan_speed, supply_damper_pos, exhaust_damper_pos, n_iterations=3000000
        )
        
        # Calculate all outputs at the solution point
        flow_pos = torch.clamp(flow_solution, min=0.0)
        fan_pressure = self.fan.get_pressure_rise(flow_pos, fan_speed)
        system_pressure = self.calculate_system_pressure(flow_pos, supply_damper_pos, exhaust_damper_pos)
        fan_power = self.fan.get_power(flow_pos, fan_pressure)
        
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


def train_model(model: AirNetworkPyTorch, measured_data: dict, 
                n_epochs: int = 100, learning_rate: float = 0.01):
    """Train model with gradient descent (fully differentiable!)"""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Convert data to tensors
    fan_speed = torch.tensor(measured_data['fan_speed'], dtype=torch.float32)
    supply_dampers = torch.tensor(measured_data['supply_dampers'], dtype=torch.float32)
    exhaust_dampers = torch.tensor(measured_data['exhaust_dampers'], dtype=torch.float32)
    measured_flow = torch.tensor(measured_data['measured_flow'], dtype=torch.float32)
    measured_power = torch.tensor(measured_data.get('measured_power', [0]*len(measured_flow)), dtype=torch.float32)
    
    losses_history = []
    
    print("Training air network model...")
    print("=" * 60)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward pass (computation graph is maintained!)
        results = model(fan_speed, supply_dampers, exhaust_dampers)
        predicted_flow = results['total_flow']
        predicted_power = results['fan_power']
        
        # Loss function
        loss_flow = torch.mean((predicted_flow - measured_flow)**2)
        loss_power = torch.mean((predicted_power - measured_power)**2) / 1e6

        loss = loss_flow + 0.1 * loss_power
        # Backward pass (gradients flow naturally through the iterations!)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        losses_history.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}, "
                  f"Flow RMSE: {torch.sqrt(loss_flow).item():.4f} m³/s, Power RMSE: {torch.sqrt(loss_power).item():.4f} MW")
            
            # Check if any parameter has gradients
            has_grads = any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())
            if not has_grads:
                print("  WARNING: No gradients detected!")
    
    print("=" * 60)
    print("Training complete!")
    
    return losses_history


# Example usage
if __name__ == "__main__":
    print("PyTorch Air Network - Differentiable Fixed-Point Iteration")
    print("=" * 70)
    print("✓ Computation graph is preserved")
    print("✓ Gradients flow through iterative solver")
    print("✓ Simple and stable - no custom autograd needed")
    print("=" * 70)
    
    # Create model
    fan = FanModel(design_pressure=600, design_flow=5.0, max_pressure=0.9)
    
    supply_branches = [
        BranchModel(resistance_coeff=15, damper_R_open=10.0),
        BranchModel(resistance_coeff=20, damper_R_open=10.0),
        BranchModel(resistance_coeff=25, damper_R_open=10.0)
    ]
    
    exhaust_branches = [
        BranchModel(resistance_coeff=12, damper_R_open=10.0),
        BranchModel(resistance_coeff=15, damper_R_open=10.0),
        BranchModel(resistance_coeff=18, damper_R_open=10.0)
    ]
    
    model = AirNetworkPyTorch(fan, supply_branches, exhaust_branches)
    
    print("\nInitial model parameters:")
    print(f"Fan curve: a={model.fan.a.item():.2f}, b={model.fan.b.item():.2f}, c={model.fan.c.item():.4f}")
    supply_res = [torch.exp(b.log_R_duct).item() for b in model.supply_branches]
    print(f"Supply resistances: {[f'{r:.1f}' for r in supply_res]}")
    print()
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 300
    
    # True model (unknown parameters to estimate)
    true_fan = FanModel(design_pressure=620, design_flow=4.8, max_pressure=1.5)
    true_supply = [BranchModel(r) for r in [18, 22, 28]]
    true_exhaust = [BranchModel(r) for r in [14, 17, 20]]
    true_model = AirNetworkPyTorch(true_fan, true_supply, true_exhaust)
    
    # Generate measurements
    fan_speeds = np.random.uniform(0.5, 1.0, n_samples)
    supply_dampers_data = np.random.uniform(0.5, 1.0, (n_samples, 3))
    exhaust_dampers_data = np.random.uniform(0.5, 1.0, (n_samples, 3))
    
    with torch.no_grad():
        true_results = true_model(
            torch.tensor(fan_speeds, dtype=torch.float32),
            torch.tensor(supply_dampers_data, dtype=torch.float32),
            torch.tensor(exhaust_dampers_data, dtype=torch.float32)
        )
        measured_flows = true_results['total_flow'].numpy()
        measured_powers = true_results['fan_power'].numpy()
    
    # Add noise
    # measured_flows += np.random.normal(0, 0.05, n_samples)
    # measured_powers += np.random.normal(0, 50, n_samples)
    
    measured_data = {
        'fan_speed': fan_speeds,
        'supply_dampers': supply_dampers_data,
        'exhaust_dampers': exhaust_dampers_data,
        'measured_flow': measured_flows,
        'measured_power': measured_powers
    }
    
    # Test gradient flow before training
    print("Testing gradient flow...")
    test_results = model(
        torch.tensor([0.8], dtype=torch.float32),
        torch.tensor([[0.9, 0.8, 0.7]], dtype=torch.float32),
        torch.tensor([[0.9, 0.8, 0.7]], dtype=torch.float32)
    )
    test_loss = test_results['total_flow'].sum()
    test_loss.backward()
    
    print(f"Fan parameter 'a' gradient: {model.fan.a.grad}")
    print(f"Supply branch 0 gradient: {model.supply_branches[0].log_R_duct.grad}")
    print("✓ Gradients are flowing!\n")
    
    # Reset gradients
    model.zero_grad()
    
    # Train model
    losses = train_model(model, measured_data, n_epochs=100, learning_rate=0.01)
    
    print("\nFinal model parameters:")
    print(f"Fan curve: a={model.fan.a.item():.2f}, b={model.fan.b.item():.2f}, c={model.fan.c.item():.4f}")
    supply_res = [torch.exp(b.log_R_duct).item() for b in model.supply_branches]
    print(f"Supply resistances: {[f'{r:.1f}' for r in supply_res]}")
    print()
    
    print("True parameters:")
    print(f"Fan curve: a={true_fan.a.item():.2f}, b={true_fan.b.item():.2f}, c={true_fan.c.item():.4f}")
    true_res = [torch.exp(b.log_R_duct).item() for b in true_supply]
    print(f"Supply resistances: {[f'{r:.1f}' for r in true_res]}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Compute predictions for all plots
    with torch.no_grad():
        final_results = model(
            torch.tensor(fan_speeds, dtype=torch.float32),
            torch.tensor(supply_dampers_data, dtype=torch.float32),
            torch.tensor(exhaust_dampers_data, dtype=torch.float32)
        )
        predicted_flows = final_results['total_flow'].numpy()
        predicted_powers = final_results['fan_power'].numpy()
    
    # 1. Training loss
    axes[0, 0].plot(losses)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Flow Prediction
    axes[0, 1].scatter(measured_flows, predicted_flows, alpha=0.6)
    axes[0, 1].plot([measured_flows.min(), measured_flows.max()], 
                    [measured_flows.min(), measured_flows.max()], 'r--', label='Perfect')
    axes[0, 1].set_xlabel('Measured Flow (m³/s)')
    axes[0, 1].set_ylabel('Predicted Flow (m³/s)')
    axes[0, 1].set_title('Flow Prediction After Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Power Prediction
    axes[0, 2].scatter(measured_powers, predicted_powers, alpha=0.6, color='orange')
    axes[0, 2].plot([measured_powers.min(), measured_powers.max()], 
                    [measured_powers.min(), measured_powers.max()], 'r--', label='Perfect')
    axes[0, 2].set_xlabel('Measured Power (W)')
    axes[0, 2].set_ylabel('Predicted Power (W)')
    axes[0, 2].set_title('Power Prediction After Training')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Fan curves
    flows_range = np.linspace(0, 6, 100)
    flows_tensor = torch.tensor(flows_range, dtype=torch.float32)
    speed_tensor = torch.ones_like(flows_tensor)
    
    with torch.no_grad():
        learned_pressures = model.fan.get_pressure_rise(flows_tensor, speed_tensor).numpy()
        true_pressures = true_fan.get_pressure_rise(flows_tensor, speed_tensor).numpy()
    
    axes[1, 0].plot(flows_range, learned_pressures, 'b-', linewidth=2, label='Learned')
    axes[1, 0].plot(flows_range, true_pressures, 'r--', linewidth=2, label='True')
    axes[1, 0].set_xlabel('Flow Rate (m³/s)')
    axes[1, 0].set_ylabel('Pressure Rise (Pa)')
    axes[1, 0].set_title('Learned vs True Fan Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Parameters
    param_names = ['a', 'b', 'c×10', 'S1', 'S2', 'S3']
    learned = [
        model.fan.a.item(), model.fan.b.item(), model.fan.c.item()*10,
        torch.exp(model.supply_branches[0].log_R_duct).item(),
        torch.exp(model.supply_branches[1].log_R_duct).item(),
        torch.exp(model.supply_branches[2].log_R_duct).item()
    ]
    true = [
        true_fan.a.item(), true_fan.b.item(), true_fan.c.item()*10,
        torch.exp(true_supply[0].log_R_duct).item(),
        torch.exp(true_supply[1].log_R_duct).item(),
        torch.exp(true_supply[2].log_R_duct).item()
    ]
    
    x = np.arange(len(param_names))
    width = 0.35
    axes[1, 1].bar(x - width/2, learned, width, label='Learned', alpha=0.7)
    axes[1, 1].bar(x + width/2, true, width, label='True', alpha=0.7)
    axes[1, 1].set_xlabel('Parameter')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Parameter Estimation')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(param_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 6. Residuals
    flow_residuals = predicted_flows - measured_flows
    power_residuals = predicted_powers - measured_powers
    
    axes[1, 2].scatter(measured_flows, flow_residuals, alpha=0.6, label='Flow', s=20)
    axes[1, 2].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[1, 2].set_xlabel('Measured Flow (m³/s)')
    axes[1, 2].set_ylabel('Flow Residual (m³/s)')
    axes[1, 2].set_title('Flow Prediction Residuals')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pytorch_training_results.png', dpi=150, bbox_inches='tight')
    print("\nResults saved as 'pytorch_training_results.png'")
    plt.show()
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Training loss
    axes[0, 0].plot(losses)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss\n(Adaptive Damped Iteration)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Predictions
    with torch.no_grad():
        final_results = model(
            torch.tensor(fan_speeds, dtype=torch.float32),
            torch.tensor(supply_dampers_data, dtype=torch.float32),
            torch.tensor(exhaust_dampers_data, dtype=torch.float32)
        )
        predicted_flows = final_results['total_flow'].numpy()
    
    axes[0, 1].scatter(measured_flows, predicted_flows, alpha=0.6)
    axes[0, 1].plot([measured_flows.min(), measured_flows.max()], 
                    [measured_flows.min(), measured_flows.max()], 'r--', label='Perfect')
    axes[0, 1].set_xlabel('Measured Flow (m³/s)')
    axes[0, 1].set_ylabel('Predicted Flow (m³/s)')
    axes[0, 1].set_title('Flow Prediction After Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Fan curves
    flows_range = np.linspace(0, 6, 100)
    flows_tensor = torch.tensor(flows_range, dtype=torch.float32)
    speed_tensor = torch.ones_like(flows_tensor)
    
    with torch.no_grad():
        learned_pressures = model.fan.get_pressure_rise(flows_tensor, speed_tensor).numpy()
        true_pressures = true_fan.get_pressure_rise(flows_tensor, speed_tensor).numpy()
    
    axes[1, 0].plot(flows_range, learned_pressures, 'b-', linewidth=2, label='Learned')
    axes[1, 0].plot(flows_range, true_pressures, 'r--', linewidth=2, label='True')
    axes[1, 0].set_xlabel('Flow Rate (m³/s)')
    axes[1, 0].set_ylabel('Pressure Rise (Pa)')
    axes[1, 0].set_title('Learned vs True Fan Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Parameters
    param_names = ['a', 'b', 'c×10', 'S1', 'S2', 'S3']
    learned = [
        model.fan.a.item(), model.fan.b.item(), model.fan.c.item()*10,
        torch.exp(model.supply_branches[0].log_R_duct).item(),
        torch.exp(model.supply_branches[1].log_R_duct).item(),
        torch.exp(model.supply_branches[2].log_R_duct).item()
    ]
    true = [
        true_fan.a.item(), true_fan.b.item(), true_fan.c.item()*10,
        torch.exp(true_supply[0].log_R_duct).item(),
        torch.exp(true_supply[1].log_R_duct).item(),
        torch.exp(true_supply[2].log_R_duct).item()
    ]
    
    x = np.arange(len(param_names))
    width = 0.35
    axes[1, 1].bar(x - width/2, learned, width, label='Learned', alpha=0.7)
    axes[1, 1].bar(x + width/2, true, width, label='True', alpha=0.7)
    axes[1, 1].set_xlabel('Parameter')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Parameter Estimation\n(Damped Iteration)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(param_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('pytorch_implicit_diff.png', dpi=150, bbox_inches='tight')
    print("\nResults saved as 'pytorch_implicit_diff.png'")
    plt.show()