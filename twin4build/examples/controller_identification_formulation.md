# Controller Identification via Continuous Relaxation

## Problem Statement

Given a Brick model with actuators and sensors, identify the control logic for each actuator by estimating:

1. Which sensor(s) provide feedback
2. Which setpoint signal(s) are being tracked
3. Which controller type(s) are active
4. The controller parameters

## Formulation

### Decision Variables

**Selection weights** (continuous relaxation of binary choices):

- $\alpha_k \in [0,1]$ — contribution of controller module $k$
- $\beta_i \in [0,1]$ — contribution of sensor $i$ as feedback signal
- $\gamma_j \in [0,1]$ — contribution of setpoint signal $j$

**Controller parameters:**

- $\theta_k$ — parameters for controller module $k$ (e.g., $K_p$, $K_i$, $K_d$)

### Control Error

The weighted error signal:

$$e_t = \sum_j \gamma_j \cdot sp_{jt} - \sum_i \beta_i \cdot y_{it}$$

where $sp_{jt}$ is setpoint candidate $j$ at time $t$ and $y_{it}$ is sensor $i$ at time $t$.

### Controller Modules

Candidate modules $f_k(e_t; \theta_k)$:

| Module | Output |
|--------|--------|
| Bias | $b$ |
| Proportional (P) | $K_p \cdot e_t$ |
| Integral (I) | $K_i \cdot \sum_{\tau=1}^{t} e_\tau \cdot \Delta t$ |
| Derivative (D) | $K_d \cdot \frac{e_t - e_{t-1}}{\Delta t}$ |

Additional modules can be added (feedforward, gated modules, etc.).

### Predicted Actuator Output

$$\hat{u}_t = \sum_k \alpha_k \cdot f_k(e_t; \theta_k)$$

### Objective Function

$$\min_{\alpha, \beta, \gamma, \theta} \sum_t (u_t - \hat{u}_t)^2 + \lambda \left[ \sum_k P(\alpha_k) + \sum_i P(\beta_i) + \sum_j P(\gamma_j) \right]$$

### Binarization Penalty

Instead of L1 regularization, use a quadratic penalty that encourages weights toward discrete values (0 or 1):

$$P(x) = x(1 - x)$$

Properties:

- $P(0) = 0$ (no penalty at 0)
- $P(1) = 0$ (no penalty at 1)
- $P(0.5) = 0.25$ (maximum penalty at midpoint)
- $\frac{dP}{dx} = 1 - 2x$ (zero gradient at $x = 0.5$)

## Initialization Strategy

All selection weights initialized at 0.5:

$$\alpha_k^{(0)} = 0.5 \quad \forall k$$
$$\beta_i^{(0)} = 0.5 \quad \forall i$$
$$\gamma_j^{(0)} = 0.5 \quad \forall j$$

**Key insight:** At $x = 0.5$, the penalty gradient is zero. The initial optimization direction is determined purely by the data fit term. Once weights move away from 0.5, the penalty gradient accelerates them toward 0 or 1.

## Algorithm

1. Initialize all $\alpha$, $\beta$, $\gamma$ at 0.5
2. Initialize $\theta$ with reasonable defaults (e.g., $K_p = 1$, $K_i = 0.1$)
3. Optimize using gradient descent (e.g., Adam)
4. After convergence, threshold weights: $\alpha_k \gets \mathbb{1}[\alpha_k > 0.5]$
5. Optionally re-estimate $\theta$ with fixed binary structure

## Constraints

- $\alpha_k \in [0, 1]$
- $\beta_i \in [0, 1]$
- $\gamma_j \in [0, 1]$

No simplex constraints — multiple modules/sensors/setpoints can be active simultaneously.

## Interpretation of Results

After optimization:

- $\alpha_k \approx 1$: Module $k$ is active in the controller
- $\alpha_k \approx 0$: Module $k$ is not used
- $\beta_i \approx 1$: Sensor $i$ provides feedback
- $\gamma_j \approx 1$: Setpoint $j$ is being tracked
- $\theta_k$: Tuned parameters for active modules

## Coverage

This formulation handles:

- P, PI, PID controllers and subsets
- Multiple feedback sensors
- Multiple or scheduled setpoints
- Bias/offset terms
- Combinations of the above

Extensions needed for:

- Cascade control (composition)
- Min/max selection
- Hard override logic