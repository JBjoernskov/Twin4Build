# Local application imports
import twin4build as tb

"""
MODEL CREATION
"""
model = tb.Model(id="example")  # Create a model
c1 = tb.SpaceHeaterSystem(...)  # Create a space heater
c2 = tb.BuildingSpaceSystem(...)  # Create a building space
model.add_connection(
    c1, c2, "Power", "heatGain"
)  # Add a connection between the space heater and the building space
model.load()  # Load the model

"""
SIMULATION
"""
simulator = tb.Simulator(model)  # Create a simulator
simulator.simulate(...)  # Run the simulator

"""
ESTIMATION
"""
estimator = tb.Estimator(simulator)  # Create an estimator
estimator.estimate(...)  # Run the estimator

"""
OPTIMIZATION
"""
optimizer = tb.Optimizer(simulator)  # Create an optimizer
optimizer.optimize(...)  # Run the optimizer

# Plot the results
tb.plot.plot(...)
