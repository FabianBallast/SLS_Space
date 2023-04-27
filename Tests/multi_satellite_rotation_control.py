# Load standard modules
from matplotlib import pyplot as plt
from Scenarios.MainScenarios import *
from Scenarios.ScenarioHandler import ScenarioHandler

# Select desired scenario
# scenario = simple_scenario_attitude
# scenario = advanced_scenario_attitude
# scenario = simple_scenario_attitude_scaled
scenario = advanced_scenario_attitude_scaled

# Setup
scenario_handler = ScenarioHandler(scenario)
scenario_handler.create_sls_system()
scenario_handler.create_storage_variables()

# Run simulation
scenario_handler.simulate_system()

orbital_sim = scenario_handler.export_results()

# Plot results
orbital_sim.plot_euler_angles()
orbital_sim.plot_angular_velocities()
orbital_sim.plot_torques()
plt.show()
