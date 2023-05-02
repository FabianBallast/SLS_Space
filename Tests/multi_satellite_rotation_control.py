# Load standard modules
from matplotlib import pyplot as plt
from Scenarios.MainScenarios import ScenarioEnum
from Scenarios.ScenarioHandler import ScenarioHandler

# Select desired scenario
# scenario = ScenarioEnum.simple_scenario_attitude
# scenario = ScenarioEnum.advanced_scenario_attitude
# scenario = ScenarioEnum.simple_scenario_attitude_scaled
scenario = ScenarioEnum.advanced_scenario_attitude_scaled

# Setup
scenario_handler = ScenarioHandler(scenario.value)
scenario_handler.create_sls_system()
scenario_handler.create_storage_variables()

# Run simulation
scenario_handler.simulate_system_closed_loop()
# scenario_handler.simulate_system_no_control()

orbital_sim = scenario_handler.export_results()

# Plot results
# orbital_sim.plot_quaternions()
# orbital_sim.plot_quaternions_rsw()
orbital_sim.plot_cylindrical_states()
orbital_sim.plot_euler_angles()
orbital_sim.plot_angular_velocities()
orbital_sim.plot_torques()
plt.show()
