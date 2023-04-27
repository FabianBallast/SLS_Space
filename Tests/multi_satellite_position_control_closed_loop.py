# Load standard modules
from matplotlib import pyplot as plt
from Scenarios.MainScenarios import *
from Scenarios.ScenarioHandler import ScenarioHandler

# Select desired scenario
# scenario = position_keeping_scenario_translation_HCW
# scenario = position_keeping_scenario_translation_ROE
scenario = simple_scenario_translation_HCW
# scenario = simple_scenario_translation_ROE

# Setup
scenario_handler = ScenarioHandler(scenario)
scenario_handler.create_sls_system()
scenario_handler.create_storage_variables()

# Run simulation
scenario_handler.simulate_system()

orbital_sim = scenario_handler.export_results()

orbital_sim.plot_cylindrical_states()
orbital_sim.plot_quasi_roe_states()
input_fig = orbital_sim.plot_thrusts()
# anim = orbital_sim.create_animation()
plt.show()

