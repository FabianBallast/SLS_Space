# Load standard modules
from matplotlib import pyplot as plt
from Scenarios.MainScenarios import ScenarioEnum
from Scenarios.ScenarioHandler import ScenarioHandler

# Select desired scenario
# scenario = ScenarioEnum.position_keeping_scenario_translation_HCW
# scenario = ScenarioEnum.position_keeping_scenario_translation_HCW_scaled
scenario = ScenarioEnum.position_keeping_scenario_translation_ROE
# scenario = ScenarioEnum.position_keeping_scenario_translation_ROE_scaled
# scenario = ScenarioEnum.simple_scenario_translation_HCW
# scenario = ScenarioEnum.simple_scenario_translation_HCW_scaled
# scenario = ScenarioEnum.simple_scenario_translation_ROE
# scenario = ScenarioEnum.simple_scenario_translation_ROE_scaled

# Setup
scenario_handler = ScenarioHandler(scenario.value)
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

