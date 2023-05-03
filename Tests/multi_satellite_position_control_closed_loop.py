# Load standard modules
from matplotlib import pyplot as plt
from Scenarios.MainScenarios import ScenarioEnum
from Scenarios.ScenarioHandler import ScenarioHandler

# Select desired scenario
# scenario = ScenarioEnum.position_keeping_scenario_translation_HCW
# scenario = ScenarioEnum.position_keeping_scenario_translation_HCW_scaled
# scenario = ScenarioEnum.position_keeping_scenario_translation_ROE
# scenario = ScenarioEnum.position_keeping_scenario_translation_ROE_scaled
# scenario = ScenarioEnum.simple_scenario_translation_HCW
scenario = ScenarioEnum.simple_scenario_translation_HCW_scaled
# scenario = ScenarioEnum.simple_scenario_translation_ROE
# scenario = ScenarioEnum.simple_scenario_translation_ROE_scaled
# scenario = ScenarioEnum.j2_scenario_pos_keep_HCW
# scenario = ScenarioEnum.j2_scenario_pos_keep_HCW_scaled
# scenario = ScenarioEnum.j2_scenario_moving_HCW_scaled

# Setup
scenario_handler = ScenarioHandler(scenario.value)
scenario_handler.create_sls_system()
scenario_handler.create_storage_variables()

# Run simulation
scenario_handler.simulate_system_closed_loop()
# scenario_handler.simulate_system_no_control()
# scenario_handler.simulate_system_controller_sim()

scenario_handler.sls.plot_states()
scenario_handler.sls.plot_inputs()

# orbital_sim = scenario_handler.export_results()
#
# orbital_sim.plot_cylindrical_states(reference_angles=scenario_handler.sls.x_ref[scenario_handler.sls.angle_states],
#                                     satellite_names=['Satellite_1'])
# orbital_sim.plot_keplerian_states(satellite_names=['Satellite_1', 'Satellite_ref'])
# orbital_sim.plot_quasi_roe_states(satellite_names=['Satellite_1'])
# input_fig = orbital_sim.plot_thrusts()
# anim = orbital_sim.create_animation()
# plt.show()

