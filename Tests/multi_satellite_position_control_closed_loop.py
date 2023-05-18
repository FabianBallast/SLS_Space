# Load standard modules
import numpy as np
from matplotlib import pyplot as plt
from Scenarios.ControlScenarios import Model
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

# scenario = ScenarioEnum.simple_scenario_translation_SimAn_scaled

# Setup
scenario_handler = ScenarioHandler(scenario.value)
scenario_handler.create_sls_system()
scenario_handler.create_storage_variables()

# Run simulation
scenario_handler.simulate_system_closed_loop()
# scenario_handler.simulate_system_no_control()
# scenario_handler.simulate_system_controller_sim()
#
# scenario_handler.controller.plot_states()
# scenario_handler.controller.plot_inputs()


orbital_sim = scenario_handler.export_results()

reference = scenario_handler.controller.x_ref[scenario_handler.controller.angle_states]

if scenario.value.model == Model.DIFFERENTIAL_DRAG:
    reference = np.concatenate((np.array([0]), reference.reshape((-1,))))
orbital_sim.plot_cylindrical_states(reference_angles=reference)
# orbital_sim.plot_keplerian_states(satellite_names=['Satellite_1', 'Satellite_ref'])
# orbital_sim.plot_quasi_roe_states(satellite_names=['Satellite_1'])
input_fig = orbital_sim.plot_thrusts()
# anim = orbital_sim.create_animation()
plt.show()

