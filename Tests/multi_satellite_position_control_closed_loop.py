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
# scenario = ScenarioEnum.simple_scenario_translation_HCW_scaled

# scenario = ScenarioEnum.simple_scenario_translation_ROE
# scenario = ScenarioEnum.simple_scenario_translation_ROE_scaled

# scenario = ScenarioEnum.j2_scenario_pos_keep_HCW
# scenario = ScenarioEnum.j2_scenario_pos_keep_HCW_scaled
# scenario = ScenarioEnum.j2_scenario_moving_HCW_scaled
# scenario = ScenarioEnum.j2_scenario_moving_ROE_scaled

# scenario = ScenarioEnum.simple_scenario_translation_SimAn_scaled
# scenario = ScenarioEnum.j2_scenario_translation_SimAn_scaled
# scenario = ScenarioEnum.simple_scenario_translation_ROEV2_scaled
# scenario = ScenarioEnum.j2_scenario_translation_ROEV2_scaled

# scenario = ScenarioEnum.simple_scenario_translation_blend_scaled
# scenario = ScenarioEnum.j2_scenario_translation_blend_scaled

# Groups
# scenario = ScenarioEnum.j2_scenario_moving_HCW_2_orbits
scenario = ScenarioEnum.j2_scenario_moving_blend_2_orbits
# scenario = ScenarioEnum.j2_scenario_moving_blend_6_orbits
# scenario = ScenarioEnum.j2_scenario_moving_blend_6_orbits_ca

# Setup
scenario_handler = ScenarioHandler(scenario.value)
scenario_handler.create_sls_system()
scenario_handler.create_storage_variables()

# Run simulation
# scenario_handler.simulate_system_closed_loop()
# scenario_handler.simulate_system_no_control()
# scenario_handler.simulate_system_controller_sim()
# scenario_handler.simulate_system_single_shot()
scenario_handler.simulate_system_controller_then_full_sim()

# scenario_handler.controller.x_states = scenario_handler.sls_states.T
# states = scenario_handler.controller.plot_states()
# inputs = scenario_handler.controller.plot_inputs()


orbital_sim = scenario_handler.export_results()

reference = scenario_handler.controller.x_ref[scenario_handler.controller.angle_states]
# reference = np.tile(np.linspace(0, 2*np.pi, 5, endpoint=False), 2)

if scenario.value.model == Model.DIFFERENTIAL_DRAG:
    reference = np.concatenate((np.array([0]), reference.reshape((-1,))))
# state_fig = orbital_sim.plot_cylindrical_states(reference_angles=reference, figure=None)
# orbital_sim.plot_3d_orbit()
# orbital_sim.plot_keplerian_states()

# mean_motion = np.rad2deg(scenario_handler.controller.dynamics.mean_motion),
# orbital_der = np.rad2deg(scenario_handler.controller.dynamics.get_orbital_differentiation())
#
# time_past = 30 * 60
# Omega_end = 20 + orbital_der[3] * time_past
# angle_end = (mean_motion + orbital_der[4] + orbital_der[5]) * time_past
# print(f"{Omega_end=}, {angle_end=}")
# orbital_sim.plot_quasi_roe_states(reference_angles=reference, figure=None)
# orbital_sim.plot_roe_states(reference_angles=reference, figure=None)
orbital_sim.plot_keplerian_states(plot_argument_of_latitude=False)
# orbital_sim.plot_kalman_states()
orbital_sim.plot_blend_states(reference_angles=reference, figure=None)
# input_fig = orbital_sim.plot_thrusts(figure=None)
# anim = orbital_sim.create_animation()
# print(orbital_sim.print_metrics())

# Second scenario
# scenario = ScenarioEnum.simple_scenario_translation_ROE_scaled
# scenario_handler = ScenarioHandler(scenario.value)
# scenario_handler.create_sls_system()
# scenario_handler.create_storage_variables()
#
# # Run simulation
# scenario_handler.simulate_system_closed_loop()
# orbital_sim = scenario_handler.export_results()
#
# reference = scenario_handler.controller.x_ref[scenario_handler.controller.angle_states]
#
# if scenario.value.model == Model.DIFFERENTIAL_DRAG:
#     reference = np.concatenate((np.array([0]), reference.reshape((-1,))))
# state_fig = orbital_sim.plot_cylindrical_states(reference_angles=reference, figure=state_fig)
# input_fig = orbital_sim.plot_thrusts(figure=input_fig)
#
plt.show()

