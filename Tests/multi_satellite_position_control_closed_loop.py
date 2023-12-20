# Load standard modules
import numpy as np
from matplotlib import pyplot as plt
from Scenarios.ControlScenarios import Model
from Scenarios.MainScenarios import ScenarioEnum
from Scenarios.ScenarioHandler import ScenarioHandler
from tudatpy.kernel.astro import element_conversion

import pandas as pd
desired_width = 200
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)

states = None

# Select desired scenario
# scenario = ScenarioEnum.position_keeping_scenario_translation_HCW
# scenario = ScenarioEnum.position_keeping_scenario_translation_HCW_scaled

# scenario = ScenarioEnum.position_keeping_scenario_translation_ROE
# scenario = ScenarioEnum.position_keeping_scenario_translation_ROE_scaled

# scenario = ScenarioEnum.simple_scenario_translation_HCW
# scenario = ScenarioEnum.simple_scenario_translation_HCW_scaled

# scenario = ScenarioEnum.simple_scenario_translation_ROE
scenario = ScenarioEnum.simple_scenario_translation_ROE_scaled

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
# scenario = ScenarioEnum.simple_scenario_translation_blend_small_scaled
# scenario = ScenarioEnum.j2_scenario_translation_blend_small_scaled

# Groups
# scenario = ScenarioEnum.simple_scenario_HCW_6_orbits
# scenario = ScenarioEnum.simple_scenario_HCW_2_orbits
# scenario = ScenarioEnum.simple_scenario_moving_blend_2_orbits
# scenario = ScenarioEnum.simple_scenario_ROE_2_orbits
# scenario = ScenarioEnum.simple_scenario_ROE_6_orbits
# scenario = ScenarioEnum.simple_scenario_moving_blend_6_orbits
# scenario = ScenarioEnum.j2_scenario_moving_HCW_2_orbits
# scenario = ScenarioEnum.j2_scenario_moving_blend_2_orbits
# scenario = ScenarioEnum.j2_scenario_moving_blend_6_orbits
# scenario = ScenarioEnum.j2_scenario_moving_blend_6_orbits_ca

# J2
# scenario = ScenarioEnum.j2_comparison_blend_on
# scenario = ScenarioEnum.j2_comparison_blend_off

# Robust
# scenario = ScenarioEnum.robustness_comparison_no_robust
# scenario = ScenarioEnum.robustness_comparison_simple_robust
# scenario = ScenarioEnum.robustness_comparison_advanced_robust

# scenario = ScenarioEnum.robustness_comparison_no_robust_noise
# scenario = ScenarioEnum.robustness_comparison_simple_robust_noise
# scenario = ScenarioEnum.robustness_comparison_advanced_robust_noise

# Large
# scenario = ScenarioEnum.large_scenario_nominal
# scenario = ScenarioEnum.large_scenario_robust_constraint_noise

# Collisions
# scenario = ScenarioEnum.collision_test_out_of_plane

# Individual tests
# scenario = ScenarioEnum.blend_model_Omega_lim

# Setup
scenario_handler = ScenarioHandler(scenario.value)
scenario_handler.create_sls_system()
scenario_handler.create_storage_variables()

# Run simulation
scenario_handler.simulate_system_closed_loop()
# scenario_handler.simulate_system_no_control()
# scenario_handler.simulate_system_controller_sim()
# scenario_handler.simulate_system_single_shot()
# scenario_handler.simulate_system_controller_then_full_sim()
# scenario_handler.simulate_system_controller_sim_no_control(np.array([-0.1, 0.0, 0, 0.0, 0.0, 0.0]).reshape((6, 1)))

# scenario_handler.controller.x_states = scenario_handler.sls_states.T
# states = scenario_handler.controller.plot_states()
# inputs = scenario_handler.controller.plot_inputs()


orbital_sim = scenario_handler.export_results()



# reference = scenario_handler.controller.x_ref[scenario_handler.controller.angle_states]
# reference = np.tile(np.linspace(0, 2*np.pi, 5, endpoint=False), 2)

# if scenario.value.model == Model.DIFFERENTIAL_DRAG:
#     reference = np.concatenate((np.array([0]), reference.reshape((-1,))))
# state_fig = orbital_sim.plot_cylindrical_states(figure=None)
# orbital_sim.plot_3d_orbit()
# plt.show()
# orbital_sim.plot_quasi_roe_states(figure=None)
# orbital_sim.plot_roe_states(reference_angles=reference, figure=None)
# orbital_sim.plot_keplerian_states(plot_argument_of_latitude=False)
# orbital_sim.plot_kalman_states()
# orbital_sim.plot_blend_states(figure=None)
# orbital_sim.plot_theta_Omega()
# orbital_sim.plot_small_blend_states(figure=None)
# input_fig = orbital_sim.plot_thrusts(figure=None)

orbital_sim.plot_controller_states(figure=states)
# orbital_sim.plot_model_errors(scenario_handler.controller.x_states)

orbital_sim.plot_main_states()
#
# if scenario_handler.scenario.collision_avoidance is False:
#     scenario_handler.scenario.collision_avoidance = True
#     scenario_handler.init_sim()
# orbital_sim.plot_out_of_plane_constraints()
# orbital_sim.plot_in_plane_constraints()
# orbital_sim.plot_side_states()
# orbital_sim.plot_inputs()

# print(orbital_sim.print_metrics())

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

# Save data
# oe_sat = orbital_sim.dep_vars[:,  orbital_sim.dependent_variables_dict['keplerian state'][orbital_sim.controlled_satellite_names[0]][0]:
#                                   orbital_sim.dependent_variables_dict['keplerian state'][orbital_sim.controlled_satellite_names[-1]][-1] + 1]
# _, oe_ref = orbital_sim.filter_oe()
# inputs = orbital_sim.thrust_forces
# np.savez("..\\Data\\Temp\\control5", oe_sat=oe_sat, oe_ref=oe_ref, inputs=inputs)
