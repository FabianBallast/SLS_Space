import numpy as np
from Scenarios.MainScenarios import ScenarioEnum
from Utils.Conversions import oe2blend
from Dynamics.BlendDynamics import Blend

scenario = ScenarioEnum.robustness_comparison_simple_robust.value
dynamics = Blend(scenario)
scenario_limits = dynamics.get_state_constraint()

# OE values to test if their correct in blend coordinates
num_tests_per_limit = 10

radial_states = np.linspace(-0.1, 0.1, num=num_tests_per_limit)
angular_states = np.linspace(-np.deg2rad(30), np.deg2rad(30), num=num_tests_per_limit)
e_states = np.linspace(0, 0.015, num=2*num_tests_per_limit)
i_states = np.linspace(-np.deg2rad(5), np.deg2rad(5), num=num_tests_per_limit)
omega_states = np.linspace(-np.deg2rad(30), np.deg2rad(30), num=num_tests_per_limit)
Omega_states = np.linspace(-np.deg2rad(30), np.deg2rad(30), num=num_tests_per_limit)

ref_state_start = np.array([55, 0, np.deg2rad(45), 0, 0, 0])

# Create array with all these options
av, ev, iv, omega_v, Omega_v, f_v = np.meshgrid(radial_states + ref_state_start[0],
                                                e_states + ref_state_start[1],
                                                i_states + ref_state_start[2],
                                                omega_states + ref_state_start[3],
                                                Omega_states + ref_state_start[4],
                                                angular_states + ref_state_start[5])

test_values = np.concatenate((av.reshape(-1, 1),
                              ev.reshape(-1, 1),
                              iv.reshape(-1, 1),
                              omega_v.reshape(-1, 1),
                              Omega_v.reshape(-1, 1),
                              f_v.reshape(-1, 1)), axis=1)

states_blend = oe2blend(test_values.reshape((1, -1)), ref_state_start.reshape((1, 1, 6)), np.zeros(test_values.shape[0])).reshape((-1, 6))

rho_good = (-scenario_limits[0] <= states_blend[:, 0]) & (states_blend[:, 0] <= scenario_limits[0])
lambda_good = (-scenario_limits[1] <= states_blend[:, 1]) & (states_blend[:, 1] <= scenario_limits[1])
ex_good = (-scenario_limits[2] <= states_blend[:, 2]) & (states_blend[:, 2] <= scenario_limits[2])
ey_good = (-scenario_limits[3] <= states_blend[:, 3]) & (states_blend[:, 3] <= scenario_limits[3])
xix_good = (-scenario_limits[4] <= states_blend[:, 4]) & (states_blend[:, 4] <= scenario_limits[4])
xiy_good = (-scenario_limits[5] <= states_blend[:, 5]) & (states_blend[:, 5] <= scenario_limits[5])

allowed_state_indices = rho_good & lambda_good & ex_good & ey_good & xix_good & xiy_good

allowed_states_blend = states_blend[allowed_state_indices, :]
allowed_state_oe = test_values[allowed_state_indices, :]

with open('allowed_states.npy', 'wb') as f:
    np.save(f, allowed_states_blend)
    np.save(f, allowed_state_oe)


# OE values to test if their correct in blend coordinates
num_tests_input = 3

zero_array = np.zeros(1)
ur_inputs = np.linspace(-0.1, 0.1, num=num_tests_input)
ut_inputs = np.linspace(-0.1, 0.1, num=num_tests_input)
uz_inputs = np.linspace(-0.1, 0.1, num=num_tests_input)

ref_state_start = np.array([55, 0, np.deg2rad(45), 0, 0, 0])

# Create array with all these options
av, ev, iv, omega_v, Omega_v, f_v, ur_v, ut_v, uz_v = np.meshgrid(radial_states + ref_state_start[0],
                                                                  e_states + ref_state_start[1],
                                                                  i_states + ref_state_start[2],
                                                                  zero_array, zero_array, zero_array,
                                                                  ur_inputs, ut_inputs, uz_inputs)

test_values = np.concatenate((av.reshape(-1, 1),
                              ev.reshape(-1, 1),
                              iv.reshape(-1, 1),
                              omega_v.reshape(-1, 1),
                              Omega_v.reshape(-1, 1),
                              f_v.reshape(-1, 1)), axis=1)

test_inputs = np.concatenate((ur_v.reshape((-1, 1)),
                              ut_v.reshape((-1, 1)),
                              uz_v.reshape((-1, 1))), axis=1)

states_blend = oe2blend(test_values.reshape((1, -1)), ref_state_start.reshape((1, 1, 6)), np.zeros(test_values.shape[0])).reshape((-1, 6))

rho_good = (-scenario_limits[0] <= states_blend[:, 0]) & (states_blend[:, 0] <= scenario_limits[0])
lambda_good = (-scenario_limits[1] <= states_blend[:, 1]) & (states_blend[:, 1] <= scenario_limits[1])
ex_good = (-scenario_limits[2] <= states_blend[:, 2]) & (states_blend[:, 2] <= scenario_limits[2])
ey_good = (-scenario_limits[3] <= states_blend[:, 3]) & (states_blend[:, 3] <= scenario_limits[3])
xix_good = (-scenario_limits[4] <= states_blend[:, 4]) & (states_blend[:, 4] <= scenario_limits[4])
xiy_good = (-scenario_limits[5] <= states_blend[:, 5]) & (states_blend[:, 5] <= scenario_limits[5])

allowed_state_indices = rho_good & lambda_good & ex_good & ey_good & xix_good & xiy_good

allowed_states_blend = states_blend[allowed_state_indices, :]
allowed_state_oe = test_values[allowed_state_indices, :]
allowed_inputs = test_inputs[allowed_state_indices, :]

with open('allowed_inputs.npy', 'wb') as f:
    np.save(f, allowed_states_blend)
    np.save(f, allowed_state_oe)
    np.save(f, allowed_inputs)
