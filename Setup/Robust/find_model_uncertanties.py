import numpy as np
from Scenarios.MainScenarios import ScenarioEnum
from Simulator.Dynamics_simulator import create_dynamics_simulator
from Dynamics.BlendDynamics import Blend
from Utils.Conversions import oe2blend

with open('allowed_states.npy', 'rb') as f:
    blend_states_allowed = np.load(f)
    oe_states_allowed = np.load(f)


scenario = ScenarioEnum.robustness_comparison_simple_robust.value

# Create simulation for various x0
dynamics = Blend(scenario)
scenario_limits = dynamics.get_state_constraint()
A_matrix = dynamics.create_model(scenario.control.control_timestep).A
B_matrix = dynamics.create_model(scenario.control.control_timestep).B


orbit_diff = dynamics.get_orbital_differentiation() * scenario.control.control_timestep
mean_motion = np.sqrt(100 / 55**3) * scenario.control.control_timestep
ref_state_start = np.array([55, 0, np.deg2rad(45), 0, 0, 0]).reshape((1, 1, 6))
ref_state = np.array([55, 0, np.deg2rad(45), orbit_diff[4], orbit_diff[3], orbit_diff[5] + mean_motion]).reshape((1, 1, 6))

# full_array = full_array.reshape((1, -1))
states_models = (A_matrix @ blend_states_allowed.T)

dyn_sim = create_dynamics_simulator(oe_states_allowed.reshape((-1, 1)), np.zeros((1, 3, oe_states_allowed.shape[0])), scenario, scenario.control.control_timestep)
mean_OE_state_after = dyn_sim.state_history[scenario.control.control_timestep]
real_states = oe2blend(mean_OE_state_after.reshape((1, -1)), ref_state, np.zeros(blend_states_allowed.shape[0])).reshape((-1, 6)).T
state_norms = np.max(np.abs(blend_states_allowed.reshape((-1, 6)).T), axis=0)

# mean_OE_state_after_full = np.array(list(dyn_sim.state_history.values()))
# real_states_full = oe2blend(mean_OE_state_after_full, ref_state, np.zeros(blend_states_allowed.shape[0])).reshape((mean_OE_state_after_full.shape[0], -1, 6)).T

# dyn_sim_zero = create_dynamics_simulator(np.array([55, 0, np.deg2rad(45), 0, 0, 0]).reshape((-1, 1)), np.zeros((1, 3, 1)), scenario, scenario.control.control_timestep)
# mean_OE_state_after_zero = dyn_sim_zero.state_history[scenario.control.control_timestep]
# zero_real_state = oe2blend(mean_OE_state_after_zero.reshape((1, -1)), ref_state, np.zeros(mean_OE_state_after_zero.shape[0])).reshape((-1, 6)).T

# Compute uncertainties
for i in range(6):
    print(f"||A[{i}, :]||_\infty = {np.max(np.abs(real_states[i] - states_models[i]) / state_norms)}")

with open('allowed_inputs.npy', 'rb') as f:
    blend_states_allowed_input = np.load(f)
    oe_states_allowed_input = np.load(f)
    allowed_inputs = np.load(f)


# full_array = full_array.reshape((1, -1))
states_models = A_matrix @ blend_states_allowed_input.reshape((-1, 6)).T + B_matrix @ allowed_inputs.T

dyn_sim = create_dynamics_simulator(oe_states_allowed_input.reshape((-1, 1)), allowed_inputs.T.reshape((1, 3, -1)).T, scenario, scenario.control.control_timestep)
mean_OE_state_after = dyn_sim.state_history[scenario.control.control_timestep]
real_states = oe2blend(mean_OE_state_after.reshape((1, -1)), ref_state, np.zeros(allowed_inputs.shape[0])).reshape((-1, 6)).T
state_norms = np.max(np.abs(allowed_inputs.T), axis=0)

zero_bool_array = np.zeros((states_models.shape[1]), dtype=bool)
zero_indices = np.arange(3**3 // 2, states_models.shape[1], 3**3)
zero_bool_array[zero_indices] = True
zero_inputs_model = states_models[:, zero_bool_array]
nonzero_input_model = states_models[:, ~zero_bool_array].reshape((6, -1, zero_inputs_model.shape[1])).transpose((1, 0, 2))
model_input_delta = (nonzero_input_model - zero_inputs_model).transpose((1, 2, 0)).reshape((6, -1))

zero_inputs_real = real_states[:, zero_bool_array]
nonzero_input_real = real_states[:, ~zero_bool_array].reshape((6, -1, zero_inputs_real.shape[1])).transpose((1, 0, 2))
real_input_delta = (nonzero_input_real - zero_inputs_real).transpose((1, 2, 0)).reshape((6, -1))

# Compute uncertainties
for i in range(6):
    print(f"||B[{i}, :]||_\infty = {np.max(np.abs(model_input_delta[i] - real_input_delta[i]) / state_norms[~zero_bool_array])}")
