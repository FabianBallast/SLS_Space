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


number_of_satellites = oe_states_allowed.shape[0]
disturbance_limit = np.array([0.0001, 0.0001, 0.00001, 0.00001, 0.0001, 0.0001] * number_of_satellites)



orbit_diff = dynamics.get_orbital_differentiation() * scenario.control.control_timestep
mean_motion = np.sqrt(100 / 55**3) * scenario.control.control_timestep
ref_state_start = np.array([55, 0, np.deg2rad(45), 0, 0, 0]).reshape((1, 1, 6))
ref_state = np.array([55, 0, np.deg2rad(45), orbit_diff[4], orbit_diff[3], orbit_diff[5] + mean_motion]).reshape((1, 1, 6))


dyn_sim_no_disturbance = create_dynamics_simulator(oe_states_allowed.reshape((-1, 1)), np.zeros((1, 3, oe_states_allowed.shape[0])), scenario, scenario.control.control_timestep)
mean_OE_state_after_no_disturbance = dyn_sim_no_disturbance.state_history[scenario.control.control_timestep]
real_states_no_disturbance = oe2blend(mean_OE_state_after_no_disturbance.reshape((1, -1)), ref_state, np.zeros(blend_states_allowed.shape[0])).reshape((-1, 6)).T

dyn_sim_disturbance = create_dynamics_simulator(oe_states_allowed.reshape((-1, 1)), np.zeros((1, 3, oe_states_allowed.shape[0])), scenario, scenario.control.control_timestep, disturbance_limit)
mean_OE_state_after_disturbance = dyn_sim_disturbance.state_history[scenario.control.control_timestep]
real_states_disturbance = oe2blend(mean_OE_state_after_disturbance.reshape((1, -1)), ref_state, np.zeros(blend_states_allowed.shape[0])).reshape((-1, 6)).T

for i in range(6):
    print(f"||\delta_{i}||_\infty = {np.max(np.abs(real_states_disturbance[i] - real_states_no_disturbance[i]))}")
