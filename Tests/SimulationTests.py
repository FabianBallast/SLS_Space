from Simulator.MOE_simulator import mean_orbital_elements_simulator
from Scenarios.MainScenarios import ScenarioEnum
from Scenarios.ScenarioHandler import ScenarioHandler
import matplotlib.pyplot as plt
import Visualisation.Plotting as Plot
import numpy as np
import time
# Select scenario
scenario = ScenarioEnum.simple_scenario_translation_HCW_scaled
# scenario = ScenarioEnum.j2_scenario_moving_HCW_scaled

# Setup
scenario_handler = ScenarioHandler(scenario.value)
scenario_handler.create_sls_system()
scenario_handler.create_storage_variables()

# Run simulation
start_tudat = time.time()
scenario_handler.simulate_system_closed_loop()
# scenario_handler.simulate_system_controller_then_full_sim()
end_tudat = time.time()
orbital_sim = scenario_handler.export_results()


fig = orbital_sim.plot_keplerian_states(plot_argument_of_latitude=False)
# fig_states = orbital_sim.plot_blend_states()

# Run simulation with MOE as well.
satellites_per_plane = orbital_sim.number_of_controlled_satellites // orbital_sim.number_of_orbits
x0 = np.tile(orbital_sim.initial_reference_state, (satellites_per_plane, )).reshape((-1, ))
x0[5::6] = orbital_sim.reference_angle_offsets.flatten()

inputs = orbital_sim.get_thrust_forces_from_acceleration().transpose((0, 2, 1)).reshape((-1, 3 * scenario.value.number_of_satellites))[1:]
# data = np.load("..\\Data\\Temp\\simulation_input.npz")
# x0 = data['x0']
# inputs = data['inputs']

start_moe = time.time()
MOE_res = mean_orbital_elements_simulator(x0, inputs, scenario.value)
end_moe = time.time()

# fig = None
for satellite in range(scenario.value.number_of_satellites):
    fig = Plot.plot_keplerian_states(MOE_res[:, 6 * satellite: 6 * (satellite + 1)],
                                     scenario.value.simulation.simulation_timestep,
                                     plot_argument_of_latitude=False,
                                     legend_name=None,
                                     figure=fig,
                                     linestyle='--')

# orbital_sim.filtered_oe = MOE_res
# orbital_sim.blend_states = None
# fig_states = orbital_sim.plot_blend_states(figure=fig_states)

    # a = MOE_res[:, 6 * satellite: 6 * satellite + 1]
    # e = MOE_res[:, 6 * satellite + 1: 6 * satellite + 2]
    # i = MOE_res[:, 6 * satellite + 2: 6 * satellite + 3]
    # omega = MOE_res[:, 6 * satellite + 3: 6 * satellite + 4]
    # Omega = MOE_res[:, 6 * satellite + 4: 6 * satellite + 5]
    # f = MOE_res[:, 6 * satellite + 5: 6 * satellite + 6]
    #
    # r = a * (1-e**2) / (1+e * np.cos(f)) - 55
    # theta_Omega = f + omega + Omega
    # ex = e * np.cos(f)
    # ey = e * np.sin(f)
    #
    # blend_states = np.concatenate((r, theta_Omega, ex, ey, i, ))
    #
    # blend_sta
    # fig_states = Plot.plot_blend(scenario.value.simulation.simulation_timestep,
    #                              legend_name=None,
    #                              figure=fig_states,
    #                              linestyle='-.')

# np.savez("..\\Data\\Temp\\simulation_input", x0=x0, inputs=inputs)

plt.show()

print(f"Duration of tudatpy: {end_tudat - start_tudat} s")
print(f"Duration of moe: {end_moe - start_moe} s")