# Load standard modules
import numpy as np
from matplotlib import pyplot as plt
from Scenarios.ControlScenarios import Model
from Scenarios.MainScenarios import ScenarioEnum
from Scenarios.ScenarioHandler import ScenarioHandler

satellites_to_plot = ['Satellite_3', 'Satellite_5', 'Satellite_8']

scenario = ScenarioEnum.simple_scenario_translation_HCW_scaled

# Setup
scenario_handler = ScenarioHandler(scenario.value)
scenario_handler.create_sls_system()
scenario_handler.create_storage_variables()

# Run simulation
scenario_handler.simulate_system_closed_loop()
orbital_sim = scenario_handler.export_results()

reference = scenario_handler.controller.x_ref[scenario_handler.controller.angle_states]

if scenario.value.model == Model.DIFFERENTIAL_DRAG:
    reference = np.concatenate((np.array([0]), reference.reshape((-1,))))
state_fig = orbital_sim.plot_cylindrical_states(satellite_names=satellites_to_plot, figure=None,
                                                reference_angles=reference, legend_name=r'$\mathrm{HCW \;model}$',
                                                states2plot=[0, 1], linestyle='--')
input_fig = orbital_sim.plot_thrusts(figure=None, satellite_names=satellites_to_plot, legend_name=r'$\mathrm{HCW \;model}$', linestyle='--')
metrics_HCW = orbital_sim.print_metrics()

# # Second scenario
scenario = ScenarioEnum.simple_scenario_translation_ROE_scaled
scenario_handler = ScenarioHandler(scenario.value)
scenario_handler.create_sls_system()
scenario_handler.create_storage_variables()

# Run simulation
scenario_handler.simulate_system_closed_loop()
orbital_sim = scenario_handler.export_results()

reference = scenario_handler.controller.x_ref[scenario_handler.controller.angle_states]

if scenario.value.model == Model.DIFFERENTIAL_DRAG:
    reference = np.concatenate((np.array([0]), reference.reshape((-1,))))
state_fig = orbital_sim.plot_cylindrical_states(reference_angles=reference, figure=state_fig,
                                                satellite_names=satellites_to_plot, legend_name=r'$\mathrm{ROE \;model}$',
                                                states2plot=[0, 1])
input_fig = orbital_sim.plot_thrusts(figure=input_fig, satellite_names=satellites_to_plot, legend_name=r'$\mathrm{ROE \;model}$')
metrics_ROE = orbital_sim.print_metrics()

# Baseline
scenario = ScenarioEnum.simple_scenario_translation_SimAn_scaled
scenario_handler = ScenarioHandler(scenario.value)
scenario_handler.create_sls_system()
scenario_handler.create_storage_variables()

# Run simulation
scenario_handler.simulate_system_closed_loop()
orbital_sim = scenario_handler.export_results()

reference = scenario_handler.controller.x_ref[scenario_handler.controller.angle_states]

if scenario.value.model == Model.DIFFERENTIAL_DRAG:
    reference = np.concatenate((np.array([0]), reference.reshape((-1,))))
state_fig = orbital_sim.plot_cylindrical_states(reference_angles=reference, figure=state_fig,
                                                satellite_names=satellites_to_plot, legend_name=r'$\mathrm{Simulated \;Annealing}$',
                                                states2plot=[1], linestyle='-.')
input_fig = orbital_sim.plot_thrusts(figure=input_fig, satellite_names=satellites_to_plot, legend_name=r'$\mathrm{Simulated \;Annealing}$', linestyle='-.')
metrics_SimAN = orbital_sim.print_metrics()

print("HCW")
print(metrics_HCW)
print()
print('ROE')
print(metrics_ROE)
print()
print('Simulated Annealing')
print(metrics_SimAN)

plt.show()

