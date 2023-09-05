import matplotlib.pyplot as plt
from Scenarios.MainScenarios import ScenarioEnum
from Scenarios.ScenarioHandler import ScenarioHandler
import pickle

scenarios_to_run = [ScenarioEnum.simple_scenario_translation_HCW_scaled,
                    # ScenarioEnum.simple_scenario_translation_ROE_scaled,
                    ScenarioEnum.simple_scenario_translation_blend_scaled]

main_naming_identifier = 'singlePlaneKeplerModel'
fig_name_list = ['MainStates', 'SideStates', 'Inputs']
fig_list = [None] * len(fig_name_list)
scenario_name_list = ['HCW',
                      # 'ROE',
                      'BLEND']
satellites_to_plot = None

print("singlePlaneKepler.py: starting")
for idx, scenario in enumerate(scenarios_to_run):
    print(f"singlePlaneKepler.py: {scenario_name_list[idx]}")
    scenario_handler = ScenarioHandler(scenario.value)
    scenario_handler.create_sls_system()
    scenario_handler.create_storage_variables()

    # Run simulation
    scenario_handler.simulate_system_closed_loop(print_progress=False)
    orbital_sim = scenario_handler.export_results()
    fig_list[0] = orbital_sim.plot_main_states(figure=fig_list[0])
    fig_list[1] = orbital_sim.plot_side_states(figure=fig_list[1])
    fig_list[2] = orbital_sim.plot_inputs(figure=fig_list[2])

    # Save orbital sim with all data
    file_name = '../Data/' + main_naming_identifier + scenario_name_list[idx]
    with open(file_name, 'wb') as file:
        pickle.dump(orbital_sim, file)

for idx, fig in enumerate(fig_list):
    fig.savefig('../Figures/' + main_naming_identifier + fig_name_list[idx] + '.eps')

print("singlePlaneKepler.py: done")
# plt.show()

