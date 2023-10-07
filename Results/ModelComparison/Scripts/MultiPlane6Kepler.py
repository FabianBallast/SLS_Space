import matplotlib.pyplot as plt
from Scenarios.MainScenarios import ScenarioEnum
from Scenarios.ScenarioHandler import ScenarioHandler
from Results.ModelComparison.Scripts.plotFromData import plot_data_comparison
import pickle

scenarios_to_run = [#ScenarioEnum.simple_scenario_HCW_6_orbits,
                    ScenarioEnum.simple_scenario_ROE_6_orbits,
                    ScenarioEnum.simple_scenario_moving_blend_6_orbits]

main_naming_identifier = 'hex_plane_kepler'
scenario_name_list = [#'HCW',
                      'ROE',
                      'BLEND']
satellites_to_plot = None

print(f"{main_naming_identifier}: starting")
for idx, scenario in enumerate(scenarios_to_run):
    print(f"{main_naming_identifier}: {scenario_name_list[idx]}")
    scenario_handler = ScenarioHandler(scenario.value)
    scenario_handler.create_sls_system()
    scenario_handler.create_storage_variables()

    # Run simulation
    scenario_handler.simulate_system_closed_loop(print_progress=False)
    orbital_sim = scenario_handler.export_results()

    # Save orbital sim with all data
    file_name = '../Data/' + main_naming_identifier + '_' + scenario_name_list[idx]
    with open(file_name, 'wb') as file:
        pickle.dump(orbital_sim, file)

plot_data_comparison(main_naming_identifier)


print(f"{main_naming_identifier}: done")
# plt.show()

