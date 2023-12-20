from Scenarios.MainScenarios import ScenarioEnum
from Scenarios.ScenarioHandler import ScenarioHandler
import pickle
import matplotlib.pyplot as plt
# from Results.LargeSimulation.Scripts.plotFromData import plot_theta_Omega

scenarios_to_run = [ScenarioEnum.blend_model_Omega_lim]

main_naming_identifier = 'blend_model_Omega_lim'
scenario_name_list = ['all']
satellites_to_plot = None

print(f"{main_naming_identifier}: starting")
for idx, scenario in enumerate(scenarios_to_run):
    print(f"{main_naming_identifier}: {scenario_name_list[idx]}")
    scenario_handler = ScenarioHandler(scenario.value)
    scenario_handler.create_sls_system()
    scenario_handler.create_storage_variables()

    # Run simulation
    scenario_handler.simulate_system_closed_loop(print_progress=True)
    orbital_sim = scenario_handler.export_results()

    # Save orbital sim with all data
    file_name = '../Data/' + main_naming_identifier + '_' + scenario_name_list[idx]
    with open(file_name, 'wb') as file:
        pickle.dump(orbital_sim, file)

# plot_theta_Omega(main_naming_identifier + '_' + scenario_name_list[0])
# plt.show()

print(f"{main_naming_identifier}: done")







