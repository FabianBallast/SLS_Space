import matplotlib.pyplot as plt
from Scenarios.MainScenarios import ScenarioEnum
from Scenarios.ScenarioHandler import ScenarioHandler
from Results.ModelComparison.Scripts.plotFromData import plot_j2_comparison
import pickle
import numpy as np

scenarios_to_run = [ScenarioEnum.j2_comparison_HCW_off,
                    #ScenarioEnum.j2_comparison_blend_on,
                    #ScenarioEnum.j2_comparison_blend_off,
                    #ScenarioEnum.j2_comparison_roe_on,
                    #ScenarioEnum.j2_comparison_roe_off,
                    ]

main_naming_identifier = 'j2_comparison'
scenario_name_list = ['HCW',
                      'BLEND(J2)',
                      'BLEND(NO J2)',
                      'ROE(J2)',
                      'ROE(NO J2)'
                      ]
satellites_to_plot = None

print(f"{main_naming_identifier}: starting")
for idx, scenario in enumerate(scenarios_to_run):
    print(f"{main_naming_identifier}: {scenario_name_list[idx]}")
    scenario_handler = ScenarioHandler(scenario.value)
    scenario_handler.create_sls_system()
    scenario_handler.create_storage_variables()

    # Run simulation
    # scenario_handler.simulate_system_closed_loop(print_progress=False)
    # scenario_handler.simulate_system_no_control()
    scenario_handler.simulate_system_controller_sim_no_control(np.array([-0.1, 0.0, 0, 0.0, 0.0, 0.0]).reshape((6, 1)))
    orbital_sim = scenario_handler.export_results()

    print(scenario_handler.orbital_mech.print_metrics())

    # Save orbital sim with all data
    file_name = '../Data/' + main_naming_identifier + '_' + scenario_name_list[idx]
    with open(file_name, 'wb') as file:
        pickle.dump(orbital_sim, file)
        pickle.dump(scenario_handler.controller.x_states, file)


plot_j2_comparison()


print(f"{main_naming_identifier}: done")
plt.show()

