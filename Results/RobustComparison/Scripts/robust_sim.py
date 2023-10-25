import matplotlib.pyplot as plt
from Scenarios.MainScenarios import ScenarioEnum
from Scenarios.ScenarioHandler import ScenarioHandler
from Results.RobustComparison.Scripts.plotFromData import *
import pickle

scenarios_to_run = [#ScenarioEnum.robustness_comparison_no_robust,
                    #ScenarioEnum.robustness_comparison_simple_robust,
                    ScenarioEnum.robustness_comparison_advanced_robust
                    ]

main_naming_identifier = 'robustness'
scenario_name_list = [#'NO',
                      #'SIMPLE',
                      'ADVANCED'
                      ]
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
plt.show()

print(f"{main_naming_identifier}: done")







