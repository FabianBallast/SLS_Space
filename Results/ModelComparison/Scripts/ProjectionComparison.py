import matplotlib.pyplot as plt
from Scenarios.MainScenarios import ScenarioEnum
from Scenarios.ScenarioHandler import ScenarioHandler
from Results.ModelComparison.Scripts.plotFromData import plot_projection_comparison
import pickle

scenarios_to_run = [ScenarioEnum.projection_comparison]

main_naming_identifier = 'projection_comparison'
satellites_to_plot = None

print(f"{main_naming_identifier}: starting")
for idx, scenario in enumerate(scenarios_to_run):
    scenario_handler = ScenarioHandler(scenario.value)
    scenario_handler.create_sls_system()
    scenario_handler.create_storage_variables()

    # Run simulation
    scenario_handler.simulate_system_no_control()
    orbital_sim = scenario_handler.export_results()

    # Fix cylindrical states
    orbital_sim.cylindrical_states = None
    orbital_sim.initial_reference_state[:, 4] = 0
    orbital_sim.convert_to_cylindrical_coordinates()

    # Save orbital sim with all data
    file_name = '../Data/' + main_naming_identifier
    with open(file_name, 'wb') as file:
        pickle.dump(orbital_sim, file)

plot_projection_comparison()


print(f"{main_naming_identifier}: done")
plt.show()

