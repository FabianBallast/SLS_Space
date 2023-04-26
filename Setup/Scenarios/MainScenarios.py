from Scenarios.OrbitalScenarios import orbital_scenarios, print_orbital_scenarios
from Scenarios.PhysicsScenarios import physics_scenarios, print_physics_scenarios
from Scenarios.InitialStateScenarios import initial_state_scenarios, print_initial_state_scenarios
from Scenarios.SimulationScenarios import simulation_scenarios, print_simulation_scenarios
from Scenarios.ControlScenarios import control_scenarios, print_control_scenarios

simple_scenario_attitude = {'orbital': orbital_scenarios['equatorial_orbit'],
                            'physics': physics_scenarios['basic_physics'],
                            'simulation': simulation_scenarios['sim_1_minute'],
                            'initial_state': initial_state_scenarios['no_state_error'],
                            'control': control_scenarios['control_attitude_default']}

advanced_scenario_attitude = {'orbital': orbital_scenarios['equatorial_orbit'],
                              'physics': physics_scenarios['advanced_grav_physics'],
                              'simulation': simulation_scenarios['sim_1_minute'],
                              'initial_state': initial_state_scenarios['large_state_error'],
                              'control': control_scenarios['control_attitude_default']}
