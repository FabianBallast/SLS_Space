from Scenarios.OrbitalScenarios import orbital_scenarios, print_orbital_scenarios
from Scenarios.PhysicsScenarios import physics_scenarios, print_physics_scenarios
from Scenarios.InitialStateScenarios import initial_state_scenarios, print_initial_state_scenarios
from Scenarios.SimulationScenarios import simulation_scenarios, print_simulation_scenarios
from Scenarios.ControlScenarios import control_scenarios, Model, print_control_scenarios

simple_scenario_attitude = {'orbital': orbital_scenarios['arbitrary_orbit'],
                            'physics': physics_scenarios['basic_physics'],
                            'simulation': simulation_scenarios['sim_1_minute'],
                            'initial_state': initial_state_scenarios['no_state_error'],
                            'control': control_scenarios['control_attitude_default'],
                            'number_of_sats': 3,
                            'model': Model.ATTITUDE}

advanced_scenario_attitude = {'orbital': orbital_scenarios['arbitrary_orbit'],
                              'physics': physics_scenarios['advanced_grav_physics'],
                              'simulation': simulation_scenarios['sim_1_minute'],
                              'initial_state': initial_state_scenarios['large_state_error'],
                              'control': control_scenarios['control_attitude_default'],
                              'number_of_sats': 3,
                              'model': Model.ATTITUDE}

position_keeping_scenario_translation_HCW = {'orbital': orbital_scenarios['tilted_orbit_45deg'],
                                             'physics': physics_scenarios['basic_physics'],
                                             'simulation': simulation_scenarios['sim_10_minute'],
                                             'initial_state': initial_state_scenarios['no_state_error'],
                                             'control': control_scenarios['control_position_default'],
                                             'number_of_sats': 5,
                                             'model': Model.HCW}

position_keeping_scenario_translation_ROE = {'orbital': orbital_scenarios['tilted_orbit_45deg'],
                                             'physics': physics_scenarios['basic_physics'],
                                             'simulation': simulation_scenarios['sim_10_minute'],
                                             'initial_state': initial_state_scenarios['no_state_error'],
                                             'control': control_scenarios['control_position_default'],
                                             'number_of_sats': 5,
                                             'model': Model.ROE}

simple_scenario_translation_HCW = {'orbital': orbital_scenarios['tilted_orbit_45deg'],
                                   'physics': physics_scenarios['basic_physics'],
                                   'simulation': simulation_scenarios['sim_1_hour'],
                                   'initial_state': initial_state_scenarios['small_state_error'],
                                   'control': control_scenarios['control_position_default'],
                                   'number_of_sats': 5,
                                   'model': Model.HCW}

simple_scenario_translation_ROE = {'orbital': orbital_scenarios['tilted_orbit_45deg'],
                                   'physics': physics_scenarios['basic_physics'],
                                   'simulation': simulation_scenarios['sim_10_minute'],
                                   'initial_state': initial_state_scenarios['small_state_error'],
                                   'control': control_scenarios['control_position_default'],
                                   'number_of_sats': 5,
                                   'model': Model.ROE}

medium_scenario_translation_HCW = {'orbital': orbital_scenarios['tilted_orbit_45deg'],
                                   'physics': physics_scenarios['advanced_grav_physics'],
                                   'simulation': simulation_scenarios['sim_1_hour'],
                                   'initial_state': initial_state_scenarios['small_state_error'],
                                   'control': control_scenarios['control_position_default'],
                                   'number_of_sats': 5,
                                   'model': Model.HCW}

medium_scenario_translation_ROE = {'orbital': orbital_scenarios['tilted_orbit_45deg'],
                                   'physics': physics_scenarios['advanced_grav_physics'],
                                   'simulation': simulation_scenarios['sim_1_hour'],
                                   'initial_state': initial_state_scenarios['small_state_error'],
                                   'control': control_scenarios['control_position_default'],
                                   'number_of_sats': 5,
                                   'model': Model.ROE}

advanced_scenario_translation_HCW = {'orbital': orbital_scenarios['tilted_orbit_45deg'],
                                     'physics': physics_scenarios['full_physics'],
                                     'simulation': simulation_scenarios['sim_1_hour'],
                                     'initial_state': initial_state_scenarios['large_state_error'],
                                     'control': control_scenarios['control_position_default'],
                                     'number_of_sats': 5,
                                     'model': Model.HCW}

advanced_scenario_translation_ROE = {'orbital': orbital_scenarios['tilted_orbit_45deg'],
                                     'physics': physics_scenarios['full_physics'],
                                     'simulation': simulation_scenarios['sim_1_hour'],
                                     'initial_state': initial_state_scenarios['large_state_error'],
                                     'control': control_scenarios['control_position_default'],
                                     'number_of_sats': 5,
                                     'model': Model.ROE}
