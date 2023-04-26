sim_1_minute = {'start_epoch': 0,
                'simulation_duration': 60,
                'simulation_timestep': 1}

sim_10_minute = {'start_epoch': 0,
                 'simulation_duration': 600,
                 'simulation_timestep': 1}

sim_30_minute = {'start_epoch': 0,
                 'simulation_duration': 1800,
                 'simulation_timestep': 1}

sim_1_hour = {'start_epoch': 0,
              'simulation_duration': 3600,
              'simulation_timestep': 1}

sim_6_hour = {'start_epoch': 0,
              'simulation_duration': 6 * 3600,
              'simulation_timestep': 1}

sim_12_hour = {'start_epoch': 0,
               'simulation_duration': 12 * 3600,
               'simulation_timestep': 1}

sim_24_hour = {'start_epoch': 0,
               'simulation_duration': 24 * 3600,
               'simulation_timestep': 1}

simulation_scenarios = {'sim_1_minute': sim_1_minute,
                        'sim_10_minute': sim_10_minute,
                        'sim_30_minute': sim_30_minute,
                        'sim_1_hour': sim_1_hour,
                        'sim_6_hour': sim_6_hour,
                        'sim_12_hour': sim_12_hour,
                        'sim_24_hour': sim_24_hour}


def print_simulation_scenarios() -> None:
    """
    Print the available scenarios regarding the simulation parameters.
    """
    print(f"The available simulation scenarios are: {list(simulation_scenarios.keys())}")


if __name__ == '__main__':
    print_simulation_scenarios()

