from scipy.spatial.transform import Rotation

no_state_error = {'attitude_offset': Rotation.from_euler('X', 0, degrees=True),
                  'angular_velocity_offset_magnitude': 0,
                  'dropouts': 0,
                  'initial_velocity_error': 0}

small_state_error = {'attitude_offset': Rotation.from_euler('XYZ', [6, -8, 4], degrees=True),
                     'angular_velocity_offset_magnitude': 0.01,
                     'dropouts': 1,
                     'initial_velocity_error': 0}

large_state_error = {'attitude_offset': Rotation.from_euler('XYZ', [30, -41, 22], degrees=True),
                     'angular_velocity_offset_magnitude': 0.05,
                     'dropouts': 2,
                     'initial_velocity_error': 0}

initial_state_scenarios = {'no_state_error': no_state_error,
                           'small_state_error': small_state_error,
                           'large_state_error': large_state_error}


def print_initial_state_scenarios() -> None:
    """
    Print the available scenarios regarding the initial states.
    """
    print(f"The available initial state scenarios are: {list(initial_state_scenarios.keys())}")


if __name__ == '__main__':
    print_initial_state_scenarios()
