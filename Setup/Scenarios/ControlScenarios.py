from enum import Enum

control_attitude_default = {'control_timestep': 5,
                            'tFIR': 10}

control_attitude_far_ahead = {'control_timestep': 15,
                              'tFIR': 20}

control_position_default = {'control_timestep': 30,
                            'tFIR': 10}

control_position_far_ahead = {'control_timestep': 60,
                              'tFIR': 20}

control_scenarios = {'control_attitude_default': control_attitude_default,
                     'control_attitude_far_ahead': control_attitude_far_ahead,
                     'control_position_default': control_position_default,
                     'control_position_far_ahead': control_position_far_ahead}


def print_control_scenarios() -> None:
    """
    Print the available scenarios regarding the control parameters.
    """
    print(f"The available control scenarios are: {list(control_scenarios.keys())}")


class Model(Enum):
    """
    Very simple enum to select a model for the model-based controller.
    """
    ATTITUDE = 1
    HCW = 2
    ROE = 3


if __name__ == '__main__':
    print_control_scenarios()