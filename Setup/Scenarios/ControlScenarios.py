from enum import Enum


class ControlParameters:
    """
    Class to represent different control related parameters.
    """

    def __init__(self, control_timestep=5, tFIR=10):
        self.control_timestep = control_timestep
        self.tFIR = tFIR

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Control parameters with {self.control_timestep=} and {self.tFIR=}"


class ControlParameterScenarios(Enum):
    """
    Simple enum for different sets of control parameters.
    """
    control_attitude_default = ControlParameters()
    control_attitude_far_ahead = ControlParameters(control_timestep=15, tFIR=20)
    control_position_default = ControlParameters(control_timestep=30)
    control_position_fine = ControlParameters(control_timestep=10)
    control_position_far_ahead = ControlParameters(control_timestep=60, tFIR=20)


class Model(Enum):
    """
    Very simple enum to select a model for the model-based controller.
    """
    ATTITUDE = 1
    HCW = 2
    ROE = 3


if __name__ == '__main__':
    print(list(ControlParameterScenarios))
