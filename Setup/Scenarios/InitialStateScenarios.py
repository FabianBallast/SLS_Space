from enum import Enum

from scipy.spatial.transform import Rotation


class InitialStateError:
    def __init__(self, attitude_offset=Rotation.from_euler('x', 0), angular_velocity_offset_magnitude=0.0, dropouts=0,
                 initial_velocity_error=0):
        self.attitude_offset = attitude_offset
        self.angular_velocity_offset_magnitude = angular_velocity_offset_magnitude
        self.dropouts = dropouts
        self.initial_velocity_error = initial_velocity_error

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Initial state error with {self.attitude_offset=}, {self.angular_velocity_offset_magnitude=}, " \
               f"{self.dropouts=} and {self.initial_velocity_error=}"


class InitialStateScenarios(Enum):
    """
    Simple enum for initial state errors
    """
    no_state_error = InitialStateError()
    small_state_error = InitialStateError(attitude_offset=Rotation.from_euler('XYZ', [6, -8, 4], degrees=True),
                                          angular_velocity_offset_magnitude=0.01, dropouts=1)
    large_state_error = InitialStateError(attitude_offset=Rotation.from_euler('XYZ', [30, -41, 22], degrees=True),
                                          angular_velocity_offset_magnitude=0.05, dropouts=2)


if __name__ == '__main__':
    print(list(InitialStateScenarios))
