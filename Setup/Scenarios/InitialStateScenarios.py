from enum import Enum
from scipy.spatial.transform import Rotation


class InitialStateError:
    def __init__(self, attitude_offset: Rotation = Rotation.from_euler('x', 0),
                 angular_velocity_offset_magnitude: float = 0.0, dropouts: float = 0.0,
                 initial_velocity_error: float = 0):
        """
        :param attitude_offset: Rotation error for attitude control.
        :param angular_velocity_offset_magnitude: Magnitude of angular velocity error. Direction is randomised.
        :param dropouts: Percentage of dropouts in satellite swarm,
        :param initial_velocity_error: Magnitude of initial velocity error.
        """
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
                                          angular_velocity_offset_magnitude=0.01, dropouts=0.1)
    large_state_error = InitialStateError(attitude_offset=Rotation.from_euler('XYZ', [90, -71, 112], degrees=True),
                                          angular_velocity_offset_magnitude=0.05, dropouts=0.2)


if __name__ == '__main__':
    print(list(InitialStateScenarios))
