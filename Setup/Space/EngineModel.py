import numpy as np
from tudatpy.kernel.astro import frame_conversion
from tudatpy.kernel.numerical_simulation import environment


class EngineModel:
    """
    Base class for Engine models. Can be used for both thrust and torque models.
    """
    def __init__(self, control_input: np.ndarray, control_timestep: float):
        """
        Initialise the engine model with basic variables.

        :param control_input: The control inputs of the given engine in shape (3, t).
        :param control_timestep: The control timestep, i.e. the time between each input.
        """
        self.control_input = control_input
        self.control_timestep = control_timestep  # Length between different control signals [s]
        self.t0 = None
        self.number_of_inputs = len(self.control_input[0, :])


class ThrustModel(EngineModel):
    """
    Class to model a thrust engine.
    """
    def __init__(self, thrust_input: np.ndarray, control_timestep: float,
                 Isp: float, propagated_body: environment.Body):
        """
        Initialise the thrust model.

        :param thrust_input: The thrust inputs in the shape (3, t).
        :param control_timestep: The timestep between each control input in s.
        :param Isp: Specific impulse in s.
        :param propagated_body: The body that is being propagated. Required for direction analysis.
        """
        # Control input in xyz direction over time (numpy.ndarray[3, number_of_inputs]) in RSW frame [N]
        super().__init__(thrust_input, control_timestep)
        self.Isp = Isp  # Specific impulse [s]
        self.propagated_body = propagated_body  # Body that is being propagated

        self.control_magnitudes = np.zeros(self.number_of_inputs)
        self.control_direction = np.zeros_like(self.control_input)
        self.find_magnitude_and_direction()

    def find_magnitude_and_direction(self) -> None:
        """
        Internally update the magnitude and direction of the control inputs.  
        """
        # Use norm for the magnitudes
        self.control_magnitudes = np.linalg.norm(self.control_input, axis=0)

        # Direction is easy for values where magnitude is not 0
        non_zero_magnitudes = self.control_magnitudes > 0.00000001
        self.control_direction[:, non_zero_magnitudes] = self.control_input[:, non_zero_magnitudes] / \
                                                         self.control_magnitudes[non_zero_magnitudes]
        self.control_direction[:, ~non_zero_magnitudes] = np.array(
            [[1, 0, 0]]).T  # Use x-axis for direction with zero magnitude

    def get_thrust_magnitude(self, time: float) -> float:
        """
        Find the total magnitude of thrust.

        :param time: The time at which the magnitude should be evaluated.
        :return: The magnitude of thrust as a float.
        """
        if np.isnan(time):
            return 0  # Check if this is allowed! -> Seems to be. Changing to 100 does not affect result
        else:
            if self.t0 is None:
                self.t0 = time

            idx = int((time - self.t0) / self.control_timestep)
            if idx < self.number_of_inputs:
                return self.control_magnitudes[idx]
            elif idx == self.number_of_inputs:
                return self.control_magnitudes[-1]
            else:
                raise Exception(f"No control input available for time {time} s. Simulation started at {self.t0} s.")

    def get_specific_impulse(self, time: float) -> float:
        """
        Return the specific impulse at a specific time. Constant for now, so always the same.
        Argument required for tudatpy.

        :param time: Time when to evaluate the specific impulse.
        :return: Specific impulse as a float
        """
        # Return the constant specific impulse
        return self.Isp

    def get_thrust_direction(self, time: float) -> np.ndarray[np.float64]:
        """
        Find the direction of thrust in the inertial frame.

        :param time: The time at which the direction should be evaluated.
        :return: The direction of thrust as a numpy array of shape (3,)
        """
        if np.isnan(time):
            return np.array([1, 0, 0])
        else:
            if self.t0 is None:
                self.t0 = time

            idx = int((time - self.t0) / self.control_timestep)
            if idx < self.number_of_inputs:
                # Find force in RSW frame
                thrust_direction_rsw_frame = self.control_direction[:, idx]

                # Find rotation matrix from RSW to inertial frame
                current_state = self.propagated_body.state
                rsw_to_inertial_frame = frame_conversion.rsw_to_inertial_rotation_matrix(current_state)

                # Compute the thrust in the inertial frame
                thrust_inertial_frame = np.dot(rsw_to_inertial_frame, thrust_direction_rsw_frame)

                # Return the thrust direction in the inertial frame
                return thrust_inertial_frame
            elif idx == self.number_of_inputs:  # In this case, simply use last value in list.
                # Find force in RSW frame
                thrust_direction_rsw_frame = self.control_direction[:, -1]

                # Find rotation matrix from RSW to inertial frame
                current_state = self.propagated_body.state
                rsw_to_inertial_frame = frame_conversion.rsw_to_inertial_rotation_matrix(current_state)

                # Compute the thrust in the inertial frame
                thrust_inertial_frame = np.dot(rsw_to_inertial_frame, thrust_direction_rsw_frame)

                # Return the thrust direction in the inertial frame
                return thrust_inertial_frame
            else:
                raise Exception(f"No control input available for time {time} s. Simulation started at {self.t0} s.")


class TorqueModel(EngineModel):
    """
    Model to provide custom torques.
    """
    def __init__(self, torque_input: np.ndarray, control_timestep: float):
        """
        Initialise a torque model.

        :param torque_input: Numpy array of shape (3, t) with the torque inputs over time.
        :param control_timestep: Timestep used for control purposes.
        """
        super().__init__(torque_input, control_timestep)

    def get_torque(self, time: float) -> np.ndarray:
        """
        Provide the torque at a required time.

        :param time: Time from which the torque should be provided.
        :return: Numpy array with torque.
        """
        if np.isnan(time):
            return np.array([1, 0, 0])
        else:
            if self.t0 is None:
                self.t0 = time

            idx = int((time - self.t0) / self.control_timestep)
            if idx < self.number_of_inputs:
                return self.control_input[:, idx]
            else:
                raise Exception(f"No control input available for time {time} s. Simulation started at {self.t0} s.")
