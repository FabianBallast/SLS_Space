from typing import Callable
import control as ct
import numpy as np
from matplotlib import pyplot as plt
from Dynamics.SystemDynamics import TranslationalDynamics
from Scenarios.MainScenarios import Scenario
import Visualisation.Plotting as Plot


class DifferentialDragDynamics(TranslationalDynamics):
    """
    Class that represents the translational dynamics from Planet Labs.
    """

    def __init__(self, scenario: Scenario):
        super().__init__(scenario)
        self.param = None
        self.state_size = 2
        self.input_size = 1
        self.atmosphere_density = scenario.physics.atmosphere_density

    def __get_max_angular_velocity(self) -> float:
        """
        Find the maximum angular velocity that can be obtained with differential drag.
        :return:
        """
        ballistic_coefficient_high = 11
        ballistic_coefficient_low = 45
        atmosphere_density = self.atmosphere_density  # Roughly at 750 km
        radius = self.orbit_radius
        mu = self.earth_gravitational_parameter
        v = np.sqrt(mu / radius ** 3) * radius
        pressure = 0.5 * atmosphere_density * v ** 2
        return 3 * pressure / radius * (1 / ballistic_coefficient_high - 1 / ballistic_coefficient_low)

    def create_model(self, sampling_time: float, **kwargs) -> ct.LinearIOSystem:
        """
        Create a discrete-time model with an A, B, C and D matrix.
        States: [theta (rad), theta_dot (rad/s)]
        Inputs: [u_theta (N)]

        :param sampling_time: Sampling time of the controller in s.
        :param kwargs: Not used here.
        :return: A linear system object.
        """
        A_matrix = np.array([[1, sampling_time],
                             [0, 1]])

        B_matrix = np.array([[0],
                             [self.__get_max_angular_velocity() * sampling_time]])

        return ct.ss(A_matrix, B_matrix, np.eye(2), 0, sampling_time)

    def create_initial_condition(self, relative_angle: float) -> np.ndarray[float]:
        return np.array([relative_angle, 0]).reshape((2, 1))

    def create_reference(self, relative_angle: float) -> np.ndarray[float]:
        return np.array([relative_angle, 0]).reshape((2, 1))

    def get_positional_angles(self) -> np.ndarray[bool]:
        return np.array([True, False])

    def get_plot_method(self) -> Callable[..., plt.figure]:
        return Plot.plot_differential_drag_states

    def get_state_constraint(self) -> list[int, float]:
        pass

    def get_input_constraint(self) -> list[int, float]:
        pass

    def get_state_cost_matrix_sqrt(self) -> np.ndarray:
        pass

    def get_input_cost_matrix_sqrt(self) -> np.ndarray:
        pass

    def get_angles_list(self) -> list[bool]:
        """
        Find all the values that represent an angle.

        :return: Return a list with True for every state that represents an angle.
        """
        return [True, False]
