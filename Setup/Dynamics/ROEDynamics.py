from typing import Callable
import control as ct
import numpy as np
from matplotlib import pyplot as plt
from Dynamics.SystemDynamics import TranslationalDynamics
import Visualisation.Plotting as Plot


class QuasiROE(TranslationalDynamics):
    """
    A class for the quasi ROE model
    """
    def __init__(self, scenario: dict):
        super().__init__(scenario)
        self.is_LTI = False

    def create_model(self, sampling_time: float, argument_of_latitude: int = 0) -> ct.LinearIOSystem:
        """
        Create a discrete-time model with an A, B, C and D matrix.
        States: [(a_d-a_c)/a_c (-), u_d-u_c+(Omega_d-Omega_c) cos i_c (rad),
                 ex_d-ex_c (-), ey_d-ey_c (-),
                 ix_d-ix_c (rad), iy_d-iy_c (rad)]
        Inputs: [u_r (N), u_t (N), u_n (N)]

        :param sampling_time: Sampling time in s.
        :param argument_of_latitude: Argument of latitude in degrees.
        :return: A linear system object.
        """
        argument_of_latitude = np.deg2rad(argument_of_latitude)

        A_matrix = np.array([[0, 0, 0, 0, 0, 0],
                             [-1.5 * self.mean_motion, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]])

        B_matrix = np.array([[0, 2, 0],
                             [-2, 0, 0],
                             [np.sin(argument_of_latitude), 2 * np.cos(argument_of_latitude), 0],
                             [-np.cos(argument_of_latitude), 2 * np.sin(argument_of_latitude), 0],
                             [0, 0, np.cos(argument_of_latitude)],
                             [0, 0, np.sin(argument_of_latitude)]]) / self.satellite_mass / self.mean_motion / self.orbit_radius

        system_state_size, _ = B_matrix.shape
        system_continuous = ct.ss(A_matrix, B_matrix, np.eye(system_state_size), 0)

        # Find discrete system
        system_discrete = ct.sample_system(system_continuous, sampling_time)

        return system_discrete

    def create_initial_condition(self, relative_angle: float) -> np.ndarray[float]:
        """
        Create an array with the initial conditions of this system.

        :param relative_angle: The relative angle with respect to a reference in degrees.
        :return: An array with the initial conditions.
        """
        return np.array([0, relative_angle, 0, 0, 0, 0]).reshape((6, 1))

    def create_reference(self, relative_angle: float) -> np.ndarray[float]:
        """
        Create an array with the reference of this system.

        :param relative_angle: The relative angle with respect to a reference in degrees.
        :return: An array with the reference.
        """
        return np.array([0, relative_angle, 0, 0, 0, 0]).reshape((6, 1))

    def get_positional_angles(self) -> np.ndarray[bool]:
        """
        Find the position of the relative angle in the model.

        :return: Array of bools with True on the position of the relative angle.
        """
        return np.array([False, True, False, False, False, False])

    def get_plot_method(self) -> Callable[[np.ndarray, float, str, plt.figure, any], plt.figure]:
        """
        Return the method that can be used to plot this dynamical model.

        :return: Callable with arguments ->
                 states: np.ndarray, timestep: float, name: str = None, figure: plt.figure = None, kwargs
        """
        return Plot.plot_quasi_roe

    def get_state_constraint(self) -> list[int, float]:
        """
        Return the vector x_lim such that -x_lim <= x <= x_lim

        :return: List with maximum state values
        """
        return [0.001, 1000, 0.001, 0.001, 1000, 1000]

    def get_input_constraint(self) -> list[int, float]:
        """
        Return the vector u_lim such that -u_lim <= u <= u_lim

        :return: List with maximum input values
        """
        return [100, 100, 100]

    def get_state_cost_matrix_sqrt(self) -> np.ndarray:
        """
        Provide the matrix Q_sqrt

        :return: An nxn dimensional matrix representing Q_sqrt
        """
        return np.diag(np.array([1000, 10, 1000, 1000, 0, 0]))

    def get_input_cost_matrix_sqrt(self) -> np.ndarray:
        """
        Provide the matrix R_sqrt

        :return: An nxm dimensional matrix representing R_sqrt
        """
        return 1e-2 * 1 * np.array([[0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]])
