from __future__ import annotations
from typing import Callable
import Visualisation.Plotting as Plot
import control as ct
import numpy as np
from matplotlib import pyplot as plt
from Dynamics.DynamicsParameters import DynamicParameters
from Dynamics.SystemDynamics import TranslationalDynamics
from Scenarios.MainScenarios import Scenario


class RelCylHCW(TranslationalDynamics):
    """
    A class for the relative cylindrical model.
    """

    def __init__(self, scenario: Scenario):
        super().__init__(scenario)

        if self.is_scaled and not self.J2_active and not isinstance(scenario.orbital.longitude, list):
            self.param = DynamicParameters(state_limit=[0.1, 10, 0.1, 0.1, self.mean_motion / 10, 0.1],
                                           input_limit=[0.1, 0.1, 0.1],
                                           q_sqrt=np.diag(np.array([5, 50, 0.01, 50, 100, 0.1])),  # 4, 50
                                           r_sqrt_scalar=1e-2,
                                           slack_variable_length=0,
                                           slack_variable_costs=[10000, 0, 0, 0, 0, 0],
                                           planetary_distance=np.deg2rad(5))
        elif self.is_scaled and not isinstance(scenario.orbital.longitude, list):  # Single orbit
            self.param = DynamicParameters(state_limit=[0.1, 10, 0.1, 0.1, self.mean_motion / 10, 0.1],
                                           input_limit=[0.1, 0.1, 0.1],
                                           q_sqrt=np.diag(np.array([5, 50, 0.01, 50, 100, 0.1])),
                                           r_sqrt_scalar=1e-2,
                                           slack_variable_length=0,
                                           slack_variable_costs=[10000, 0, 0, 0, 0, 0],
                                           planetary_distance=np.deg2rad(5))
        elif self.is_scaled:
            self.param = DynamicParameters(state_limit=[0.1, 10, 4, 0.1, self.mean_motion / 10, 0.1],
                                           input_limit=[0.1, 0.1, 0.1],
                                           q_sqrt=np.diag(np.array([5, 50, 0.01, 50, 100, 0.1])),
                                           r_sqrt_scalar=1e-2,
                                           slack_variable_length=0,
                                           slack_variable_costs=[10000, 0, 0, 0, 0, 0],
                                           planetary_distance=np.deg2rad(5),
                                           inter_planetary_distance=1)
        else:
            self.param = DynamicParameters(state_limit=[10000, 10000, 100, 10, self.mean_motion / 10, 1],
                                           input_limit=[100, 100, 100],
                                           q_sqrt=np.diag(np.array([4, 50, 4, 0, 0, 0])),
                                           r_sqrt_scalar=1e-2)

    def create_model(self, sampling_time: float, **kwargs) -> ct.LinearIOSystem:
        """
        Create a discrete-time model with an A, B, C and D matrix.
        States: [rho (m), theta (rad), z (m), rho_dot (m/s), theta_dot (rad/s), z_dot (m/s)]
        Inputs: [u_rho (N), u_theta (N), u_z (N)]

        :param sampling_time: Sampling time in s.
        :param kwargs: Not used here. Used for generality with parent class.
        :return: A linear system object.
        """
        A_matrix = np.array([[0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1],
                             [3 * self.mean_motion ** 2, 0, 0, 0, 2 * self.orbit_radius * self.mean_motion, 0],
                             [0, 0, 0, -2 * self.mean_motion / self.orbit_radius, 0, 0],
                             [0, 0, -self.mean_motion ** 2, 0, 0, 0]])

        B_matrix = np.array([[0, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0],
                             [1, 0, 0],
                             [0, 1 / self.orbit_radius, 0],
                             [0, 0, 1]]) / self.satellite_mass

        system_state_size, _ = B_matrix.shape
        system_continuous = ct.ss(A_matrix, B_matrix, np.eye(system_state_size), 0)

        # Find discrete system
        system_discrete = ct.sample_system(system_continuous, sampling_time)

        return system_discrete

    def create_initial_condition(self, relative_angle: float) -> np.ndarray[float]:
        """
        Create an array with the initial conditions of this system.

        :param relative_angle: The relative angle with respect to a reference in radians.
        :return: An array with the initial conditions.
        """
        return np.array([0, relative_angle, 0, 0, 0, 0]).reshape((6, 1))

    def create_reference(self, relative_angle: float) -> np.ndarray[float]:
        """
        Create an array with the reference of this system.

        :param relative_angle: The relative angle with respect to a reference in radians.
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

        :return: Callable with arguments:
                 states: np.ndarray, timestep: float, name: str = None, figure: plt.figure = None, kwargs
        """
        return Plot.plot_cylindrical_states

    def get_state_constraint(self) -> list[int, float]:
        """
        Return the vector x_lim such that -x_lim <= x <= x_lim

        :return: List with maximum state values
        """
        return self.param.state_limit

    def get_input_constraint(self) -> list[float | int | int, float]:
        """
        Return the vector u_lim such that -u_lim <= u <= u_lim

        :return: List with maximum input values
        """
        return self.param.input_limit

    def get_state_cost_matrix_sqrt(self) -> np.ndarray:
        """
        Provide the matrix Q_sqrt

        :return: An nxn dimensional matrix representing Q_sqrt
        """
        return self.param.Q_sqrt

    def get_input_cost_matrix_sqrt(self) -> np.ndarray:
        """
        Provide the matrix R_sqrt

        :return: An nxm dimensional matrix representing R_sqrt
        """
        return self.param.R_sqrt

    def get_angles_list(self) -> list[bool]:
        """
        Find all the values that represent an angle.

        :return: Return a list with True for every state that represents an angle.
        """
        return [False, True, False, False, False, False]

    def get_slack_variable_length(self) -> int:
        """
        Get the time horizon for which to use slack variables.

        :return: Time for which slack variables are used.
        """
        return self.param.slack_variable_length

    def get_slack_costs(self) -> list[int]:
        """
        Find the states for which to use slack variables and their costs

        :return: List with positive cost for states where slack variables should be applied.
        """
        return self.param.slack_variable_costs

    def get_orbital_parameter(self) -> list[bool]:
        """
        Find the parameter that determines if two orbits are close.

        :return: List with bool with True for the orbital parameter.
        """
        return [False, False, True, False, False, False]

    def get_planetary_distance(self) -> int | float:
        """
        Find the minimum planetary distance.

        :return: The minimum planetary distance
        """
        return self.param.planetary_distance

    def get_inter_planetary_distance(self) -> int | float:
        """
        Find the minimum inter_planetary distance.

        :return: The minimum inter_planetary distance
        """
        return self.param.inter_planetary_distance
