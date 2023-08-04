from typing import Callable
import control as ct
import numpy as np
from matplotlib import pyplot as plt
from tudatpy.kernel.astro import element_conversion

from Dynamics.DynamicsParameters import DynamicParameters
from Dynamics.SystemDynamics import TranslationalDynamics
import Visualisation.Plotting as Plot
from Scenarios.MainScenarios import Scenario


class Blend(TranslationalDynamics):
    """
    A class for the blend model
    """

    def __init__(self, scenario: Scenario):
        super().__init__(scenario)
        self.is_LTI = False

        self.e_c = scenario.orbital.eccentricity

        self.state_size = 6
        if not self.J2_active:
            self.param = DynamicParameters(state_limit=[0.002, 0.1, 1000, self.mean_motion / 10, 0.1, 0.5],
                                           input_limit=[0.1, 0.1, 0.1],
                                           q_sqrt=np.diag(np.array([220, 200, 50, 200, 150, 150])),
                                           r_sqrt_scalar=1e-2,
                                           slack_variable_length=10,
                                           slack_variable_costs=[1000000, 10, 10, 10, 10, 10],
                                           planetary_distance=np.deg2rad(5)
                                           )
        elif not isinstance(scenario.orbital.longitude, list):  # Single orbit
            self.param = DynamicParameters(state_limit=[0.002, 0.1, 1000, self.mean_motion / 10, np.deg2rad(0.1), np.deg2rad(0.01)],
                                           input_limit=[0.1, 0.1, 0.001],
                                           q_sqrt=np.diag(np.array([220, 10, 50, 10, 250, 250])),
                                           r_sqrt_scalar=1e-2,
                                           slack_variable_length=10,
                                           slack_variable_costs=[1000000, 0, 0, 0, 0, 0],
                                           planetary_distance=np.deg2rad(5))
        else:
            self.param = DynamicParameters(state_limit=[0.004, 0.1, 1000, self.mean_motion / 10, np.deg2rad(1), np.deg2rad(20)],
                                           input_limit=[0.1, 0.1, 0.1],
                                           q_sqrt=np.diag(np.array([220, 50, 50, 10, 250, 250])),
                                           r_sqrt_scalar=1e-2,
                                           slack_variable_length=10,
                                           slack_variable_costs=[1000000, 0, 0, 0, 0, 0],
                                           planetary_distance=np.deg2rad(5),
                                           inter_planetary_distance=np.deg2rad(5),
                                           radial_distance=-1)

    def create_model(self, sampling_time: float, argument_of_latitude: float = 0,
                     true_anomaly: float = 0) -> ct.LinearIOSystem:
        """
        Create a discrete-time model with an A, B, C and D matrix.
        States: [(r_d-r_c)/r_c (-), f_d + omega_d - f_c - omega_c (rad), i_d-i_c (rad), Omega_d-Omega_c (rad)]
        Inputs: [u_r (N), u_t (N), u_n (N)]

        :param sampling_time: Sampling time in s.
        :param argument_of_latitude: Argument of latitude in rad.
        :param true_anomaly: True anomaly in rad.
        :return: A linear system object.
        """
        # A_matrix = -1.5 * self.mean_motion * np.array([[0, 0, 0, 0],
        #                                                # [1, 0, np.cos(true_anomaly), 0, 0],
        #                                                [1, 0, 0, 0],
        #                                                # [0, 0, 0, 0],
        #                                                [0, 0, 0, 0],
        #                                                [0, 0, 0, 0]])
        A_matrix = np.array([[0, 1 / self.orbit_radius, 0, 0, 0, 0],
                             [3 * self.mean_motion**2 * self.orbit_radius, 0, 0, 2 * self.orbit_radius * self.mean_motion, 0, 0],
                             [0, 0, 0, 1, 0, 0],
                             [0, -2 * self.mean_motion / self.orbit_radius, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]])

        if self.J2_active:
            M_base = 3 * np.cos(self.inclination) ** 2 - 1
            omega_base = 5 * np.cos(self.inclination) ** 2 - 1
            Omega_base = -2 * np.cos(self.inclination)

            # M_dot = np.array([-7/2 * M_base, 0, -7/2 * M_base * np.cos(true_anomaly), -3 * eta * np.sin(self.inclination * 2), 0])
            # omega_dot = np.array([-7 / 2 * omega_base, 0, -7/2 * omega_base * np.cos(true_anomaly), -5 * np.sin(self.inclination * 2), 0])
            # Omega_dot = np.array([-7/2 * Omega_base, 0, -7/2 * Omega_base * np.cos(true_anomaly), 2 * np.sin(self.inclination), 0])

            M_dot = np.array([-7 / 2 * M_base * -3, 0, 0, 0, -3 * np.sin(self.inclination * 2), 0])
            omega_dot = np.array([-7 / 2 * omega_base * -3, 0, 0, 0, -5 * np.sin(self.inclination * 2), 0])
            Omega_dot = np.array([-7 / 2 * Omega_base * -3, 0, 0, 0, 2 * np.sin(self.inclination), 0])

            A_J2 = self.J2_scaling_factor * np.array([np.zeros(self.state_size),
                                                      np.zeros(self.state_size),
                                                      M_dot + omega_dot,
                                                      np.zeros(self.state_size),
                                                      np.zeros(self.state_size),
                                                      Omega_dot])

            A_matrix += A_J2

        # B_matrix = np.array([[0, 2, 0],
        #                      [-2, 0, -np.sin(argument_of_latitude) / np.tan(self.inclination)],
        #                      # [np.sin(true_anomaly), 2 * np.cos(true_anomaly), 0],
        #                      [0, 0, np.cos(argument_of_latitude)],
        #                      [0, 0, np.sin(argument_of_latitude) / np.sin(
        #                          self.inclination)]]) / self.satellite_mass / self.mean_motion / self.orbit_radius

        B_matrix = np.array([[0, 0, 0],
                             [self.mean_motion * self.orbit_radius, 0, 0],
                             [0, 0, -np.sin(argument_of_latitude) / np.tan(self.inclination)],
                             [0, self.mean_motion, 0],
                             [0, 0, np.cos(argument_of_latitude)],
                             [0, 0, np.sin(argument_of_latitude) / np.sin(
                                 self.inclination)]]) / self.satellite_mass / self.mean_motion / self.orbit_radius

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
        return np.array([0, 0, relative_angle, 0, 0, 0]).reshape((-1, 1))

    def create_reference(self, relative_angle: float) -> np.ndarray[float]:
        """
        Create an array with the reference of this system.

        :param relative_angle: The relative angle with respect to a reference in degrees.
        :return: An array with the reference.
        """
        return np.array([0, 0, relative_angle, 0, 0, 0]).reshape((-1, 1))

    def get_positional_angles(self) -> np.ndarray[bool]:
        """
        Find the position of the relative angle in the model.

        :return: Array of bools with True on the position of the relative angle.
        """
        return np.array([False, False, True, False, False, False])

    def get_plot_method(self) -> Callable[[np.ndarray, float, str, plt.figure, any], plt.figure]:
        """
        Return the method that can be used to plot this dynamical model.

        :return: Callable with arguments ->
                 states: np.ndarray, timestep: float, name: str = None, figure: plt.figure = None, kwargs
        """
        return Plot.plot_blend

    def get_state_constraint(self) -> list[int, float]:
        """
        Return the vector x_lim such that -x_lim <= x <= x_lim

        :return: List with maximum state values
        """
        return self.param.state_limit

    def get_input_constraint(self) -> list[int, float]:
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

    def get_latitude_and_periapsis(self, state: np.ndarray, time_since_start: int) -> tuple[list, list]:
        """
        Find the arguments of latitude and periapsis from the current state and the time since the start.

        :param state: Current state.
        :param time_since_start: Time since the start of the simulation in s.
        :return: Tuple with arguments of latitude and periapsis in rad.
        """
        argument_of_periapsis_dot = self.get_orbital_differentiation()[4]
        relative_latitude = state[2::6].flatten()
        absolute_latitude = (relative_latitude + (
                    self.mean_motion + argument_of_periapsis_dot) * time_since_start)  # % (2 * np.pi)
        return absolute_latitude, np.zeros_like(relative_latitude)

        #
        # argument_of_periapsis_dot = self.get_orbital_differentiation()[4]
        # RAAN = state[5::6] / np.tan(self.inclination)
        # relative_periapsis = state[3::6] - RAAN
        # absolute_periapsis = relative_periapsis + argument_of_periapsis_dot * time_since_start + self.periapsis
        #
        # relative_mean_anomaly = state[1::6] - np.sqrt(1 - self.eccentricity ** 2) * state[3::6]
        # absolute_mean_anomaly = relative_mean_anomaly + self.mean_motion * time_since_start
        #
        # true_anomaly = np.zeros_like(absolute_mean_anomaly)
        # for idx, mean_anomaly in enumerate(absolute_mean_anomaly):
        #     true_anomaly[idx] = element_conversion.mean_to_true_anomaly(self.eccentricity, mean_anomaly)
        #
        # absolute_latitude = true_anomaly + absolute_periapsis
        #
        # return absolute_latitude.flatten(), absolute_periapsis.flatten()

    def get_angles_list(self) -> list[bool]:
        """
        Find all the values that represent an angle.

        :return: Return a list with True for every state that represents an angle.
        """
        return [False, False, True, False, True, True]

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
        return [False, False, False, False, False, True]

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

    def get_radial_distance(self) -> int | float:
        """
        Find the minimum radial distance between satellites when crossing.

        :return: The minimum rddial distance
        """
        return self.param.radial_distance
