from __future__ import annotations
from typing import Callable
import control as ct
import numpy as np
from matplotlib import pyplot as plt
from Dynamics.DynamicsParameters import DynamicParameters
from Dynamics.SystemDynamics import TranslationalDynamics
import Visualisation.Plotting as Plot
from Scenarios.MainScenarios import Scenario


class ROE(TranslationalDynamics):
    """
    A class for the ROE model
    """

    def __init__(self, scenario: Scenario):
        super().__init__(scenario)
        self.is_LTI = False
        self.e_c = scenario.orbital.eccentricity

        if self.is_scaled and not self.J2_active:
            self.param = DynamicParameters(state_limit=[0.01, 10, 0.1, 0.1, 0.1, 0.1],
                                           # [0.0015, 10, 0.01, 0.01, 0.001, 0.001],
                                           input_limit=[0.1, 0.1, 0.1],
                                           q_sqrt=np.diag(np.array([220, 50, 700, 50, 1, 1])),
                                           r_sqrt_scalar=1e-2)
        elif self.is_scaled:
            self.param = DynamicParameters(state_limit=[0.0015, 10, 0.003, 0.01, 0.01, 0.01],
                                           input_limit=[0.1, 0.1, 0.1],
                                           q_sqrt=np.diag(np.array([220, 50, 800, 50, 100, 100])),
                                           r_sqrt_scalar=1e-2,
                                           slack_variable_length=9,
                                           slack_variable_costs=[10000, 0, 0, 0, 0, 0])
        else:
            self.param = DynamicParameters(state_limit=[0.001, 10, 0.001, 0.001, 1000, 1000],
                                           input_limit=[0.1, 0.1, 0.1],
                                           q_sqrt=np.diag(np.array([1000, 10, 1000, 1000, 0, 0])),
                                           r_sqrt_scalar=1e-2)

    def create_model(self, sampling_time: float, argument_of_latitude: float = 0,
                     true_anomaly: float = 0) -> ct.LinearIOSystem:
        """
        Create a discrete-time model with an A, B, C and D matrix.
        States: [(a_d-a_c)/a_c (-), M_d-M_c+eta(omega_d - omega_c + (Omega_d-Omega_c) cos i_c) (rad),
                 e_d-e_c (-), omega_d - omega_c + (Omega_d-Omega_c) cos i_c (rad),
                 i_d-i_c (rad), (Omega_d-Omega_c) sin i_c (rad)]
        Inputs: [u_r (N), u_t (N), u_n (N)]

        :param sampling_time: Sampling time in s.
        :param argument_of_latitude: Argument of latitude in rad.
        :param true_anomaly: True anomaly in rad.
        :return: A linear system object.
        """
        # argument_of_latitude = np.deg2rad(argument_of_latitude)
        # true_anomaly = np.deg2rad(true_anomaly)

        eta = np.sqrt(1 - self.e_c ** 2)
        kappa = 1 + self.e_c * np.cos(true_anomaly)

        A_matrix = np.array([[0, 0, 0, 0, 0, 0],
                             [-1.5 * self.mean_motion, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]])

        if self.J2_active:
            P = 3 * np.cos(self.inclination) ** 2 - 1
            Q = 5 * np.cos(self.inclination) ** 2 - 1
            R = np.cos(self.inclination)
            S = np.sin(2 * self.inclination)
            U = np.sin(self.inclination)
            G = 1 / eta ** 2
            M_dot = np.array([-7 / 2 * eta * P, 0, 3 * self.eccentricity * eta * G * P, 0, -3 * eta * S, 0])
            omega_dot = np.array([-7 / 2 * Q, 0, 4 * self.eccentricity * G * Q, 0, -5 * S, 0])
            Omega_dot = np.array([7 * self.inclination, 0, -8 * self.eccentricity * G * R, 0, 2 * U, 0])

            A_J2 = self.J2_scaling_factor * np.array([np.zeros(6),
                                                      M_dot + eta * (omega_dot + Omega_dot * np.cos(self.inclination)),
                                                      np.zeros(6),
                                                      omega_dot + Omega_dot * np.cos(self.inclination),
                                                      np.zeros(6),
                                                      Omega_dot * np.sin(self.inclination)])

            A_matrix += A_J2

        B_matrix = np.array([[2 / eta * self.e_c * np.sin(true_anomaly), 2 / eta * kappa, 0],
                             [-2 * eta ** 2 / kappa, 0, 0],
                             [eta * np.sin(true_anomaly), eta * (self.e_c + np.cos(true_anomaly) * (1 + kappa)) / kappa,
                              0],
                             [-eta / self.e_c * np.cos(true_anomaly),
                              eta / self.e_c * np.sin(true_anomaly) * (1 + kappa) / kappa, 0],
                             [0, 0, eta * np.cos(argument_of_latitude) / kappa],
                             [0, 0, eta * np.sin(
                                 argument_of_latitude) / kappa]]) / self.satellite_mass / self.mean_motion / self.orbit_radius

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
        return Plot.plot_roe

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
        RAAN = state[5::6] / np.tan(self.inclination)
        relative_periapsis = state[3::6] - RAAN
        absolute_periapsis = relative_periapsis + argument_of_periapsis_dot * time_since_start + self.periapsis

        relative_mean_anomaly = state[1::6] - np.sqrt(1 - self.eccentricity ** 2) * state[3::6]
        absolute_mean_anomaly = relative_mean_anomaly + self.mean_motion * time_since_start

        true_anomaly = np.zeros_like(absolute_mean_anomaly)
        for idx, mean_anomaly in enumerate(absolute_mean_anomaly):
            true_anomaly[idx] = element_conversion.mean_to_true_anomaly(self.eccentricity, mean_anomaly)

        absolute_latitude = true_anomaly + absolute_periapsis

        return absolute_latitude.flatten(), absolute_periapsis.flatten()

    def get_angles_list(self) -> list[bool]:
        """
        Find all the values that represent an angle.

        :return: Return a list with True for every state that represents an angle.
        """
        return [False, True, False, True, True, True]

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


class QuasiROE(TranslationalDynamics):
    """
    A class for the quasi ROE model
    """

    def __init__(self, scenario: Scenario):
        super().__init__(scenario)
        self.is_LTI = False

        if self.is_scaled and not self.J2_active:
            self.param = DynamicParameters(state_limit=[0.1 / self.orbit_radius, 100, 0.002, 0.002, 0.05, 1],
                                           input_limit=[0.1, 0.1, 0.1],
                                           q_sqrt=np.diag(np.array([5 * self.orbit_radius, 50, 200, 200, 15, 15])),
                                           r_sqrt_scalar=1e-2,
                                           slack_variable_length=0,
                                           slack_variable_costs=[10000, 0, 0, 0, 0, 0])
        elif self.is_scaled:
            self.param = DynamicParameters(state_limit=[0.1 / self.orbit_radius, 100, 0.002, 0.002, 0.05, 1],
                                           input_limit=[0.1, 0.1, 0.1],
                                           q_sqrt=np.diag(np.array([5 * self.orbit_radius, 50, 200, 200, 15, 15])),
                                           r_sqrt_scalar=1e-2,
                                           slack_variable_length=0,
                                           slack_variable_costs=[10000, 0, 0, 0, 0, 0])
        else:
            self.param = DynamicParameters(state_limit=[0.1 / self.orbit_radius, 100, 0.002, 0.002, 0.05, 1],
                                           input_limit=[0.1, 0.1, 0.1],
                                           q_sqrt=np.diag(np.array([5 * self.orbit_radius, 50, 200, 200, 15, 15])),
                                           r_sqrt_scalar=1e-2)

    def create_model(self, sampling_time: float, argument_of_latitude: int = 0, **kwargs) -> ct.LinearIOSystem:
        """
        Create a discrete-time model with an A, B, C and D matrix.
        States: [(a_d-a_c)/a_c (-), u_d-u_c+(Omega_d-Omega_c) cos i_c (rad),
                 ex_d-ex_c (-), ey_d-ey_c (-),
                 ix_d-ix_c (rad), iy_d-iy_c (rad)]
        Inputs: [u_r (N), u_t (N), u_n (N)]

        :param sampling_time: Sampling time in s.
        :param argument_of_latitude: Argument of latitude in rad.
        :return: A linear system object.
        """
        # argument_of_latitude = np.deg2rad(argument_of_latitude)

        A_matrix = np.array([[0, 0, 0, 0, 0, 0],
                             [-1.5 * self.mean_motion, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]])

        if self.J2_active:
            eta = np.sqrt(1 - self.eccentricity ** 2)
            A_10 = -21 * (eta * (3 * np.cos(self.inclination)**2 - 1) + 5 * np.cos(self.inclination)**2 - 1)

            gamma = 1 / 8 * self.J2_scaling_factor * self.orbit_radius ** 2 * self.mean_motion / self.orbit_radius ** 2 / eta**4
            A_J2 = gamma * np.array([np.zeros(6),
                                     [A_10, 0, 0, 0, -6 * np.sin(2 * self.inclination) * (3 * eta + 5), 0],
                                     [0, 0, 0, -6 * (5 * np.cos(self.inclination)**2 - 1), 0, 0],
                                     [0, 0,6 * (5 * np.cos(self.inclination)**2 - 1), 0, 0, 0],
                                     np.zeros(6),
                                     [21 * np.sin(2 * self.inclination), 0, 0, 0, 12 * np.sin(self.inclination)**2, 0]])

            A_matrix += A_J2

        B_matrix = np.array([[0, 2, 0],
                             [-2, 0, 0],
                             [np.sin(argument_of_latitude), 2 * np.cos(argument_of_latitude), 0],
                             [-np.cos(argument_of_latitude), 2 * np.sin(argument_of_latitude), 0],
                             [0, 0, np.cos(argument_of_latitude)],
                             [0, 0, np.sin(
                                 argument_of_latitude)]]) / self.satellite_mass / self.mean_motion / self.orbit_radius

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
        anomaly_dot = self.get_orbital_differentiation()[5]
        RAAN = state[5::6] / np.sin(self.inclination)
        relative_latitude = (state[1::6] - RAAN * np.cos(self.inclination)).flatten()
        absolute_latitude = (relative_latitude + (
                    self.mean_motion + argument_of_periapsis_dot + anomaly_dot) * time_since_start) % (
                                    2 * np.pi)
        return absolute_latitude, np.zeros_like(relative_latitude)

    def get_angles_list(self) -> list[bool]:
        """
        Find all the values that represent an angle.

        :return: Return a list with True for every state that represents an angle.
        """
        return [False, True, False, False, True, True]

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
        pass

    def get_planetary_distance(self) -> int | float:
        pass

    def get_inter_planetary_distance(self) -> int | float:
        pass
