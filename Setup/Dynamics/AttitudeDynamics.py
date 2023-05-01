from typing import Callable, List
import Visualisation.Plotting as Plot
import control as ct
import numpy as np
from matplotlib import pyplot as plt
from Dynamics.SystemDynamics import AttitudeDynamics
from scipy.spatial.transform import Rotation
from Dynamics.DynamicsParameters import DynamicParameters
from Scenarios.MainScenarios import Scenario
from Scenarios.PhysicsScenarios import ScaledPhysics


class LinAttModel(AttitudeDynamics):
    """
    A class for a linear attitude model.
    """
    def __init__(self, scenario: Scenario):
        super().__init__(scenario)

        if isinstance(scenario.physics, ScaledPhysics):
            self.param = DynamicParameters(state_limit=[2, 4, 2, 0.1, 0.1, 0.1],
                                           input_limit=[0.1, 0.1, 0.1],
                                           q_sqrt=np.diag(np.array([1, 1, 1, 0.1, 0.1, 0.1])),
                                           r_sqrt_scalar=1)
        else:
            self.param = DynamicParameters(state_limit=[2, 4, 2, 0.1, 0.1, 0.1],
                                           input_limit=[0.1, 0.1, 0.1],
                                           q_sqrt=np.diag(np.array([1, 1, 1, 0.1, 0.1, 0.1])),
                                           r_sqrt_scalar=1)

    def create_model(self, sampling_time: float, **kwargs) -> ct.LinearIOSystem:
        """
        Create a discrete-time model with an A, B, C and D matrix.
        States: [roll, (rad) pitch (rad), yaw (rad), omega_x (rad/s), del_omega_y (rad/s), omega_z (rad/s)]
        Inputs: [T_x (Nm), T_y (Nm), T_z (Nm)]

        :param sampling_time: Sampling time in s.
        :param kwargs: Not used for this model.
        :return: A linear system object.
        """
        N_1 = np.array([[0, 0, self.satellite_moment_of_inertia[2, 2] - self.satellite_moment_of_inertia[1, 1]],
                        [0, 0, 0],
                        [self.satellite_moment_of_inertia[1, 1] - self.satellite_moment_of_inertia[0, 0], 0, 0]])
        N_3 = np.array([[self.satellite_moment_of_inertia[2, 2] - self.satellite_moment_of_inertia[1, 1], 0, 0],
                        [0, self.satellite_moment_of_inertia[2, 2] - self.satellite_moment_of_inertia[0, 0], 0],
                        [0, 0, 0]])

        A_00 = np.array([[0, 0, self.mean_motion],
                         [0, 0, 0],
                         [-self.mean_motion, 0, 0]])
        A_01 = np.eye(3)
        A_10 = 3 * self.mean_motion ** 2 * np.linalg.inv(self.satellite_moment_of_inertia) * N_3
        A_11 = self.mean_motion * np.linalg.inv(self.satellite_moment_of_inertia) * N_1

        A_matrix = np.block([[A_00, A_01],
                             [A_10, A_11]])

        B_matrix = np.block([[np.zeros((3, 3))],
                             [np.linalg.inv(self.satellite_moment_of_inertia)]])

        system_state_size, _ = B_matrix.shape
        system_continuous = ct.ss(A_matrix, B_matrix, np.eye(system_state_size), 0)

        # Find discrete system
        system_discrete = ct.sample_system(system_continuous, sampling_time)

        return system_discrete

    @staticmethod
    def quaternion_to_euler(quaternion: np.ndarray) -> np.ndarray:
        """
        Convert a quaternion to euler angles for this model.

        :param quaternion: Quaternion to convert in a (4,) shape.
        :return: Euler angles in a (3,) shape in the order roll-pitch-yaw in rad.
        """
        rot_quaternion = Rotation.from_quat(quaternion)
        return rot_quaternion.as_euler('ZXY')

    @staticmethod
    def euler_to_quaterion(euler: np.ndarray) -> np.ndarray:
        """
        Convert euler angles to a quaternion for this model.

        :param euler: Euler angles to convert in a (3,) shape in the order roll-pitch-yaw in rad.
        :return: Quaternion in a (4,) shape.
        """
        rot_euler = Rotation.from_euler('ZXY', euler)
        return rot_euler.as_quat()

    def create_initial_condition(self, quaternion: np.ndarray[float]) -> np.ndarray[float]:
        """
        Create an array with the initial conditions of this system.

        :param quaternion: The initial state as a quaternion.
        :return: An array with the initial conditions.
        """
        return np.concatenate((self.quaternion_to_euler(quaternion),
                               np.array([0, 0, 0]))).reshape((6, 1))

    def create_reference(self, quaternion: np.ndarray[float]) -> np.ndarray[float]:
        """
        Create an array with the reference of this system.

        :param quaternion: The reference state as a quaternion.
        :return: An array with the reference.
        """
        return np.concatenate((self.quaternion_to_euler(quaternion), np.array([0, 0, 0]))).reshape((6, 1))

    def get_plot_method(self) -> Callable[[np.ndarray, float, str, plt.figure, any], plt.figure]:
        """
        Return the method that can be used to plot this dynamical model.

        :return: Callable with arguments:
                 states: np.ndarray, timestep: float, name: str = None, figure: plt.figure = None, kwargs
        """
        return Plot.plot_cylindrical_states

    def get_state_constraint(self) -> list[int | float | int, float]:
        """
        Return the vector x_lim such that -x_lim <= x <= x_lim

        :return: List with maximum state values
        """
        return self.param.state_limit

    def get_input_constraint(self) -> list[int | float | int, float]:
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
