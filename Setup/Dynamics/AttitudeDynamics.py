from typing import Callable
import Visualisation.Plotting as Plot
import control as ct
import numpy as np
from matplotlib import pyplot as plt
from Dynamics.SystemDynamics import AttitudeDynamics
from scipy.spatial.transform import Rotation

class LinAttModel(AttitudeDynamics):
    """
    A class for a linear attitude model.
    """

    def create_model(self, sampling_time: float, **kwargs) -> ct.LinearIOSystem:
        """
        Create a discrete-time model with an A, B, C and D matrix.
        States: [roll, (rad) pitch (rad), yaw (rad), omega_x (rad/s), del_omega_y (rad/s), omega_z (rad/s)]
        Inputs: [T_x (Nm), T_y (Nm), T_z (Nm)]

        :param sampling_time: Sampling time in s.
        :param kwargs: Not used for this model.
        :return: A linear system object.
        """
        N_1 = np.array([[0, 0, self.satellite_moment_of_inertia[2, 2]-self.satellite_moment_of_inertia[1, 1]],
                       [0, 0, 0],
                       [self.satellite_moment_of_inertia[1, 1]-self.satellite_moment_of_inertia[0, 0], 0, 0]])
        N_3 = np.array([[self.satellite_moment_of_inertia[2, 2]-self.satellite_moment_of_inertia[1, 1], 0, 0],
                       [0, self.satellite_moment_of_inertia[2, 2]-self.satellite_moment_of_inertia[0, 0], 0],
                       [0, 0, 0]])

        A_00 = np.array([[0, 0, self.mean_motion],
                         [0, 0, 0],
                         [-self.mean_motion, 0, 0]])
        A_01 = np.eye(3)
        A_10 = 3 * self.mean_motion**2 * np.linalg.inv(self.satellite_moment_of_inertia) * N_3
        A_11 = self.mean_motion * np.linalg.inv(self.satellite_moment_of_inertia) * N_1

        A_matrix = np.block([[A_00, A_01],
                             [A_10, A_11]])

        B_matrix = np.block([[np.zeros(3)],
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
        return rot_quaternion.as_euler('YZX')

    @staticmethod
    def euler_to_quaterion(euler: np.ndarray) -> np.ndarray:
        """
        Convert euler angles to a quaternion for this model.

        :param euler: Euler angles to convert in a (3,) shape in the order roll-pitch-yaw in rad.
        :return: Quaternion in a (4,) shape.
        """
        rot_euler = Rotation.from_euler('YZX', euler)
        return rot_euler.as_quat()

    def create_initial_condition(self, quaternion: np.ndarray[float]) -> np.ndarray[float]:
        """
        Create an array with the initial conditions of this system.

        :param quaternion: The initial state as a quaternion.
        :return: An array with the initial conditions.
        """
        return np.concatenate((self.quaternion_to_euler(quaternion), np.array([0, 0, 0]))).reshape((6, 1))

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
