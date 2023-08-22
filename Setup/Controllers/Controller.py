from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt
import random
from Dynamics import SystemDynamics as SysDyn
from Dynamics.DifferentialDragDynamics import DifferentialDragDynamics


class Controller(ABC):
    """
    Base class for different controllers.
    """

    def __init__(self, sampling_time: float | int, system_dynamics: SysDyn.GeneralDynamics, prediction_horizon: int) -> None:
        """
        Initialise the controller.

        :param sampling_time: Sampling time of the controller in s.
        :param prediction_horizon: Prediction horizon of the controller in steps.
        """
        self.sys = None
        self.synthesizer = None
        self.controller = None
        self.sampling_time = sampling_time
        self.prediction_horizon = prediction_horizon
        self.number_of_systems = None
        self.total_state_size = None
        self.total_input_size = None
        self.system_state_size = None
        self.x0 = None
        self.x_ref = None
        self.x_states = None
        self.u_inputs = None
        self.angle_states = None
        self.dynamics = system_dynamics
        self.all_angle_states = None

    @abstractmethod
    def create_system(self, number_of_systems: int) -> None:
        """
        Create a system of several homogeneous satellites. For LTV systems, these matrices are updated when setting the
        initial conditions using 'set_initial_condition'.

        :param number_of_systems: Number of satellites.
        """
        pass

    def create_x0(self, number_of_dropouts: int = None, seed: int = 129) -> None:
        """
        Create an evenly spaced initial state given a number of dropouts.

        :param number_of_dropouts: Number of satellites that have dropped out of their orbit.
        :param seed: Random seed.
        """
        self.x0 = np.zeros((self.total_state_size, 1))

        if isinstance(self.dynamics, SysDyn.TranslationalDynamics):

            if number_of_dropouts is None:
                raise Exception("Enter number of dropouts!")
            number_of_original_systems = self.number_of_systems + number_of_dropouts

            if isinstance(self.dynamics, DifferentialDragDynamics):
                number_of_original_systems += 1

            random.seed(129)
            start_angles = np.linspace(0, 2 * np.pi, number_of_original_systems, endpoint=False) \
                .reshape((number_of_original_systems, 1))
            selected_indices = random.sample(range(number_of_original_systems),
                                             number_of_original_systems - number_of_dropouts)

            if isinstance(self.dynamics, DifferentialDragDynamics):
                if min(selected_indices) != 0:
                    selected_indices[-1] = 0
                angles_sorted = start_angles[np.sort(selected_indices)][1:]
            else:
                angles_sorted = start_angles[np.sort(selected_indices)]

            for i in range(self.number_of_systems):
                self.x0[i * self.system_state_size:
                        (i + 1) * self.system_state_size] = self.dynamics.create_initial_condition(angles_sorted[i, 0])

            # print(self.x0[self.angle_states])
        elif isinstance(self.dynamics, SysDyn.AttitudeDynamics):
            for i in range(self.number_of_systems):
                self.x0[i * self.system_state_size:
                        (i + 1) * self.system_state_size] = self.dynamics.create_initial_condition(
                    np.array([0, 0, 0, 1]))

    def create_reference(self) -> None:
        """
        Create an evenly spaced reference for the satellites.
        """
        self.x_ref = np.zeros((self.total_state_size, 1))

        if isinstance(self.dynamics, SysDyn.TranslationalDynamics):

            if isinstance(self.dynamics, DifferentialDragDynamics):
                ref_rel_angles = np.linspace(0, 2 * np.pi, self.number_of_systems + 1, endpoint=False) \
                    .reshape((self.number_of_systems + 1, 1))[1:]
                ref_rel_angles -= np.mean(ref_rel_angles - self.x0[self.angle_states]) * self.number_of_systems / (self.number_of_systems + 1)
            else:
                ref_rel_angles = np.linspace(0, 2 * np.pi, self.number_of_systems, endpoint=False) \
                    .reshape((self.number_of_systems, 1))
                ref_rel_angles -= np.mean(ref_rel_angles - self.x0[self.angle_states])

            # print(ref_rel_angles)
            # Smaller reference for shorter sim:
            # alpha = 0
            # ref_rel_angles = alpha * self.x0[self.angle_states] + (1 - alpha) * ref_rel_angles

            for i in range(self.number_of_systems):
                self.x_ref[i * self.system_state_size:
                           (i + 1) * self.system_state_size] = self.dynamics.create_reference(ref_rel_angles[i, 0])
        elif isinstance(self.dynamics, SysDyn.AttitudeDynamics):
            for i in range(self.number_of_systems):
                self.x_ref[i * self.system_state_size:
                           (i + 1) * self.system_state_size] = self.dynamics.create_reference(np.array([0, 0, 0, 1]))

    @abstractmethod
    def set_initial_conditions(self, x0: np.ndarray) -> None:
        """
        Set the initial condition for SLS. If the system is LTV, the matrices are updated according to the initial
        condition.

        :param x0: The initial condition to set.
        """
        pass

    @abstractmethod
    def simulate_system(self, t_horizon: int, noise=None, progress: bool = False,
                        inputs_to_store: int = None, fast_computation: bool = False,
                        add_collision_avoidance: bool = False) -> (np.ndarray, np.ndarray):
        """
        Simulate the system assuming a perfect model is known.

        :param t_horizon: Horizon for which to simulate.
        :param noise: The noise present in the simulation
        :param progress: Whether to provide progress updates
        :param inputs_to_store: How many inputs to store in u_inputs.
                                Usually equal to t_horizon, but for simulation with RK4 can be set to 1.
        :param fast_computation: Whether to speed up the computations using a transformed problem.
        :param add_collision_avoidance: Whether to add collision avoidance constraints.
        :return: Tuple with (x_states, u_inputs)
        """
        pass

    def get_relative_angles(self) -> np.ndarray:
        """
        Return the relative angles of the satellites in degrees.

        :return: Array with the relative angles in degrees.
        """
        return np.rad2deg(self.x_states[self.angle_states, :])

    @abstractmethod
    def plot_inputs(self, satellite_numbers: np.ndarray = None, figure: plt.figure = None) -> plt.figure:
        """
        Plot the inputs in a figure.

        :param satellite_numbers: Numbers of the satellites to plot the inputs of. If None, plot all.
        :param figure: Figure to plot the inputs into. If None, make a new one.
        :return: Figure with the added inputs.
        """
        pass

    def plot_states(self, satellite_numbers: np.ndarray = None, figure: plt.figure = None) -> plt.figure:
        """
        Plot with states of the system.

        :param satellite_numbers: Numbers of the satellites to plot the states of. If None, plot all.
        :param figure: Figure to plot the states into. If None, make a new one.
        :return: Figure with the added states.
        """
        if satellite_numbers is None:
            satellite_numbers = np.arange(0, self.number_of_systems)

        for satellite_number in satellite_numbers:
            indices = np.arange(satellite_number * self.system_state_size,
                                (satellite_number + 1) * self.system_state_size)
            rel_states = self.x_states[indices, :].T - 0 * self.x_ref[indices].T

            if rel_states[0, self.angle_states[0]] > np.pi:
                rel_states[0, self.angle_states[0]] -= 2 * np.pi
            plot_method = self.dynamics.get_plot_method()
            figure = plot_method(rel_states,
                                 self.sampling_time,
                                 legend_name=f"SLS_prediction_{satellite_number}", figure=figure, linestyle='--')

        return figure
