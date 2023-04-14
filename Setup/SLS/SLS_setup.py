from Dynamics import SystemDynamics as SysDyn
import numpy as np
from slspy import LTI_System, SLS, SLS_Obj_H2, LTV_System, SLS_Cons_Input, SLS_Cons_State
import random
from Visualisation import Plotting as Plot
import matplotlib.pyplot as plt


class SLSSetup:
    """
    Class to use the SLS framework.
    """

    def __init__(self, sampling_time: int, system_dynamics: SysDyn.TranslationalDynamics, tFIR: int):
        """
        Initialise the class to deal with the SLS framework.

        :param sampling_time: The sampling time used for control purposes.
        :param system_dynamics: Dynamics used for control design
        :param tFIR: FIR length of SLS algorithm.
        """
        self.sys = None
        self.synthesizer = None
        self.controller = None
        self.sampling_time = sampling_time
        self.tFIR = tFIR
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

    def create_system(self, number_of_systems: int) -> None:
        """
        Create a system of several homogeneous satellites. For LTV systems, these matrices are updated when setting the
        initial conditions using 'set_initial_condition'.

        :param number_of_systems: Number of satellites.
        """
        # system = self.dynamics.create_model(sampling_time=self.sampling_time, argument_of_latitude=0)
        # self.system_state_size, system_input_size = system.B.shape
        # self.total_state_size = number_of_systems * self.system_state_size
        # self.total_input_size = number_of_systems * system_input_size
        self.number_of_systems = number_of_systems
        self.system_state_size = self.dynamics.state_size
        self.total_state_size = self.number_of_systems * self.system_state_size
        self.total_input_size = self.number_of_systems * self.dynamics.input_size

        # Always create an LTV system. For LTI, this simply contains the same matrices over time.
        self.sys = LTV_System(Nx=self.total_state_size, Nu=self.total_input_size, Nw=3, tFIR=self.tFIR)
        self.__fill_matrices(np.linspace(0, 0, self.number_of_systems))  # Is updated before use when setting init. pos
        self.sys._C2 = np.eye(self.total_state_size)  # All states as measurements

        # A_system = np.kron(np.eye(self.number_of_systems, dtype=int), np.round(system.A, 10))
        # B2_system = np.kron(np.eye(self.number_of_systems, dtype=int), np.round(system.B, 10))
        #
        # self.sys._A = A_system
        # self.sys._B1 = np.tile(system.B, (self.number_of_systems, 1))
        # self.sys._B2 = B2_system

        # Some parameters to use later
        angle_bools = np.tile(self.dynamics.get_positional_angles(), self.number_of_systems)
        self.angle_states = np.arange(0, self.total_state_size)[angle_bools]

    def __fill_matrices(self, arguments_of_latitude: np.ndarray) -> None:
        """
        Fill the A and B matrices of the system for the given current arguments of latitude.

        :param arguments_of_latitude: Arguments of latitude in degrees in shape (number_of_satellites,).
        """
        self.sys._A = np.zeros((self.tFIR, self.total_state_size, self.total_state_size))
        self.sys._B2 = np.zeros((self.tFIR, self.total_state_size, self.total_input_size))
        for t in range(self.tFIR):
            latitude_list = arguments_of_latitude + t * self.dynamics.mean_motion * self.sampling_time
            model = self.dynamics.create_multi_satellite_model(self.sampling_time,
                                                               argument_of_latitude_list=latitude_list)
            self.sys._A[t], self.sys._B2[t] = model.A, model.B

    def create_cost_matrices(self, Q_matrix_sqrt: np.ndarray = None, R_matrix_sqrt: np.ndarray = None) -> None:
        """
        Create the large cost matrices given the cost for each state/input.
        :param Q_matrix_sqrt: Matrix with square root values of the cost for each state.
                              If None, takes default of dynamical model.
        :param R_matrix_sqrt: Matrix with square toot values of the cost for each input.
                              If None, takes default of dynamical model.
        """
        if Q_matrix_sqrt is None:
            Q_matrix_sqrt = self.dynamics.get_state_cost_matrix_sqrt()

        if R_matrix_sqrt is None:
            R_matrix_sqrt = self.dynamics.get_input_cost_matrix_sqrt()

        full_Q_matrix = np.kron(np.eye(self.number_of_systems, dtype=int), Q_matrix_sqrt)
        full_R_matrix = np.kron(np.eye(self.number_of_systems, dtype=int), R_matrix_sqrt)

        # Set them as matrices for the regulator
        self.sys._C1 = full_Q_matrix
        self.sys._D12 = full_R_matrix

    def create_spaced_x0(self, number_of_dropouts: int, seed: int = 129, add_small_velocity: bool = False) -> None:
        """
        Create an evenly spaced initial state given a number of dropouts.

        :param number_of_dropouts: Number of satellites that have dropped out of their orbit.
        :param seed: Random seed.
        :param add_small_velocity: Add a small offset to the initial velocities to help with convergence.
                                   Seemed to be more important in Matlab. No effect currently
        """
        number_of_original_systems = self.number_of_systems + number_of_dropouts
        start_angles = np.linspace(0, 2 * np.pi, number_of_original_systems, endpoint=False) \
            .reshape((number_of_original_systems, 1))

        random.seed(seed)
        selected_indices = random.sample(range(number_of_original_systems),
                                         number_of_original_systems - number_of_dropouts)
        angles_sorted = start_angles[np.sort(selected_indices)]

        self.x0 = np.zeros((self.total_state_size, 1))
        for i in range(self.number_of_systems):
            self.x0[i * self.system_state_size:
                    (i + 1) * self.system_state_size] = self.dynamics.create_initial_condition(angles_sorted[i, 0])

        # Add small velocity to help system
        if add_small_velocity:
            raise Exception("Not implemented currently")
        #     rel_vel_angle_states = np.arange(4, self.total_state_size - 1, self.system_state_size)
        #     self.x0[rel_vel_angle_states] = 0.000001

    def create_reference(self) -> None:
        """
        Create an evenly spaced reference for the satellites.
        """
        self.x_ref = np.zeros((self.total_state_size, 1))
        ref_rel_angles = np.linspace(0, 2 * np.pi, self.number_of_systems, endpoint=False) \
            .reshape((self.number_of_systems, 1))
        ref_rel_angles -= np.max(ref_rel_angles - self.x0[self.angle_states])

        for i in range(self.number_of_systems):
            self.x_ref[i * self.system_state_size:
                       (i + 1) * self.system_state_size] = self.dynamics.create_reference(ref_rel_angles[i, 0])

    def set_initial_conditions(self, x0: np.ndarray) -> None:
        """
        Set the initial condition for SLS. If the system is LTV, the matrices are updated according to the initial
        condition.

        :param x0: The initial condition to set.
        """
        self.x0 = x0
        if not self.dynamics.is_LTI:  # Update every time for LTV system
            angles = np.rad2deg(self.x0[self.angle_states].reshape((-1,)))
            self.__fill_matrices(angles)

    def simulate_system(self, t_horizon: int, noise=None, progress: bool = False,
                        inputs_to_store: int = None) -> (np.ndarray, np.ndarray):
        """
        Simulate the system assuming a perfect model is known.

        :param t_horizon: Horizon for which to simulate.
        :param noise: The noise present in the simulation
        :param progress: Whether to provide progress updates
        :param inputs_to_store: How many inputs to store in u_inputs.
                                Usually equal to t_horizon, but for simulation with RK4 can be set to 1.
        :return: Tuple with (x_states, u_inputs)
        """
        if noise is None:  # and self.synthesizer is None:
            self.synthesizer = SLS(system_model=self.sys, FIR_horizon=self.tFIR, noise_free=True)

            # set SLS objective
            self.synthesizer << SLS_Obj_H2()

            input_constraint = np.array(self.dynamics.get_input_constraint()*self.number_of_systems).reshape((-1, 1))
            self.synthesizer << SLS_Cons_Input(state_feedback=True,
                                               maximum_input=input_constraint)

            state_constraint = np.array(self.dynamics.get_state_constraint()*self.number_of_systems).reshape((-1, 1))
            self.synthesizer += SLS_Cons_State(state_feedback=True,
                                               maximum_state=state_constraint)
            # Make it distributed
            # self.synthesizer << SLS_Cons_dLocalized(d=4)
        elif noise is not None:
            raise Exception("Sorry, not implemented yet!")

        if inputs_to_store is None:
            inputs_to_store = t_horizon
        elif inputs_to_store != 1:
            raise Exception("This value is not supported yet!")

        self.x_states = np.zeros((self.total_state_size, t_horizon + 1))
        self.u_inputs = np.zeros((self.total_input_size, inputs_to_store))

        self.x_states[:, 0:1] = self.x0
        progress_counter = 0

        for t in range(t_horizon):
            if progress and t / t_horizon * 100 > progress_counter:
                print(f"SLS simulation progress: {progress_counter}%. ")
                progress_counter = max(progress_counter + 10, int(t / t_horizon * 100))

            # Set past position as initial state
            self.set_initial_conditions(self.x_states[:, t:t + 1])
            self.sys.initialize(self.x0)

            # Synthesise controller
            self.controller = self.synthesizer.synthesizeControllerModel(self.x_ref)

            # Update state and input
            self.u_inputs[:, t] = self.controller._Phi_u[1] @ self.x_states[:, t]  # Do first, you append x in next step
            self.x_states[:, t + 1] = self.controller._Phi_x[2] @ self.x_states[:, t]

            if inputs_to_store != t_horizon and inputs_to_store > 1:
                self.u_inputs[:, t + 1] = self.controller._Phi_u[2] @ self.x_states[:, t]

        return self.x_states, self.u_inputs

    def get_relative_angles(self) -> np.ndarray:
        """
        Return the relative angles of the satellites in degrees.

        :return: Array with the relative angles in degrees.
        """
        return np.rad2deg(self.x_states[self.angle_states, :])

    def find_latest_objective_value(self) -> float:
        """
        Find the objective value of the latest simulation.

        :return: Total cost
        """
        cost = 0
        for t in range(self.x_states.shape[1] - 1):
            cost += np.sum((self.sys._C1 @ (self.x_states[:, t:t + 1] - self.x_ref)) ** 2)
            cost += np.sum((self.sys._D12 @ self.u_inputs[:, t:t + 1]) ** 2)

        cost += np.sum((self.sys._C1 @ (self.x_states[:, -1:] - self.x_ref)) ** 2)
        return cost

    def plot_inputs(self, satellite_numbers: np.ndarray = None, figure: plt.figure = None) -> plt.figure:
        """
        Plot the inputs in a figure.

        :param satellite_numbers: Numbers of the satellites to plot the inputs of. If None, plot all.
        :param figure: Figure to plot the inputs into. If None, make a new one.
        :return: Figure with the added inputs.
        """
        if satellite_numbers is None:
            satellite_numbers = np.arange(0, self.number_of_systems)

        for satellite_number in satellite_numbers:
            figure = Plot.plot_inputs(self.u_inputs[satellite_number * 3:(satellite_number + 1) * 3, :].T,
                                      self.sampling_time, f"Inputs_{satellite_number}", figure, linestyle='--')

        return figure

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
            plot_method = self.dynamics.get_plot_method()
            figure = plot_method(self.x_states[satellite_number * self.system_state_size:
                                               (satellite_number + 1) * self.system_state_size, :].T,
                                 self.sampling_time,
                                 f"SLS_prediction_{satellite_number}", figure, linestyle='--')

        return figure
