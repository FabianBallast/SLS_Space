from Dynamics import SystemDynamics as SysDyn
import numpy as np
from slspy import SLS, SLS_Obj_H2, LTV_System, SLS_Cons_Input, SLS_Cons_State
from Visualisation import Plotting as Plot
import matplotlib.pyplot as plt
from Controllers.Controller import Controller
from Optimisation.OSQP_Solver import OSQP_Synthesiser


class SLSSetup(Controller):
    """
    Class to use the SLS framework.
    """

    def __init__(self, sampling_time: int, system_dynamics: SysDyn.GeneralDynamics, prediction_horizon: int):
        """
        Initialise the class to deal with the SLS framework.

        :param sampling_time: The sampling time used for control purposes.
        :param system_dynamics: Dynamics used for control design
        :param prediction_horizon: FIR length of SLS algorithm.
        """
        super().__init__(sampling_time, system_dynamics, prediction_horizon)

    def create_system(self, number_of_systems: int) -> None:
        """
        Create a system of several homogeneous satellites. For LTV systems, these matrices are updated when setting the
        initial conditions using 'set_initial_condition'.

        :param number_of_systems: Number of satellites.
        """
        self.number_of_systems = number_of_systems
        self.system_state_size = self.dynamics.state_size
        self.total_state_size = self.number_of_systems * self.system_state_size
        self.total_input_size = self.number_of_systems * self.dynamics.input_size

        # Always create an LTV system. For LTI, this simply contains the same matrices over time.
        self.sys = LTV_System(Nx=self.total_state_size, Nu=self.total_input_size, Nw=3, tFIR=self.prediction_horizon)
        self.__fill_matrices(np.linspace(0, 0, self.number_of_systems))  # Is updated before use when setting init. pos
        self.sys._C2 = np.eye(self.total_state_size)  # All states as measurements

        # Some parameters to use later for translational systems
        if isinstance(self.dynamics, SysDyn.TranslationalDynamics):
            angle_bools = np.tile(self.dynamics.get_positional_angles(), self.number_of_systems)
            self.angle_states = np.arange(0, self.total_state_size)[angle_bools]

    def __fill_matrices(self, arguments_of_latitude: np.ndarray) -> None:
        """
        Fill the A and B matrices of the system for the given current arguments of latitude.

        :param arguments_of_latitude: Arguments of latitude in degrees in shape (number_of_satellites,).
        """
        self.sys._A = np.zeros((self.prediction_horizon, self.total_state_size, self.total_state_size))
        self.sys._B2 = np.zeros((self.prediction_horizon, self.total_state_size, self.total_input_size))
        for t in range(self.prediction_horizon):
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
                        inputs_to_store: int = None, fast_computation: bool = False) -> (np.ndarray, np.ndarray):
        """
        Simulate the system assuming a perfect model is known.

        :param t_horizon: Horizon for which to simulate.
        :param noise: The noise present in the simulation
        :param progress: Whether to provide progress updates
        :param inputs_to_store: How many inputs to store in u_inputs.
                                Usually equal to t_horizon, but for simulation with RK4 can be set to 1.
        :param fast_computation: Whether to speed up the computations using a transformed problem.
        :return: Tuple with (x_states, u_inputs)
        """
        if self.synthesizer is None:
            if noise is None and not fast_computation:  # and self.synthesizer is None:
                self.synthesizer = SLS(system_model=self.sys, FIR_horizon=self.prediction_horizon, noise_free=True,
                                       fast_computation=fast_computation)

                # set SLS objective
                self.synthesizer << SLS_Obj_H2()

                input_constraint = np.array(self.dynamics.get_input_constraint() * self.number_of_systems).reshape((-1, 1))
                self.synthesizer << SLS_Cons_Input(state_feedback=True,
                                                   maximum_input=input_constraint)

                state_constraint = np.array(self.dynamics.get_state_constraint() * self.number_of_systems).reshape((-1, 1))
                self.synthesizer += SLS_Cons_State(state_feedback=True,
                                                   maximum_state=state_constraint)
                # Make it distributed
                # self.synthesizer << SLS_Cons_dLocalized(d=4)
            elif noise is None:
                self.synthesizer = OSQP_Synthesiser(self.number_of_systems, self.prediction_horizon, self.sys, 0*self.x_ref)
                self.synthesizer.create_optimisation_problem(self.dynamics.get_state_cost_matrix_sqrt(),
                                                             self.dynamics.get_input_cost_matrix_sqrt()[3:],
                                                             self.dynamics.get_state_constraint(),
                                                             self.dynamics.get_input_constraint())
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

            if fast_computation:
                self.sys.initialize(self.x0 - self.x_ref)
            else:
                self.sys.initialize(self.x0)

            # Synthesise controller
            self.controller = self.synthesizer.synthesizeControllerModel(self.x_ref)

            # Update state and input
            if fast_computation:
                self.u_inputs[:, t:t + 1] = self.controller._Phi_u[1]
                self.x_states[:, t + 1:t + 2] = self.controller._Phi_x[2] + self.x_ref

                if inputs_to_store != t_horizon and inputs_to_store > 1:
                    self.u_inputs[:, t + 1:t + 2] = self.controller._Phi_u[2]
            else:
                self.u_inputs[:, t] = self.controller._Phi_u[1] @ self.x_states[:, t]
                self.x_states[:, t + 1] = self.controller._Phi_x[2] @ self.x_states[:, t]

                if inputs_to_store != t_horizon and inputs_to_store > 1:
                    self.u_inputs[:, t + 1] = self.controller._Phi_u[2] @ self.x_states[:, t]

        return self.x_states, self.u_inputs

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
            figure = Plot.plot_thrust_forces(self.u_inputs[satellite_number * 3:(satellite_number + 1) * 3, :].T,
                                             self.sampling_time, f"Inputs_{satellite_number}", figure, linestyle='--')

        return figure


