from tudatpy.kernel.astro import element_conversion
from Dynamics import SystemDynamics as SysDyn
from Dynamics.BlendDynamics import BlendSmall
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

        self.arguments_of_latitude = None
        self.arguments_of_periapsis = None

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
        self.__fill_matrices(np.linspace(0, 0, self.number_of_systems), np.linspace(0, 0, self.number_of_systems))  # Is updated before use when setting init. pos
        self.sys._C2 = np.eye(self.total_state_size)  # All states as measurements

        # Some parameters to use later for translational systems
        if isinstance(self.dynamics, SysDyn.TranslationalDynamics):
            angle_bools = np.tile(self.dynamics.get_positional_angles(), self.number_of_systems)
            self.angle_states = np.arange(0, self.total_state_size)[angle_bools]

            all_angle_bools = np.array(self.dynamics.get_angles_list() * self.number_of_systems)
            self.all_angle_states = np.arange(0, self.total_state_size)[all_angle_bools]

    def __fill_matrices(self, arguments_of_latitude: np.ndarray, argument_of_periapsis: np.ndarray) -> None:
        """
        Fill the A and B matrices of the system for the given current arguments of latitude.

        :param arguments_of_latitude: Arguments of latitude in degrees in shape (number_of_satellites,).
        """
        self.sys._A = np.zeros((self.prediction_horizon, self.total_state_size, self.total_state_size))
        self.sys._B2 = np.zeros((self.prediction_horizon, self.total_state_size, self.total_input_size))
        argument_of_periapsis_dot = self.dynamics.get_orbital_differentiation()[4]
        true_anomaly_start = arguments_of_latitude - argument_of_periapsis
        mean_anomaly_start = np.zeros_like(true_anomaly_start)
        true_anomalies = np.zeros_like(true_anomaly_start)

        # for idx, true_anomaly in enumerate(true_anomaly_start):
        #     mean_anomaly_start[idx] = element_conversion.true_to_mean_anomaly(self.dynamics.eccentricity, true_anomaly)

        for t in range(self.prediction_horizon):
            # mean_anomalies = mean_anomaly_start + (t + 0.5) * self.dynamics.mean_motion * self.sampling_time
            # for idx, mean_anomaly in enumerate(mean_anomalies):
            #     true_anomalies[idx] = element_conversion.mean_to_true_anomaly(self.dynamics.eccentricity, mean_anomaly)
            true_anomalies = true_anomaly_start + (t + 0.5) * self.dynamics.mean_motion * self.sampling_time

            arguments_of_periapsis = argument_of_periapsis + (t + 0.5) * argument_of_periapsis_dot * self.sampling_time

            latitude_list = true_anomalies + arguments_of_periapsis
            model = self.dynamics.create_multi_satellite_model(self.sampling_time,
                                                               argument_of_latitude_list=latitude_list.tolist(),
                                                               true_anomaly_list=true_anomalies.tolist())
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

    def set_initial_conditions(self, x0: np.ndarray, time_since_start: int = 0,
                               true_anomalies: list[float | int] = None) -> None:
        """
        Set the initial condition for SLS. If the system is LTV, the matrices are updated according to the initial
        condition.

        :param x0: The initial condition to set.
        :param time_since_start: Time since start of the simulation in s.
        :param true_anomalies: True anomalies of the system. Only have to be provided for SMALL_BLEND.
        """
        self.x0 = x0
        if not self.dynamics.is_LTI:  # Update every time for LTV system
            self.arguments_of_latitude, self.arguments_of_periapsis = self.dynamics.get_latitude_and_periapsis(x0, time_since_start)

            if true_anomalies is not None:
                self.arguments_of_periapsis = self.arguments_of_latitude - np.array(true_anomalies)
            # print(x0, self.arguments_of_latitude, self.arguments_of_periapsis)
            self.__fill_matrices(self.arguments_of_latitude, self.arguments_of_periapsis)

            if self.synthesizer is not None:
                self.synthesizer.update_model(self.sys)

    def simulate_system(self, t_horizon: int, noise=None, progress: bool = False, inputs_to_store: int = 1,
                        fast_computation: bool = False, time_since_start: int = 0,
                        add_collision_avoidance: bool = False, absolute_longitude_refs: list[float | int] = None,
                        current_true_anomalies: list[float | int]= None) -> (np.ndarray, np.ndarray):
        """
        Simulate the system assuming a perfect model is known.

        :param t_horizon: Horizon for which to simulate.
        :param noise: The noise present in the simulation
        :param progress: Whether to provide progress updates
        :param inputs_to_store: How many inputs to store in u_inputs.
                                Usually equal to t_horizon, but for simulation with RK4 can be set to 1.
        :param fast_computation: Whether to speed up the computations using a transformed problem.
        :param time_since_start: Time since start of all simulations in s. Used for latitude calculations.
        :param add_collision_avoidance: Whether to add collision avoidance constraints.
        :param absolute_longitude_refs: List with the absolute RAAN refs
        :param current_true_anomalies: List of current true anomalies. Only needed for Small blend
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
            elif noise is None and not add_collision_avoidance:
                self.synthesizer = OSQP_Synthesiser(self.number_of_systems, self.prediction_horizon, self.sys,
                                                    0*self.x_ref, self.dynamics.get_slack_variable_length(),
                                                    self.dynamics.get_slack_costs())
                self.synthesizer.create_optimisation_problem(self.dynamics.get_state_cost_matrix_sqrt(),
                                                             self.dynamics.get_input_cost_matrix_sqrt()[3:],
                                                             self.dynamics.get_state_constraint(),
                                                             self.dynamics.get_input_constraint())
            elif noise is None:
                self.synthesizer = OSQP_Synthesiser(self.number_of_systems, self.prediction_horizon, self.sys,
                                                    0 * self.x_ref, self.dynamics.get_slack_variable_length(),
                                                    self.dynamics.get_slack_costs(), inter_planetary_constraints=True,
                                                    longitudes=absolute_longitude_refs,
                                                    reference_angles=self.x_ref[self.angle_states],
                                                    planar_state=list(self.dynamics.get_positional_angles()),
                                                    inter_planar_state=self.dynamics.get_orbital_parameter(),
                                                    inter_planetary_limit=self.dynamics.get_inter_planetary_distance(),
                                                    planetary_limit=self.dynamics.get_planetary_distance(),
                                                    radial_limit=self.dynamics.get_radial_distance(),
                                                    mean_motion=self.dynamics.mean_motion,
                                                    sampling_time=self.sampling_time)
                self.synthesizer.create_optimisation_problem(self.dynamics.get_state_cost_matrix_sqrt(),
                                                             self.dynamics.get_input_cost_matrix_sqrt()[3:],
                                                             self.dynamics.get_state_constraint(),
                                                             self.dynamics.get_input_constraint())
            elif noise is not None:
                raise Exception("Sorry, not implemented yet!")

        if inputs_to_store != 1 and t_horizon != 1:
            raise Exception("This value is not supported yet!")
            # inputs_to_store = t_horizon
        # elif inputs_to_store != 1 and inputs_to_store != self.prediction_horizon:
        #

        self.x_states = np.zeros((self.total_state_size, t_horizon * inputs_to_store + 1))
        self.u_inputs = np.zeros((self.total_input_size, t_horizon * inputs_to_store))

        self.x_states[:, 0:1] = self.x0
        progress_counter = 0

        for t in range(t_horizon):
            if progress and t / t_horizon * 100 > progress_counter:
                print(f"SLS simulation progress: {progress_counter}%. ")
                progress_counter = max(progress_counter + 10, int(t / t_horizon * 100))

            # Set past position as initial state
            if isinstance(self.dynamics, BlendSmall):
                self.set_initial_conditions(self.x_states[:, t:t + 1], time_since_start + t * self.sampling_time, current_true_anomalies)
            else:
                self.set_initial_conditions(self.x_states[:, t:t + 1], time_since_start + t * self.sampling_time)

            # if fast_computation:
            #     difference = self.x0 - self.x_ref
            #
            #     difference[self.angle_states[difference[self.angle_states, 0] > np.pi]] -= 2 * np.pi
            #     difference[self.angle_states[difference[self.angle_states, 0] < -np.pi]] += 2 * np.pi
            #
            #     # print(np.max(np.abs(difference.reshape((-1, 6))), axis=0))
            #     self.sys.initialize(difference)
            #     # print((self.x0 - self.x_ref))
            # else:
            self.sys.initialize(self.x0)
            # print(np.max(np.abs(self.x0.reshape((6, -1))), axis=1))

            # Synthesise controller
            self.controller = self.synthesizer.synthesizeControllerModel(self.x_ref, time_since_start + t * self.sampling_time)

            # print(f"Predicted next state: {(self.controller._Phi_x[2] + self.x_ref).T}")
            # Update state and input
            if fast_computation:
                for i in range(inputs_to_store):
                    self.u_inputs[:, t + i:t + i + 1] = self.controller._Phi_u[i+1]
                    self.x_states[:, t + i + 1:t + i + 2] = self.controller._Phi_x[i+2]

                # if inputs_to_store > 1:
                #     self.u_inputs[:, t + 1:t + 2] = self.controller._Phi_u[]
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


