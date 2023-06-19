import numpy as np
import control as ct
import matplotlib.pyplot as plt
from Controllers.Controller import Controller
from Dynamics import SystemDynamics as SysDyn
from Visualisation import Plotting as Plot
from scipy import sparse
from Dynamics.DifferentialDragDynamics import DifferentialDragDynamics as dyn
import time
from Scenarios.MainScenarios import ScenarioEnum
np.random.seed(129)


def get_new_inputs(old_inputs: np.ndarray, i: int = None) -> np.ndarray:
    """
        Find random new inputs by flipping a value.

        :param old_inputs: The currently best inputs.
        :param i: Time of input to change
        :return: The new inputs.
        """
    if i is None:
        i = np.random.randint(0, old_inputs.shape[0])
    j = np.random.randint(0, old_inputs.shape[1])
    new_inputs = old_inputs.copy()
    new_inputs[i, j] = 1 - old_inputs[i, j]
    return new_inputs


def probability_calculator(cost: float, cost_new: float, k: int, k_max: int, t_0: int) -> float:
    """
    Probability_calculator for simulated annealing algorithm.

    :param cost: Current best cost
    :param cost_new: New cost
    :param k: Iteration
    :param k_max: Maximum iteration
    :param t_0: Initial temperature
    :return: Probability of accepting new cost as best cost
    """
    if cost_new < cost:
        return 1
    else:
        t = t_0 * (1 - k / k_max)
        return np.exp((cost - cost_new) / t)


def compute_cost(states_over_time: np.ndarray, target_positions: np.ndarray) -> float:
    """
    Compute the cost of the state predictions

    :param states_over_time: Array with complete state of the system over time (t, number_of_states)
    :param target_positions: Desired angular positions in rad.
    :return: Cost of the current state
    """
    predicted_angles = states_over_time[:, 0::2]
    error = np.rad2deg(predicted_angles - target_positions.T)

    return np.sum(error ** 2)


class SimulatedAnnealing(Controller):
    """
    Class for Simulated Annealing approach of Planet Labs.
    """

    def __init__(self, sampling_time: float | int, system_dynamics: SysDyn.TranslationalDynamics,
                 prediction_horizon: int):
        """
        Initialise the Simulated Annealing class.

        :param sampling_time: Sampling time of the controller in s.
        :param prediction_horizon: How many steps to look ahead.
        """
        super().__init__(sampling_time, system_dynamics, prediction_horizon)

    def create_system(self, number_of_systems: int) -> None:
        """
        Create a system of several homogeneous satellites. For LTV systems, these matrices are updated when setting the
        initial conditions using 'set_initial_condition'.

        :param number_of_systems: Number of satellites.
        """
        self.number_of_systems = number_of_systems - 1  # Mostly act as if one less system is present: ref always at 0
        self.system_state_size = self.dynamics.state_size
        self.total_state_size = self.number_of_systems * self.system_state_size
        self.total_input_size = (self.number_of_systems + 1) * self.dynamics.input_size

        single_model = self.dynamics.create_model(sampling_time=self.sampling_time)
        A_full_system = np.kron(np.eye(self.number_of_systems), single_model.A)
        B_full_system = np.zeros((2 * self.number_of_systems, self.number_of_systems + 1))
        B_full_system[1::2, 0] = -single_model.B[1, 0] * self.sampling_time
        B_full_system[:, 1:] = np.kron(np.eye(self.number_of_systems), single_model.B)

        self.sys = ct.ss(A_full_system, B_full_system, np.eye(2 * self.number_of_systems), 0, self.sampling_time)

        # Some parameters to use later
        angle_bools = np.tile(self.dynamics.get_positional_angles(), self.number_of_systems)
        self.angle_states = np.arange(0, self.total_state_size)[angle_bools]

        all_angle_bools = np.array(self.dynamics.get_angles_list() * self.number_of_systems)
        self.all_angle_states = np.arange(0, self.total_state_size)[all_angle_bools]

    def set_initial_conditions(self, x0: np.ndarray) -> None:
        """
        Set the initial condition for SLS. If the system is LTV, the matrices are updated according to the initial
        condition.

        :param x0: The initial condition to set.
        """
        self.x0 = x0

    def simulate_system(self, t_horizon: int, noise=None, progress: bool = False,
                        inputs_to_store: int = None, fast_computation: bool = False,
                        time_since_start: int = 0) -> (np.ndarray, np.ndarray):
        """
        Simulate the system assuming a perfect model is known.

        :param time_since_start: Unused
        :param t_horizon: Horizon for which to simulate.
        :param noise: The noise present in the simulation
        :param progress: Whether to provide progress updates
        :param inputs_to_store: How many inputs to store in u_inputs.
                                Usually equal to t_horizon, but for simulation with RK4 can be set to 1.
        :param fast_computation: Whether to speed up the computations using a transformed problem.
        :return: Tuple with (x_states, u_inputs)
        """
        if inputs_to_store is None:
            inputs_to_store = t_horizon
        elif inputs_to_store != 1:
            raise Exception("This value is not supported yet!")

        self.x_states = np.zeros((self.total_state_size, t_horizon + 1))
        self.u_inputs = np.zeros((self.total_input_size, inputs_to_store))

        self.x_states[:, 0:1] = self.x0
        progress_counter = 0

        # print(t_horizon)
        for t in range(t_horizon):
            if progress and t / t_horizon * 100 > progress_counter:
                print(f"SLS simulation progress: {progress_counter}%. ")
                progress_counter = max(progress_counter + 10, int(t / t_horizon * 100))

            # Set past position as initial state
            self.set_initial_conditions(self.x_states[:, t:t + 1])

            # Synthesise controller
            # print(int(((self.number_of_systems + 1 )/ 4)**2.7 * 3000))
            states, inputs, _, _ = self.__run_simulated_annealing(k_max=int(((self.number_of_systems + 1) / 4)**2.7 * 6000))

            # Update state and input
            self.u_inputs[:, t] = inputs[0]
            self.x_states[:, t + 1] = states[1]

            if inputs_to_store != t_horizon and inputs_to_store > 1:
                self.u_inputs[:, t + 1] = inputs[1]

        return self.x_states, \
            self.u_inputs * self.sys.B[1, 0] * self.dynamics.orbit_radius * self.dynamics.satellite_mass

    def __run_simulated_annealing(self, k_max: int = 1 * 100 ** 2,
                                  t_0: int = 100) -> tuple[np.ndarray, np.ndarray, list, list[int]]:
        """
        Find the optimal inputs values for phasing.

        :param k_max: Maximum number of iterations.
        :param t_0: Initial temperature of the annealing algorithm.
        :return: Optimal states and inputs in shape (t, number_of_states/number_of_inputs).
        """
        sim_state_matrix, sim_input_matrix = self.__simulate_model_matrices(self.prediction_horizon)
        target_angles = self.x_ref[self.angle_states]
        initial_state = self.x0

        input_best_raw = self.__create_initial_guesses(target_angles.tolist(), initial_state)
        input_best = np.zeros((self.prediction_horizon, self.sys.B.shape[1]))
        input_best[0:input_best_raw.shape[0], :] = input_best_raw[:self.prediction_horizon]
        states_best = self.update_states(sim_state_matrix, sim_input_matrix, initial_state, input_best)
        best_cost = compute_cost(states_best, target_angles)

        best_cost_list = [best_cost]
        iteration_list = [0]
        for k in range(k_max):
            input_try = get_new_inputs(input_best)
            future_states = self.update_states(sim_state_matrix, sim_input_matrix, initial_state, input_try)
            new_cost = compute_cost(future_states, target_angles)

            if probability_calculator(best_cost, new_cost, k, k_max, t_0) >= np.random.rand():
                best_cost = new_cost
                input_best = input_try
                states_best = future_states

                best_cost_list.append(best_cost)
                iteration_list.append(k + 1)

        return states_best, input_best, best_cost_list, iteration_list

    def update_states(self, sim_state_matrix, sim_input_matrix, initial_state, input_try):
        return (sim_state_matrix @ initial_state + sim_input_matrix @ input_try.reshape((-1, 1))).reshape(
            -1, self.total_state_size)

    def __create_initial_guesses(self, target_angles: list[float], current_state: np.ndarray) -> np.ndarray:
        """
        Create an array of initial inputs to reach the target angles.

        :param target_angles: Target angles for the different satellites in rad.
        :param current_state: Current angles/velocities for the different satellites in rad or rad/s.
        :return: Array of shape (t, number_of_satellites).
        """
        number_of_satellites = len(target_angles) + 1
        current_angles = current_state[0::2]
        initial_velocities = current_state[1::2]
        delta_t_max = 0
        delta_t_A_list = []
        delta_t_B_list = []
        theta_ddot_A_pos = []

        for idx, target_angle in enumerate(target_angles):
            delta_t_A, delta_t_B, theta_ddot_A = self.__find_time_slots(target_angle - current_angles[idx],
                                                                        initial_velocities[idx])
            delta_t_A_list.append(int(np.round(delta_t_A / self.sampling_time) + 0.01))
            delta_t_B_list.append(int(np.round(delta_t_B / self.sampling_time) + 0.01))
            theta_ddot_A_pos.append(theta_ddot_A > 0)

            if delta_t_A_list[-1] + delta_t_B_list[-1] > delta_t_max:
                delta_t_max = delta_t_A_list[-1] + delta_t_B_list[-1]

        input_guesses = np.zeros((delta_t_max, number_of_satellites))
        inputs_reference = np.zeros((delta_t_max, number_of_satellites - 1))

        for satellite_idx, delta_t_A in enumerate(delta_t_A_list):
            delta_t_B = delta_t_B_list[satellite_idx]
            time_indices_A = np.arange(0, delta_t_A)
            try:
                time_indices_B = np.arange(time_indices_A[-1] + 1, time_indices_A[-1] + 1 + delta_t_B)
            except IndexError:
                # print(delta_t_A, time_indices_A, delta_t_B, delta_t_max)
                time_indices_B = np.arange(0, delta_t_B)

            if theta_ddot_A_pos[satellite_idx]:
                input_guesses[time_indices_A, satellite_idx + 1] = 1
                inputs_reference[time_indices_B, satellite_idx] = 1
            else:
                input_guesses[time_indices_B, satellite_idx + 1] = 1
                inputs_reference[time_indices_A, satellite_idx] = 1

        inputs_reference_prob = np.sum(inputs_reference, axis=1) / (number_of_satellites - 1)
        inputs_random = np.random.random(inputs_reference_prob.shape)
        input_guesses[:, 0] = inputs_reference_prob > inputs_random

        return input_guesses

    def __find_time_slots(self, delta_theta: float, theta_dot_0: float) -> tuple[float, float, float]:
        """
        Find the analytical solution to the timeslots required to move delta_theta rad with delta_theta_dot initial speed.

        :param delta_theta: Target angular difference in rad.
        :param theta_dot_0: Current angular velocity in rad/s.
        :return: (delta_t_A, delta_t_B, theta_ddot_A)
        """
        theta_ddot = -self.sys.B[1, 0]
        a = -theta_ddot
        b = -2 * theta_dot_0
        c = delta_theta - theta_dot_0 ** 2 / (2 * theta_ddot)
        D = b ** 2 - 4 * a * c

        # Never allow imaginairy results
        if D < 0:
            a = theta_ddot
            c = delta_theta + theta_dot_0 ** 2 / (2 * theta_ddot)
            D = b ** 2 - 4 * a * c

        # Find solutions
        delta_t_A = (-b - np.sqrt(D)) / (2 * a)
        if delta_t_A > 0:
            delta_t_B = (-theta_dot_0 + a * delta_t_A) / a

            if delta_t_B > 0:
                return delta_t_A, delta_t_B, -a

        delta_t_A = (-b + np.sqrt(D)) / (2 * a)
        delta_t_B = (-theta_dot_0 + a * delta_t_A) / a

        if delta_t_A >= 0 and delta_t_B >= 0:
            return delta_t_A, delta_t_B, -a

        # Try with different D
        a = theta_ddot
        c = delta_theta + theta_dot_0 ** 2 / (2 * theta_ddot)
        D = b ** 2 - 4 * a * c

        # Find solutions
        delta_t_A = (-b - np.sqrt(D)) / (2 * a)
        if delta_t_A > 0:
            delta_t_B = (-theta_dot_0 + a * delta_t_A) / a

            if delta_t_B > 0:
                return delta_t_A, delta_t_B, -a

        delta_t_A = (-b + np.sqrt(D)) / (2 * a)
        delta_t_B = (-theta_dot_0 + a * delta_t_A) / a

        if delta_t_A >= 0 and delta_t_B >= 0:
            return delta_t_A, delta_t_B, -a

        raise Exception(f"No valid solution found for {delta_theta=} and {theta_dot_0=}!")

    def __simulate_model_matrices(self, simulation_time: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Find the matrices to simulate the system given inputs.

        :param model: Model to simulate.
        :param simulation_time: Duration of the simulation.
        :return: Matrices such that you can compute x_k+1 = A x_k + B u_k in one go for all k.
        """
        state_size, input_size = self.sys.B.shape
        state_matrix = np.zeros((state_size * (simulation_time + 1), state_size))
        input_matrix = np.zeros((state_size * (simulation_time + 1), input_size * simulation_time))

        state_matrix[:state_size] = np.eye(state_size)

        for t in range(simulation_time):
            state_matrix[state_size * (t + 1):state_size * (t + 2)] = self.sys.A @ state_matrix[
                                                                                   state_size * t:state_size * (t + 1)]
            input_matrix[state_size * (t + 1):state_size * (t + 2)] = self.sys.A @ input_matrix[
                                                                                   state_size * t:state_size * (t + 1)]
            input_matrix[state_size * (t + 1):state_size * (t + 2), input_size * t: input_size * (t + 1)] = self.sys.B

        return sparse.csc_matrix(state_matrix), sparse.csc_matrix(input_matrix)

    def plot_inputs(self, satellite_numbers: np.ndarray = None, figure: plt.figure = None) -> plt.figure:
        """
        Plot the inputs in a figure.

        :param satellite_numbers: Numbers of the satellites to plot the inputs of. If None, plot all.
        :param figure: Figure to plot the inputs into. If None, make a new one.
        :return: Figure with the added inputs.
        """
        if satellite_numbers is None:
            satellite_numbers = np.arange(0, self.number_of_systems + 1)

        for satellite_number in satellite_numbers:
            figure = Plot.plot_drag_forces(self.u_inputs[satellite_number:satellite_number + 1, :].T,
                                           self.sampling_time, f"Inputs_{satellite_number}", figure,
                                           linestyle='--')

        return figure

    def plot_annealing_progress(self, t_horizon: int, max_iterations: list[int] = None,
                                figure: plt.figure = None, legend_name: str = None) -> plt.figure:
        """
        Plot the process of the algorithm over the iterations.
        """
        if max_iterations is None:
            max_iterations = np.logspace(np.log10(10), np.log10(100000), num=10, dtype=int)

        x0 = self.x0

        costs = []
        for max_iteration in max_iterations:
            print(f"{max_iteration=}")

            x_states = np.zeros((self.total_state_size, t_horizon + 1))

            x_states[:, 0:1] = x0
            cost = 0
            for t in range(t_horizon):

                # Set past position as initial state
                self.set_initial_conditions(x_states[:, t:t + 1])

                # Synthesise controller
                states, _, cost_list, _ = self.__run_simulated_annealing(k_max=int(max_iteration))
                cost += cost_list[-1]

                # Update state and input
                x_states[:, t + 1] = states[1]

            costs.append(cost)

        if figure is None:
            figure = plt.figure()
            plt.grid(True)
            plt.xlabel(r'$\mathrm{Number \;of \; iterations\;[-]}$', fontsize=14)
            plt.ylabel(r'$\mathrm{Cost \;[-]}$', fontsize=14)

        plt.loglog(max_iterations, costs, label=legend_name)
        return figure


def time_optimisation(number_of_satellites: int, prediction_horizon: int):

    # General values
    scenario = ScenarioEnum.simple_scenario_translation_SimAn_scaled.value
    scenario.number_of_satellites = number_of_satellites
    scenario.control.tFIR = prediction_horizon
    dynamics = dyn(scenario)
    simAn = SimulatedAnnealing(scenario.control.control_timestep, dynamics, scenario.control.tFIR)

    simAn.create_system(number_of_systems=scenario.number_of_satellites)


    # Create x0 and x_ref
    simAn.create_x0(number_of_dropouts=int(scenario.initial_state.dropouts *
                                                scenario.number_of_satellites) + 1)
    simAn.create_reference()
    t_0 = time.time()
    runs = 1
    nsim = 10
    x = np.zeros((simAn.total_state_size, nsim + 1))
    # x[:, 0] = simAn.x0

    for run in range(runs):
        simAn.simulate_system(t_horizon=nsim)

    t_end = time.time()
    avg_time = (t_end - t_0) / runs / nsim

    return avg_time


if __name__ == '__main__':
    from Scenarios.MainScenarios import ScenarioEnum
    from Dynamics.DifferentialDragDynamics import DifferentialDragDynamics

    figure = None
    number_satellite_list = [3, 5, 7, 10, 15]

    for number_of_satellites in number_satellite_list:
        scenario = ScenarioEnum.simple_scenario_translation_SimAn_scaled.value
        scenario.number_of_satellites = number_of_satellites
        controller = SimulatedAnnealing(scenario.control.control_timestep, DifferentialDragDynamics(scenario),
                                        scenario.control.tFIR)
        controller.create_system(number_of_systems=scenario.number_of_satellites)
        controller.create_x0(number_of_dropouts=int(scenario.initial_state.dropouts *
                                                    scenario.number_of_satellites) + 1)
        controller.create_reference()

        t_horizon_control = int(np.ceil(scenario.simulation.simulation_duration /
                                        scenario.control.control_timestep)) + 1
        figure = controller.plot_annealing_progress(t_horizon_control, figure=figure,
                                                    legend_name=f"{number_of_satellites} sats")
    plt.legend(fontsize=12)
    plt.show()
