import control
import numpy as np
from scipy.optimize import dual_annealing
import control as ct
import matplotlib.pyplot as plt

np.random.seed(129)


# def simulated_annealing_scipy()

def simulated_annealing(sampling_time: float, simulation_time: int, number_of_satellites: int, k_max: int = 1e6,
                        t_0: int = 100) -> np.ndarray:
    """
    Find the optimal inputs values for phasing.

    :param sampling_time: Sampling time of the model in s.
    :param simulation_time: Number of sampling steps.
    :param number_of_satellites: Number of satellites in the simulation.
    :param k_max: Maximum number of iterations.
    :param t_0: Initial temperature of the annealing algorithm.
    :return: Optimal inputs in shape (t, number_of_inputs).
    """
    model = create_model(number_of_satellites, sampling_time)
    sim_state_matrix, sim_input_matrix = simulate_model_matrices(model, simulation_time)

    # target_angles = np.linspace(0, 0.2 * np.pi, number_of_satellites, endpoint=False)[1:].reshape((-1, 1))
    target_angles = np.deg2rad(np.linspace(20, 80, number_of_satellites - 1))
    initial_state = np.zeros((model.A.shape[0], 1))
    initial_state[1::2] = create_initial_angular_velocity(number_of_satellites).reshape((-1, 1))
    # input_best = np.random.randint(0, 2, (simulation_time, model.B.shape[1]))
    input_best_raw = create_initial_guesses(target_angles.tolist(), sampling_time, initial_state[1::2])
    input_best = np.zeros((simulation_time, model.B.shape[1]))
    input_best[0:input_best_raw.shape[0], :] = input_best_raw
    states_best = (sim_state_matrix @ initial_state + sim_input_matrix @ input_best.reshape((-1, 1))).reshape(-1, (number_of_satellites - 1) * 2)
    best_cost = compute_cost(states_best, target_angles)

    for k in range(k_max):
        input_try = get_new_inputs(input_best)
        future_states = (sim_state_matrix @ initial_state + sim_input_matrix @ input_try.reshape((-1, 1))).reshape(-1, (number_of_satellites -1) * 2)
        new_cost = compute_cost(future_states, target_angles)
        # print(new_cost)

        if probability_calculator(best_cost, new_cost, k, k_max, t_0) >= np.random.rand():
            best_cost = new_cost
            input_best = input_try
            states_best = future_states

    plot_results(states_best)
    print(best_cost)
    return input_best


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
    # print(i, j)
    new_inputs = old_inputs.copy()
    new_inputs[i, j] = 1 - old_inputs[i, j]
    return new_inputs


def get_max_ang() -> float:
    """
    Find the maximum angular acceleration for the simulation.

    :return: Maximum angular acceleration in rad/s^2
    """
    ballistic_coefficient_high = 11
    ballistic_coefficient_low = 45
    atmosphere_density = 1e-13  # Roughly at 750 km
    radius = (750 + 6371) * 1000
    mu = 3.986e14
    v = np.sqrt(mu / radius**3) * radius
    pressure = 0.5 * atmosphere_density * v**2
    return 3 * pressure / radius * (1 / ballistic_coefficient_high - 1 / ballistic_coefficient_low)


def create_model(number_of_satellites: int, sampling_time: float | int) -> ct.LinearIOSystem:
    """
    Create a LTI system used for simulated annealing.

    :param number_of_satellites: Number of satellites to control.
    :param sampling_time: Sampling time.
    :return: Discrete linear LTI system with A=(2 * (n-1), 2 * (n-1)) and B=(2 * (n-1), n) with n=number_of_sats
    """
    A_single_sat = np.array([[1, sampling_time],
                             [0, 1]])

    B_single_sat = np.array([[0],
                             [get_max_ang() * sampling_time]])

    A_full_system = np.kron(np.eye(number_of_satellites - 1), A_single_sat)
    B_full_system = np.zeros((2 * (number_of_satellites - 1), number_of_satellites))
    B_full_system[1::2, 0] = -get_max_ang() * sampling_time
    B_full_system[:, 1:] = np.kron(np.eye(number_of_satellites - 1), B_single_sat)

    return ct.ss(A_full_system, B_full_system, np.eye(2 * (number_of_satellites - 1)), 0, sampling_time)


def simulate_model_matrices(model: ct.LinearIOSystem, simulation_time: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the matrices to simulate the system given inputs.

    :param model: Model to simulate.
    :param simulation_time: Duration of the simulation.
    :return: Matrices such that you can compute x_k+1 = A x_k + B u_k in one go for all k.
    """
    state_size, input_size = model.B.shape
    state_matrix = np.zeros((state_size * (simulation_time + 1), state_size))
    input_matrix = np.zeros((state_size * (simulation_time + 1), input_size * simulation_time))

    state_matrix[:state_size] = np.eye(state_size)

    for t in range(simulation_time):
        state_matrix[state_size*(t+1):state_size*(t+2)] = model.A @ state_matrix[state_size*t:state_size*(t+1)]
        input_matrix[state_size*(t+1):state_size*(t+2)] = model.A @ input_matrix[state_size*t:state_size*(t+1)]
        input_matrix[state_size * (t + 1):state_size*(t + 2), input_size * t: input_size * (t+1)] = model.B

    # print(state_matrix, input_matrix)
    return state_matrix, input_matrix


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
        t = t_0 * (1 - k/k_max)
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

    return np.sum(error**2)


def plot_results(states: np.ndarray):
    """
    Plot the angular states.

    :param states: States in the shape (t, number_of_states)
    """

    plt.figure()
    plt.plot(np.rad2deg(states[:, 0::2]))
    plt.show()


def find_time_slots(delta_theta: float, theta_dot_0: float) -> tuple[float, float, float]:
    """
    Find the analytical solution to the timeslots required to move delta_theta rad with delta_theta_dot initial speed.

    :param delta_theta: Target angular difference in rad.
    :param theta_dot_0: Current angular velocity in rad/s.
    :return: (delta_t_A, delta_t_B, theta_ddot_A)
    """
    theta_ddot = get_max_ang()
    a = -theta_ddot
    b = -2 * theta_dot_0
    c = delta_theta - theta_dot_0**2 / (2 * theta_ddot)

    D = b**2 - 4 * a * c

    if D < 0:
        a = theta_ddot
        c = delta_theta + theta_dot_0**2 / (2 * theta_ddot)
        D = b ** 2 - 4 * a * c

    # print(a, b, c, D)

    delta_t_A = (-b - np.sqrt(D)) / (2 * a)
    if delta_t_A > 0:
        delta_t_B = (-theta_dot_0 + a * delta_t_A) / a

        if delta_t_B > 0:
            return delta_t_A, delta_t_B, -a

    delta_t_A = (-b + np.sqrt(D)) / (2 * a)
    delta_t_B = (-theta_dot_0 + a * delta_t_A) / a

    if delta_t_A > 0 and delta_t_B > 0:
        return delta_t_A, delta_t_B, -a

    raise Exception(f"No valid solution found for {delta_theta=} and {theta_dot_0=}!")


def create_initial_guesses(target_angles: list[float], sampling_time: float, initial_velocities: np.ndarray) -> np.ndarray:
    """
    Create an array of initial inputs to reach the target angles.

    :param target_angles: Target angles for the different satellites in rad.
    :param sampling_time: Sampling time of the controller in s.
    :param initial_velocities: Initial velocities of the satellites
    :return: Array of shape (t, number_of_satellites).
    """
    number_of_satellites = len(target_angles) + 1
    delta_t_max = 0
    delta_t_A_list = []
    delta_t_B_list = []
    theta_ddot_A_pos = []

    for idx, target_angle in enumerate(target_angles):
        delta_t_A, delta_t_B, theta_ddot_A = find_time_slots(target_angle, initial_velocities[idx])
        delta_t_A_list.append(int(np.round(delta_t_A / sampling_time) + 0.01))
        delta_t_B_list.append(int(np.round(delta_t_B / sampling_time) + 0.01))
        theta_ddot_A_pos.append(theta_ddot_A > 0)

        if delta_t_A_list[-1] + delta_t_B_list[-1] > delta_t_max:
            delta_t_max = delta_t_A_list[-1] + delta_t_B_list[-1]

    # print(delta_t_max, delta_t_A_list[-1], delta_t_B_list[-1])
    input_guesses = np.zeros((delta_t_max, number_of_satellites))
    inputs_reference = np.zeros((delta_t_max, number_of_satellites - 1))

    for satellite_idx, delta_t_A in enumerate(delta_t_A_list):
        delta_t_B = delta_t_B_list[satellite_idx]
        time_indices_A = np.arange(0, delta_t_A)
        time_indices_B = np.arange(time_indices_A[-1] + 1, time_indices_A[-1] + 1 + delta_t_B)

        if theta_ddot_A_pos[satellite_idx]:
            input_guesses[time_indices_A, satellite_idx + 1] = 1
            inputs_reference[time_indices_B, satellite_idx] = 1
        else:
            input_guesses[time_indices_B, satellite_idx + 1] = 1
            inputs_reference[time_indices_A, satellite_idx] = 1

    inputs_reference_prob = np.sum(inputs_reference, axis=1) / (number_of_satellites - 1)
    inputs_random = np.random.random(inputs_reference_prob.shape)
    input_guesses[:, 0] = inputs_reference_prob > inputs_random

    # print(input_guesses, np.sum(input_guesses, axis=0))
    # print(inputs_reference_prob, inputs_random)
    # print(input_guesses)

    return input_guesses


def create_initial_angular_velocity(number_of_satellites: int) -> np.ndarray:
    """
    Create the initial angular velocity array at the start.

    :param number_of_satellites: Number of satellites present.
    :return: Array with initial angular velocities.
    """
    high_velocity_satellites = number_of_satellites // 2
    low_velocity_satellites = number_of_satellites - high_velocity_satellites - 1

    high_velocity_speeds = np.random.random((high_velocity_satellites, )) / 5 + 0.3  # Range of 0.3-0.5 m/s
    low_velocity_speeds = np.random.random((low_velocity_satellites, )) / 5  # Range of 0-0.2 m/s)

    velocity_array = np.concatenate((np.sort(low_velocity_speeds), np.sort(high_velocity_speeds))) / (750e3 + 6371e3)
    return velocity_array


if __name__ == '__main__':
    day2seconds = 24 * 3600
    best_inputs = simulated_annealing(day2seconds, 100, 5, 100000, 100)
    # print(best_inputs)
    # delta_t_A, delta_t_B, theta_ddot_A = find_time_slots(np.deg2rad(40), np.deg2rad(0))
    # print(f"t_A = {round(delta_t_A / day2seconds)} days, t_B = {round(delta_t_B /  day2seconds)} days with theta_ddot_A positive: {theta_ddot_A > 0}")
