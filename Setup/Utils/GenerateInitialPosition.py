import random
import Dynamics.SystemDynamics as SysDyn
from Dynamics.DifferentialDragDynamics import DifferentialDragDynamics
import numpy as np
# from Utils.HungarianAlgorithm import hungarian_algorithm, ans_calculation
from scipy.spatial.distance import cdist
import munkres
import matplotlib.pyplot as plt
import os, pickle
from Visualisation.PlotResults import plot_theta_Omega_triple


def generate_anomalies_and_longitudes(number_of_dropouts: int, longitude_list: list[float], number_of_systems: int,
                                      dynamics: SysDyn.GeneralDynamics, advanced_assignment: bool=True) -> tuple[list[float], list[float], list[int]]:
    """
    Generate the initial true anomalies and longitudes for the satellites.

    :param number_of_dropouts: Number of dropouts.
    :param longitude_list: List with the longitudinal angles to divide the satellites into.
    :param number_of_systems: Number of (controlled) satellites.
    :param dynamics: The dynamics used for the controller.
    :param advanced_assignment: Whether to use an advanced assignment strategy.
    :return: List with the true anomalies and a list with the longitudes for each satellite in radians.
    """
    if number_of_dropouts is None:
        raise Exception("Enter number of dropouts!")
    number_of_original_systems = number_of_systems + number_of_dropouts

    try:
        number_of_planes = len(longitude_list)
    except TypeError:
        longitude_list = [longitude_list]
        number_of_planes = 1
    random.seed(100)

    if number_of_planes == 1 or number_of_systems == 1:
        if isinstance(dynamics, DifferentialDragDynamics):
            number_of_original_systems += 1

        start_angles = np.linspace(0, 2 * np.pi, number_of_original_systems, endpoint=False) \
            .reshape((number_of_original_systems, 1))
        selected_indices = random.sample(range(number_of_original_systems),
                                         number_of_original_systems - number_of_dropouts)

        if isinstance(dynamics, DifferentialDragDynamics):
            if min(selected_indices) != 0:
                selected_indices[-1] = 0
            angles_sorted = start_angles[np.sort(selected_indices)][1:]
        else:
            angles_sorted = start_angles[np.sort(selected_indices)]

        if number_of_systems == 1:
            angles_sorted = np.array([[np.deg2rad(15)]])
            longitude_list = [longitude_list[0] - 5 / np.cos(np.pi / 4)]

        return angles_sorted[:, 0].tolist(), longitude_list * number_of_systems, []

    else:
        if number_of_original_systems % number_of_planes != 0:
            number_of_original_systems = number_of_planes * (number_of_original_systems // number_of_planes + 1)
            number_of_dropouts = number_of_original_systems - number_of_systems

        satellites_per_plane_start = number_of_original_systems // number_of_planes
        satellites_per_plane_end = number_of_systems // number_of_planes

        start_angles = np.tile(np.linspace(0, 2 * np.pi, satellites_per_plane_start, endpoint=False), number_of_planes).reshape((-1, 1))

        selected_indices = random.sample(range(number_of_original_systems), number_of_original_systems - number_of_dropouts)

        angles_sorted = start_angles[np.sort(selected_indices)]

        longitudes = np.tile(np.array(longitude_list).reshape((-1, 1)), satellites_per_plane_start).flatten()
        longitudes_selected = longitudes[np.sort(selected_indices)]

        if advanced_assignment:
            sorted_start_state_vector, order_matrix_end = find_advanced_assignment(angles_sorted,
                                                                                   longitudes_selected,
                                                                                   satellites_per_plane_end,
                                                                                   longitude_list)

            return np.deg2rad(sorted_start_state_vector[:, 0]).tolist(), \
                   (sorted_start_state_vector[:, 1]).tolist(), order_matrix_end
        else:
            return angles_sorted[:, 0].tolist(), longitudes_selected.tolist(), []


def generate_reference(number_of_systems: int, anomaly_list: list[float], longitude_list: list[float]):
    """
    Generate a reference for the satellites.

    :param number_of_systems: Total number of satellites.
    :param anomaly_list: List with the initial true anomalies.
    :param longitude_list: List with the initial longitudes.

    :return: Array with the reference angles.
    """
    try:
        number_of_planes = len(longitude_list)
    except TypeError:
        longitude_list = [longitude_list]
        number_of_planes = 1

    # if number_of_systems > 1:
    satellites_per_plane = number_of_systems // number_of_planes
    reference = np.tile(np.linspace(0, 2 * np.pi, satellites_per_plane, endpoint=False), number_of_planes)
    # else:
    #     reference = np.linspace(0, 2 * np.pi, 1, endpoint=False)


    anomalies_rad = np.deg2rad(anomaly_list)

    reference[reference - anomalies_rad > np.pi] -= 2 * np.pi
    reference[reference - anomalies_rad < -np.pi] += 2 * np.pi

    if number_of_systems > 1:
        reference -= np.mean(reference - anomalies_rad)

    return reference.reshape((-1, 1))


def find_advanced_assignment(angles_sorted: np.ndarray, longitudes_selected: np.ndarray, satellites_per_plane_end: int,
                             longitude_list: list):
    """
    Find an advanced assignment such that the satellites are efficiently assigned.
    :return: Sorted vector with both theta-Omega pairs for the starting and end positions.
    """
    number_of_planes = len(longitude_list)
    sorted_start_state_vector = None
    order_matrix_end = None

    try:
        for file in os.listdir("../Setup/Utils/AssignmentData"):
            if file.endswith(f'({number_of_planes},{satellites_per_plane_end})'):
                with open(os.path.join("../Setup/Utils/AssignmentData", file), 'rb') as f:
                    sorted_start_state_vector = pickle.load(f)
                    order_matrix_end = pickle.load(f)

                    # longitudes_end = np.tile(np.array(longitude_list).reshape((-1, 1)), satellites_per_plane_end).reshape((-1, 1))
                    # angles_end = np.tile(np.linspace(0, 360, satellites_per_plane_end, endpoint=False), len(longitude_list)).reshape((-1, 1))
                    # ending_state_vector = np.concatenate((angles_end, longitudes_end), axis=1)
                    #
                    # theta_values = np.unwrap(np.concatenate((sorted_start_state_vector[:, 0:1], ending_state_vector[:, 0:1]), axis=1), axis=1, period=360).T
                    # Omega_values = np.concatenate((sorted_start_state_vector[:, 1:2], ending_state_vector[:, 1:2]), axis=1).T
                    #
                    # x_interp = np.linspace(0, 1,  num=100).reshape((-1, 1))
                    #
                    # delta_theta = theta_values[-1] - theta_values[0]
                    # delta_Omega = (Omega_values[-1] - Omega_values[0])
                    # delta_Omega-= (delta_Omega > 180) * 360
                    # # delta_theta = theta_values[]
                    #
                    # theta_values = theta_values[0] + x_interp * delta_theta
                    # Omega_values = Omega_values[0] + x_interp * delta_Omega
                    #
                    # fig = plot_theta_Omega_triple(np.deg2rad(theta_values), np.zeros_like(theta_values), np.deg2rad(Omega_values), np.zeros_like(Omega_values))
                    # fig.savefig("../Setup/Utils/Figures/assignment.eps")

                    # plt.show()
                return sorted_start_state_vector, order_matrix_end
    except FileNotFoundError:
        for file in os.listdir("../../../Setup/Utils/AssignmentData"):
            if file.endswith(f'({number_of_planes},{satellites_per_plane_end})'):
                with open(os.path.join("../../../Setup/Utils/AssignmentData", file), 'rb') as f:
                    sorted_start_state_vector = pickle.load(f)
                    order_matrix_end = pickle.load(f)

                return sorted_start_state_vector, order_matrix_end

    starting_state_vector = np.concatenate((np.rad2deg(angles_sorted), longitudes_selected.reshape((-1, 1))), axis=1)

    longitudes_end = np.tile(np.array(longitude_list).reshape((-1, 1)), satellites_per_plane_end).reshape((-1, 1))
    angles_end = np.tile(np.linspace(0, 360, satellites_per_plane_end, endpoint=False), len(longitude_list)).reshape((-1, 1))

    ending_state_vector = np.concatenate((angles_end, longitudes_end), axis=1)

    cost_matrix = cdist(starting_state_vector, ending_state_vector, cost_function)
    # ans_pos = hungarian_algorithm(cost_matrix.copy())  # Get the element position.
    # ans, ans_mat = ans_calculation(cost_matrix, ans_pos) # Get the minimum value and corresponding matrix for visualisation
    print('Start Hungarian algorithm')
    m = munkres.Munkres()
    indexes = m.compute(cost_matrix.copy())
    # munkres.print_matrix(cost_matrix.copy(), msg='Lowest cost through this matrix:')
    # total = 0
    # for row, column in indexes:
    #     value = cost_matrix[row][column]
    #     total += value
    #     print(f'({row}, {column}) -> {value}')
    # print(f'total cost: {total}')

    order_matrix_end = [0] * angles_sorted.shape[0]
    order_matrix_start = [0] * angles_sorted.shape[0]

    for row, column in indexes:
        order_matrix_end[row] = column
        order_matrix_start[column] = row

    sorted_start_state_vector = starting_state_vector[order_matrix_start]
    # sorted_end_state_vector = ending_state_vector[order_matrix_end]


    theta_values = np.unwrap(np.concatenate((sorted_start_state_vector[:, 0:1], ending_state_vector[:, 0:1]), axis=1), axis=1, period=360).T
    omega_values = np.concatenate((sorted_start_state_vector[:, 1:2], ending_state_vector[:, 1:2]), axis=1).T

    plt.figure()
    plt.plot(theta_values, omega_values, '-')
    plt.plot(theta_values[0], omega_values[0], 'o')
    plt.plot(theta_values[1], omega_values[1], 's')
    plt.show()

    try:
        with open(os.path.join(f"../Setup/Utils/AssignmentData/assignment_({number_of_planes},{satellites_per_plane_end})"), 'wb') as f:
            pickle.dump(sorted_start_state_vector, f)
            pickle.dump(order_matrix_end, f)
    except FileNotFoundError:
        with open(os.path.join(
                f"../../../Setup/Utils/AssignmentData/assignment_({number_of_planes},{satellites_per_plane_end})"),
                  'wb') as f:
            pickle.dump(sorted_start_state_vector, f)
            pickle.dump(order_matrix_end, f)

    return sorted_start_state_vector, order_matrix_end
    # print(cost_matrix)
    #
    # plane_array, _ = np.histogram(np.array(selected_indices) // satellites_per_plane_start,
    #                               bins=np.arange(number_of_planes + 1))
    # theta_array, _ = np.histogram(np.array(selected_indices) % satellites_per_plane_start,
    #                               bins=np.arange(satellites_per_plane_start + 1))
    #
    # plane_array_donors_copy = plane_array.copy()
    # donors = np.array([])
    # while len(np.arange(number_of_planes)[plane_array_donors_copy > satellites_per_plane_end]) > 0:
    #     donors = np.concatenate(
    #         (donors, np.arange(number_of_planes)[plane_array_donors_copy > satellites_per_plane_end]))
    #     plane_array_donors_copy -= (plane_array_donors_copy > satellites_per_plane_end)
    # donors = np.sort(donors)
    #
    # plane_array_receivers_copy = plane_array.copy()
    # receivers = np.array([])
    # while len(np.arange(number_of_planes)[plane_array_receivers_copy < satellites_per_plane_end]) > 0:
    #     receivers = np.concatenate(
    #         (receivers, np.arange(number_of_planes)[plane_array_receivers_copy < satellites_per_plane_end]))
    #     plane_array_receivers_copy += (plane_array_receivers_copy < satellites_per_plane_end)
    # receivers = np.sort(receivers)
    #
    # distance_matrix = cdist(donors.reshape((-1, 1)), receivers.reshape((-1, 1)))
    #
    # # Look for loop arounds
    # distance_matrix_loop = np.minimum(number_of_planes - distance_matrix, distance_matrix)
    #
    # ans_pos = hungarian_algorithm(distance_matrix_loop.copy())  # Get the element position.
    # ans, ans_mat = ans_calculation(distance_matrix_loop, ans_pos)  # Get the minimum value and corresponding matrix for visualisation
    #
    # inter_plane_movements = []
    # for donor_idx, receiver_idx in ans_pos:
    #     donor_plane = donors[donor_idx]
    #     receiver_plane = receivers[receiver_idx]
    #     sign = (receiver_plane - donor_plane) / abs(donor_plane - receiver_plane)
    #
    #     while donor_plane != receiver_plane:
    #         inter_plane_movements.append((donor_plane, donor_plane + sign))
    #         donor_plane += sign
    #
    #
    # print(plane_array)
    # print(donors)
    # print(receivers)
    # # print(distance_matrix)
    # # print(distance_matrix_loop)
    # print(ans_pos)
    # # print(ans)
    # print(ans_mat)
    # print(inter_plane_movements)
    # # print(distance_matrix_final)


def cost_function(arr_1: np.ndarray, arr_2: np.ndarray) -> float:
    """
    Compute the custom cost function between arr_1 and arr_2.

    :param arr_1: Array 1 representing current pos.
    :param arr_2: Array 2 representing target pos.
    :return: Cost value.
    """
    theta_err = np.abs(arr_2[0] - arr_1[0])
    Omega_err = np.abs(arr_2[1] - arr_1[1])

    theta_err_signed = arr_2[0] - arr_1[0]
    Omega_err_signed = arr_2[1] - arr_1[1]

    theta_err_signed += (theta_err_signed < -180) * 360 - (theta_err_signed > 180) * 360
    Omega_err_signed += (Omega_err_signed < -180) * 360 - (Omega_err_signed > 180) * 360

    sign_direction = theta_err_signed * Omega_err_signed > 0

    return np.minimum(theta_err, 360 - theta_err)**2 + np.minimum(Omega_err, 360 - Omega_err)**2 + sign_direction * theta_err_signed * Omega_err_signed
    #np.cos(np.deg2rad(45)) *


