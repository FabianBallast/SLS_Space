import numpy as np
# from sympy import *
from scipy.spatial.distance import cdist

# inclination = symbols('i')
# Omega_1 = symbols('Omega_1')
# Omega_2 = symbols('Omega_2')
#
# R_inc = np.array([[1, 0, 0],
#                   [0, cos(inclination), -sin(inclination)],
#                   [0, sin(inclination), cos(inclination)]])
#
# R_Omega_1 = np.array([[cos(Omega_1), -sin(Omega_1), 0],
#                       [sin(Omega_1), cos(Omega_1), 0],
#                       [0, 0, 1]])
#
# R_Omega_2 = np.array([[cos(Omega_2), -sin(Omega_2), 0],
#                       [sin(Omega_2), cos(Omega_2), 0],
#                       [0, 0, 1]])
#
# n_1 = R_Omega_1 @ R_inc @ np.array([0, 0, 1]).reshape((-1, 1))
# n_2 = R_Omega_2 @ R_inc @ np.array([0, 0, 1]).reshape((-1, 1))
# print(simplify(n_1), simplify(n_2))
# #
# # Find collision line
# n_coll = np.cross(n_2.flatten(), n_1.flatten()).reshape((-1, 1))
# print(simplify(n_coll))
# #
# # Project to orbits
# n_coll_orbit_1 = (R_Omega_1 @ R_inc).T @ n_coll
# n_coll_orbit_2 = (R_Omega_2 @ R_inc).T @ n_coll
# #
# # theta_1 = atan2(n_coll_orbit_1[1], n_coll_orbit_1[0])
# # theta_2 = atan2(n_coll_orbit_2[1], n_coll_orbit_2[0])
#
# print(simplify(n_coll_orbit_1))
# print(simplify(n_coll_orbit_2))
def find_collision_angles(inclination: float, Omega_1: float, Omega_2: float) -> (float, float):
    """
    Given two planes with provided inclination and Omega_1 and Omega_2, find theta_1 and theta_2 where they intersect.

    :param inclination: Inclination of the planes in rad.
    :param Omega_1: RAAN of plane 1 in rad.
    :param Omega_2: RAAN of plane 2 in rad.
    :return: Tuple with (theta_1, theta_2) in rad.
    """
    # Find normal vectors.
    R_inc = np.array([[1, 0, 0],
                      [0, np.cos(inclination), -np.sin(inclination)],
                      [0, np.sin(inclination), np.cos(inclination)]])

    R_Omega_1 = np.array([[np.cos(Omega_1), -np.sin(Omega_1), 0],
                          [np.sin(Omega_1), np.cos(Omega_1), 0],
                          [0, 0, 1]])

    R_Omega_2 = np.array([[np.cos(Omega_2), -np.sin(Omega_2), 0],
                          [np.sin(Omega_2), np.cos(Omega_2), 0],
                          [0, 0, 1]])

    n_1 = R_Omega_1 @ R_inc @ np.array([0, 0, 1]).reshape((-1, 1))
    n_2 = R_Omega_2 @ R_inc @ np.array([0, 0, 1]).reshape((-1, 1))
    # print(n_1, n_2)

    # Find collision line
    n_coll = np.cross(n_1.flatten(), n_2.flatten()).reshape((-1, 1))

    # Project to orbits
    n_coll_orbit_1 = (R_Omega_1 @ R_inc).T @ n_coll
    n_coll_orbit_2 = (R_Omega_2 @ R_inc).T @ n_coll

    # Find angles
    theta_1 = np.arctan2(n_coll_orbit_1[1], n_coll_orbit_1[0]) % (2 * np.pi)
    theta_2 = np.arctan2(n_coll_orbit_2[1], n_coll_orbit_2[0]) % (2 * np.pi)

    return theta_1[0], theta_2[0]


def find_collision_angle_vect(inclination: float, delta_Omega: np.ndarray) -> np.ndarray:
    """
    Find the collision angles for various delta Omegas.

    :param inclination: Inclination for all scenarios.
    :param delta_Omega: Array with delta Omegas in Omega_2 - Omega_1.
    :return: Tuple with delta_collision_angle in theta_2 - theta_1.
    """
    theta_1 = np.arctan2(np.sin(-delta_Omega), np.cos(inclination) * (1 - np.cos(delta_Omega)))
    theta_2 = np.arctan2(np.sin(-delta_Omega), np.cos(inclination) * (np.cos(delta_Omega) - 1))

    return theta_2 - theta_1


if __name__ == '__main__':
    number_of_planes = np.arange(14, 31)
    number_of_satellites = np.arange(14, 31)
    min_dist = np.zeros((len(number_of_planes), len(number_of_satellites)))
    for i, num_planes in enumerate(number_of_planes):
        for j, num_sats in enumerate(number_of_satellites):

            planes = np.linspace(0, 2 * np.pi, num=num_planes, endpoint=False)
            difference_array = np.zeros(len(planes) - 1)

            for idx, plane in enumerate(planes[1:]):
                theta_1, theta_2 = find_collision_angles(np.pi/4, 0, plane)
                difference_array[idx] = np.rad2deg(theta_1 - theta_2) % 360
        # print(np.rad2deg(find_collision_angles(np.pi / 4, 0, np.pi / 2)))
        # print(np.rad2deg(find_collision_angles(np.pi / 4, 0.1, np.pi / 2 + 0.1)))

            dist = cdist(difference_array.reshape((-1, 1)),
                         np.linspace(0, 360, num=num_sats, endpoint=False).reshape((-1, 1)),
                         'cityblock')
            min_dist[i, j] = np.min(dist)

    # print(min_dist)
    # print(np.max(min_dist))

    planes = np.linspace(0, 2 * np.pi, num=15, endpoint=False)
    difference_array = np.zeros(len(planes) - 1)

    for idx, plane in enumerate(planes[1:]):
        theta_1, theta_2 = find_collision_angles(np.pi / 4, 0, plane)
        difference_array[idx] = np.rad2deg(theta_1 - theta_2) % 360

    theta_arr = np.linspace(0, 360, num=15, endpoint=False)

    # print(difference_array)
    # print(theta_arr)
