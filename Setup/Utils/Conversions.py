import numpy as np
from matplotlib import pyplot as plt
from tudatpy.kernel.astro import element_conversion


def oe2cartesian(oe: np.ndarray, mu: float | int) -> np.ndarray:
    """
    Transform a set of orbital elements to their cartesian equivalents.

    :param oe: Array with orbital elements in the shape (t, 6 * number of satellites).
    :param mu: Gravitational parameter of the Earth.
    :return: Array with Cartesian coordinates in the shape (t, 6 * number of satellites).
    """
    time = oe.shape[0]
    number_of_satellites = oe.shape[1] // 6

    cartesian_coordinates = np.zeros_like(oe)

    for t in range(time):
        for satellite_idx in range(number_of_satellites):
            cartesian_coordinates[t, satellite_idx * 6:
                                     (satellite_idx+1) * 6] = element_conversion.keplerian_to_cartesian(oe[t, satellite_idx * 6:
                                                                                                              (satellite_idx + 1) * 6], mu)

    return cartesian_coordinates


def extract_rotation_matrices(oe: np.ndarray) -> np.ndarray:
    """
    Extract the rotation matrices from inertial to planar frame from the orbital elements.

    :param oe: The orbital elements in the shape (t, 6).
    :return: Array with rotation matrices in the shape (t, 3, 3).
    """

    time = oe.shape[0]

    rotation_matrices = np.zeros((time, 3, 3))
    for t in range(time):
        i = oe[t, 2]
        omega = oe[t, 3]
        Omega = oe[t, 4]
        f = oe[t, 5]

        R_Omega = np.array([[np.cos(Omega), np.sin(Omega), 0],
                            [-np.sin(Omega), np.cos(Omega), 0],
                            [0, 0, 1]])

        R_omega = np.array([[np.cos(omega + f), np.sin(omega + f), 0],
                            [-np.sin(omega + f), np.cos(omega + f), 0],
                            [0, 0, 1]])

        R_i = np.array([[1, 0, 0],
                        [0, np.cos(i), np.sin(i)],
                        [0, -np.sin(i), np.cos(i)]])

        rotation_matrices[t] = R_omega @ R_i @ R_Omega

    return rotation_matrices


def oe2cylindrical(oe: np.ndarray, mu: float | int, reference_oe: np.ndarray) -> np.ndarray:
    """
    Transform a set of orbital elements to cylindrical coordinates.

    :param oe: Array with orbital elements in the shape (t, 6 * number of satellites).
    :param mu: Gravitational parameter of the Earth.
    :param reference_oe: Array with elements of the reference (number_of_references, t, 6).
    :return: Array with cylindrical coordinates in the shape (t, 6 * number of satellites).
    """
    time_length = oe.shape[0]
    number_of_satellites = oe.shape[1] // 6
    number_of_ref = reference_oe.shape[0]
    satellites_per_ref = number_of_satellites // number_of_ref

    # Find correct coordinates
    states_rsw = np.zeros((time_length, 6 * satellites_per_ref))
    cylindrical_states = np.zeros_like(oe)

    for ref in range(number_of_ref):
        inertial_coordinates = oe2cartesian(oe[:, ref * satellites_per_ref * 6:(ref+1) * satellites_per_ref * 6], mu)
        rotation_matrices_ref = extract_rotation_matrices(reference_oe[ref])

        for i in range(time_length):
            inertial_to_rsw_kron = np.kron(np.eye(2 * satellites_per_ref), rotation_matrices_ref[i])
            states_rsw[i, :] = inertial_to_rsw_kron @ inertial_coordinates[i]

        for satellite in range(satellites_per_ref):
            states_satellite_rsw = states_rsw[:, satellite * 6:(satellite + 1) * 6]
            rho = np.sqrt(states_satellite_rsw[:, 0:1] ** 2 + states_satellite_rsw[:, 1:2] ** 2) - reference_oe[ref, 0, 0]
            theta = np.unwrap(np.arctan2(states_satellite_rsw[:, 1:2], states_satellite_rsw[:, 0:1]), axis=0)
            z = states_satellite_rsw[:, 2:3]

            # Estimate gradient
            edge_order = 2 if time_length > 2 else 1
            rho_dot = np.gradient(rho, axis=0, edge_order=edge_order)
            theta_dot = np.gradient(theta, axis=0, edge_order=edge_order)
            z_dot = np.gradient(z, axis=0, edge_order=edge_order)
            # z_dot = states_satellite_rsw[:, 5:6]

            # Convert to range [0, 2 * pi]
            if theta[0] < 0:
                theta += 2 * np.pi

            cylindrical_states[:, (ref * satellites_per_ref + satellite) * 6:(ref * satellites_per_ref + satellite + 1) * 6] = np.concatenate(
                (rho, theta, z, rho_dot, theta_dot,
                 z_dot), axis=1)

    return cylindrical_states


def oe2blend(oe: np.ndarray, reference_oe: np.ndarray) -> np.ndarray:
    """
    Transform a set of orbital elements to blend coordinates.

    :param oe: Array with orbital elements in the shape (t, 6 * number of satellites).
    :param mu: Gravitational parameter of the Earth.
    :param reference_oe: Array with elements of the reference (number_of_references, t, 6).
    :return: Array with blend coordinates in the shape (t, 6 * number of satellites).
    """
    # Basic variables
    time_length = oe.shape[0]
    number_of_satellites = oe.shape[1] // 6
    number_of_ref = reference_oe.shape[0]
    satellites_per_ref = number_of_satellites // number_of_ref

    # Reference variables
    rho_ref = reference_oe[:, :, 0::6] * (1 - reference_oe[:, :, 1::6] ** 2) / (1 + reference_oe[:, :, 1::6] * np.cos(reference_oe[:, :, 5::6]))
    theta_ref = np.unwrap(reference_oe[:, :, 3::6] + reference_oe[:, :, 5::6], axis=0)
    inclination_ref = reference_oe[:, :, 2::6]
    Omega_ref = reference_oe[:, :, 4::6]

    blend_states = np.zeros_like(oe)

    for ref in range(number_of_ref):
        for idx in range(satellites_per_ref):
            oe_sat = oe[:, (ref * satellites_per_ref + idx) * 6: (ref * satellites_per_ref + idx + 1) * 6]

            rho = oe_sat[:, 0:1] * (1 - oe_sat[:, 1:2] ** 2) / (1 + oe_sat[:, 1:2] * np.cos(oe_sat[:, 5:6])) - rho_ref[ref]
            theta = np.unwrap(oe_sat[:, 3:4] + oe_sat[:, 5:6] - theta_ref[ref], axis=0)

            edge_order = 2 if time_length > 2 else 1
            rho_dot = np.gradient(rho, axis=0, edge_order=edge_order)
            theta_dot = np.gradient(theta, axis=0, edge_order=edge_order)

            rho = rho / rho_ref[ref]
            delta_i = oe_sat[:, 2:3] - inclination_ref[ref]
            delta_Omega = oe_sat[:, 4:5] - Omega_ref[ref]

            # if oe_sat.shape[0] > 2:
            #     plt.figure()
            #
            #     a = oe_sat[:, 0:1]
            #     e = oe_sat[:, 1:2]
            #     theta_t = oe_sat[:, 5:6]
            #
            #     eta = 1 - e**2
            #     kappa = 1 + e * np.cos(theta_t)
            #
            #     a_dot = np.gradient(a, axis=0, edge_order=edge_order)
            #     e_dot = np.gradient(e, axis=0, edge_order=edge_order)
            #     theta_dot_t = np.gradient(theta_t, axis=0, edge_order=edge_order)
            #     rho_dot_test = a_dot * eta / kappa - 2 * e * e_dot * a / kappa - e_dot * np.cos(theta_t) * a * eta / kappa**2 + theta_dot_t * np.sin(theta_t) * eta * e / kappa**2
            #     plt.plot(rho_dot)
            #     plt.plot(rho_dot_test, '--')
            #     plt.plot(a_dot * eta / kappa- e_dot * np.cos(theta_t) * a * eta / kappa**2, '.-')
            #     plt.plot(a_dot - e_dot * np.cos(theta_t) * a, 'o-')
            #     plt.show()
            # delta_lambda = kepler_sat[:, 3:4] + kepler_sat[:, 5:6] - kepler_ref[:, 3:4] - kepler_ref[:, 5:6]
            #
            # delta_r = (np.sqrt(states_satellite[:, 0:1] ** 2 + states_satellite[:, 1:2] ** 2) - ref_rho) / ref_rho
            # delta_theta = np.unwrap(np.arctan2(states_satellite[:, 1:2], states_satellite[:, 0:1]), axis=0) - ref_theta
            # rho_dot = np.cos(delta_theta) * states_satellite[:, 3:4] + np.sin(delta_theta) * states_satellite[:,
            #                                                                                  4:5] - ref_rho_dot
            # theta_dot = (-np.sin(delta_theta) * states_satellite[:, 3:4] +
            #              np.cos(delta_theta) * states_satellite[:, 4:5]) / (delta_r * ref_rho + ref_rho) - ref_theta_dot
            # delta_i = kepler_sat[:, 2:3] - kepler_ref[:, 2:3]
            # delta_Omega = kepler_sat[:, 4:5] - kepler_ref[:, 4:5]

            # Convert to range [0, 2 * pi]
            theta[0] %= 2 * np.pi
            # if delta_lambda[0] > np.pi:
            #     delta_lambda -= 2 * np.pi

            blend_states[:, (ref * satellites_per_ref + idx) * 6: (ref * satellites_per_ref + idx + 1) * 6] = \
                np.concatenate((rho, rho_dot, theta, theta_dot, delta_i, delta_Omega), axis=1)

    return blend_states
