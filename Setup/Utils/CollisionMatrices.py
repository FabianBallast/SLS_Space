import numpy as np
from Utils.CollisionAngles import find_collision_angles, find_collision_angle_vect
from scipy.spatial.distance import cdist
from scipy import sparse

def compute_in_plane_collision_matrix(Omega_start: list[float], longitude_list: list, order_matrix_end: list[int],
                                      satellites_per_plane_end: int, reference_angles: np.ndarray,
                                      safety_margin: float = 0) -> (np.ndarray, np.ndarray):
    """
    Compute the collision matrix for the in-plane collisions.

    :param Omega_start: The vector with starting Omega's for each satellites.
    :param longitude_list: List with the longitudes of the different planes.
    :param order_matrix_end: Positions of the satellites at the start.
    :param satellites_per_plane_end: Number of satellites per plane.
    :param reference_angles: Reference angles for each satellite in rad.
    :param safety_margin: Required safety margin for the constraint.
    :return: Matrix A and vector b such that the constraints are modelled with Ax >= b.
    """
    # Collision
    if isinstance(longitude_list, int):
        longitude_list = [longitude_list]

    sats_per_plane_start, _ = np.histogram(Omega_start, bins=len(longitude_list))
    sats_per_plane_order = np.concatenate((np.array([0]), np.cumsum(sats_per_plane_start)))
    sats_in_plane_start = [order_matrix_end[sats_per_plane_order[i]:sats_per_plane_order[i + 1]] for i in
                           range(len(longitude_list))]

    # print('Find neighbours')
    neighbour_dict = {}
    for plane in sats_in_plane_start:
        for sat_idx in range(len(plane)):

            # Check if in end state, this plane is at the end of the plane and we loop back.
            if (plane[sat_idx] + 1) % satellites_per_plane_end == 0:
                neighbour_dict[plane[sat_idx]] = plane[sat_idx] + 1 - satellites_per_plane_end
            else:
                neighbour_dict[plane[sat_idx]] = plane[sat_idx] + 1

            if sat_idx == len(plane) - 1:
                neighbour_dict[plane[sat_idx]] = list({plane[0], neighbour_dict[plane[sat_idx]]})
            else:
                neighbour_dict[plane[sat_idx]] = list({plane[sat_idx + 1], neighbour_dict[plane[sat_idx]]})

    total_number_of_constraints = 0
    for sat in neighbour_dict:
        total_number_of_constraints += len(neighbour_dict[sat])

    # print("Create constraint matrix")
    longitudes = np.kron(np.deg2rad(longitude_list), np.ones((satellites_per_plane_end)))
    constraint_matrix = np.zeros((total_number_of_constraints, len(Omega_start)))
    constraint_vector = np.zeros((total_number_of_constraints, 1))
    constraint_idx = 0
    for sat in neighbour_dict:
        for neighbour in neighbour_dict[sat]:
            constraint_matrix[constraint_idx, sat] = -1
            constraint_matrix[constraint_idx, neighbour] = 1

            theta_difference = reference_angles[sat] - reference_angles[neighbour]
            theta_difference += (theta_difference < -np.pi) * 2 * np.pi - (theta_difference > np.pi) * 2 * np.pi

            Omega_difference = longitudes[sat] - longitudes[neighbour]
            Omega_difference += (Omega_difference < -np.pi) * 2 * np.pi - (Omega_difference > np.pi) * 2 * np.pi
            constraint_vector[constraint_idx] = theta_difference + np.cos(np.deg2rad(45)) * Omega_difference
            constraint_idx += 1

    constraint_matrix = np.kron(constraint_matrix, np.array([0, 1, 0, 0, 0, 0]))
    return constraint_matrix, constraint_vector + safety_margin


def compute_out_plane_collision_matrix(theta_start: list[float], longitude_list: list, order_matrix_end: list[int],
                                       satellites_per_plane_end: int, reference_angles: np.ndarray,
                                       satellites_per_plane_start: int, Omega_start: list[float],
                                       safety_margin: float = 0, Omega_end: list = None) -> (np.ndarray, np.ndarray):
    """
    Compute the collision matrix for the in-plane collisions.

    :param theta_start: The vector with starting thetas for each satellites.
    :param longitude_list: List with the longitudes of the different planes.
    :param order_matrix_end: Positions of the satellites at the start.
    :param satellites_per_plane_end: Number of satellites per plane at end.
    :param reference_angles: Reference angles for each satellite in rad.
    :param satellites_per_plane_start: Number of satellites per plane at the start.
    :param Omega_start: The vector with starting Omega's for each satellites.
    :param safety_margin: Required safety margin for the constraint.
    :return: Matrix A and vectors b1/b2/b3/b4 such that the constraints are modelled with |Ax - b1| >= b2,
            and where b3=theta_ref_diff and b4=Omega_ref_diff
    """
    # Find collision angles
    if isinstance(longitude_list, int):
        longitude_list = [longitude_list]

    planes = np.deg2rad(longitude_list)
    difference_array = np.zeros((len(planes) - 1, 1))

    for idx, plane in enumerate(planes[1:]):
        theta_1, theta_2 = find_collision_angles(np.pi / 4, 0, plane)
        difference_array[idx] = np.rad2deg(theta_2 - theta_1) % 360

    difference_array_start = np.concatenate((np.array([0]), difference_array.flatten()))

    # Find plane differences at start and end
    # sats_per_plane_start, bins = np.histogram(Omega_start, bins=len(longitude_list) + 1)
    sats_in_plane_start = np.digitize(Omega_start, longitude_list).reshape((-1, 1)) - 1

    if Omega_end is None:
        sats_in_plane_end = np.kron(np.arange(len(planes)), np.ones(satellites_per_plane_end)).reshape((-1, 1))
    else:
        sats_in_plane_end = np.array(Omega_end).reshape((-1, 1)) // 23

    if len(longitude_list) > 10:
        plane_distance_start = cdist(sats_in_plane_start, sats_in_plane_start, difference).reshape((-1, 1)) % len(longitude_list)
        plane_distance_end = cdist(sats_in_plane_end, sats_in_plane_end, difference).reshape((-1, 1)) % len(longitude_list)
        plane_dummy = np.zeros((plane_distance_start.shape[0], len(longitude_list)), dtype=bool)
    else:
        difference_array_shadow = np.zeros_like(difference_array)
        for idx, plane in enumerate(planes[1:]):
            theta_1, theta_2 = find_collision_angles(np.pi / 4, plane, 0)
            difference_array_shadow[idx] = np.rad2deg(theta_2 - theta_1) % 360

        plane_distance_start = cdist(sats_in_plane_start, sats_in_plane_start, difference).reshape((-1, 1)) % (2 * len(longitude_list) - 1)
        plane_distance_end = cdist(sats_in_plane_end, sats_in_plane_end, difference).reshape((-1, 1)) % (2 * len(longitude_list) - 1)
        plane_dummy = np.zeros((plane_distance_start.shape[0], 2 * len(longitude_list) - 1), dtype=bool)


    plane_dummy[:, 0] = True
    plane_selection = sparse.coo_matrix(plane_dummy)
    plane_selection.col = np.array(plane_distance_start + 1e-5, dtype=int).flatten()

    plane_selection_start = np.array(plane_selection.todense())[:, 1:]

    plane_selection.col = np.array(plane_distance_end + 1e-5, dtype=int).flatten()
    plane_selection_end = np.array(plane_selection.todense())[:, 1:]

    plane_selection_total = plane_selection_start | plane_selection_end

    # Find satellites that will be close to other satellites limit angles.
    distances_start = cdist(np.array(theta_start).reshape((-1, 1)), np.array(theta_start).reshape((-1, 1)), difference).reshape((-1, 1)) % 360
    distances_end = cdist(np.rad2deg(reference_angles).reshape((-1, 1)), np.rad2deg(reference_angles).reshape((-1, 1)), difference).reshape((-1, 1)) % 360

    if len(longitude_list) <= 10:
        difference_array = np.concatenate((difference_array, difference_array_shadow))

    collision_distance_start = cdist(distances_start, difference_array, difference) % 360
    collision_distance_end = cdist(distances_end, difference_array, difference) % 360

    collision_distance_start -= (collision_distance_start > 180) * 360
    collision_distance_end -= (collision_distance_end > 180) * 360

    collisions_start = (collision_distance_start > -safety_margin) & (collision_distance_start < safety_margin)
    collisions_end = (collision_distance_end > -safety_margin) & (collision_distance_end < safety_margin)
    collisions_halfway = ((collision_distance_start < -safety_margin) & (collision_distance_end > safety_margin)) | ((collision_distance_start > safety_margin) & (collision_distance_end < -safety_margin))
    collisions_halfway &= np.abs(collision_distance_start - collision_distance_end) < 180

    collisions = (collisions_start | collisions_end | collisions_halfway) & plane_selection_total
    # collisions -= np.eye(collisions.shape[0], dtype=bool)

    collisions = collisions.reshape((len(theta_start), len(theta_start), -1))

    number_of_constraints = np.sum(np.any(collisions, axis=2)) // 2  # Every collision is present twice

    plane_distance_start = np.array(plane_distance_start.reshape((len(Omega_start), len(Omega_start), -1)), dtype=int)
    collision_matrix = np.zeros((number_of_constraints, len(theta_start)))
    collision_vector = np.zeros((number_of_constraints, ))
    collision_counter = 0
    # print(difference_array_start, difference_array_start.shape)

    for i in range(collisions.shape[0]):
        for j in range(i+1, collisions.shape[1]):
            if np.any(collisions[i, j]):
                collision_matrix[collision_counter, i] = -1
                collision_matrix[collision_counter, j] = 1
                collision_vector[collision_counter] = difference_array_start[plane_distance_start[i, j]]
                collision_counter += 1

    theta_ref_diff = (collision_matrix @ reference_angles).flatten() % (2 * np.pi)
    theta_ref_diff -= (theta_ref_diff > np.pi) * 2 * np.pi

    if Omega_end is None:
        reference_planes = np.kron(np.deg2rad(longitude_list), np.ones(satellites_per_plane_end)).reshape((-1, 1))
    else:
        reference_planes = np.deg2rad(Omega_end).reshape((-1, 1))
    Omega_ref_diff = (collision_matrix @ reference_planes).flatten() % (2 * np.pi)
    Omega_ref_diff -= (Omega_ref_diff > np.pi) * 2 * np.pi

    collision_vector -=  (collision_vector > np.pi) * 360

    return collision_matrix, np.deg2rad(collision_vector), np.ones_like(collision_vector) * safety_margin, theta_ref_diff, Omega_ref_diff


def update_collision_vector(collision_vector_old: np.ndarray, Omega_differences_new: np.ndarray,
                            Omega_differences_old: np.ndarray, Omega_reff_diff: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Find the new collision vector for out-of-plane collision avoidance.

    :param collision_vector_old: The old collision vector.
    :param Omega_differences_new: The new Omegas in rad.
    :param Omega_differences_old: The old Omegas in rad.
    :param Omega_reff_diff: Differences between reference Omegas for all collisions.
    :return: New collision vector + Omegas for which it is calculated.
    """
    # print(collision_vector_old.shape, Omega_differences_new.shape, Omega_differences_old.shape)
    # print(Omega_differences_new.reshape((20, -1)))
    # print(Omega_differences_old.reshape((20, -1)))
    changed_differences = np.abs(Omega_differences_new - Omega_differences_old) > np.deg2rad(0.01)
    # changed_indices = np.arange(changed_differences.shape[0])[changed_differences.flatten()]
    # print(np.rad2deg(Omega_new), np.rad2deg(Omega_old))
    # print(np.rad2deg(collision_vector_old))
    # print(np.rad2de)
    # print(np.rad2deg(Omega_differences_new[changed_differences] - Omega_differences_old[changed_differences]))
    collision_vector_new = collision_vector_old.copy()
    # print(np.sum(changed_differences), collision_matrix.shape[0])
    collision_vector_new[changed_differences.flatten()] = find_collision_angle_vect(np.pi / 4, Omega_differences_new[changed_differences] + Omega_reff_diff[changed_differences.flatten()]) % (2 * np.pi)
    # for i, Omega_difference in enumerate(Omega_differences_new[changed_differences]):
    #     # print(Omega_difference, Omega_reff_diff[idx])
    #     idx = changed_indices[i]
    #     theta_1, theta_2 = find_collision_angles(np.pi / 4, 0, Omega_difference + Omega_reff_diff[idx])
    #     collision_vector_new[idx] = (theta_2 - theta_1) % (2 * np.pi)

    collision_vector_new -= (collision_vector_new > np.pi) * 2 * np.pi
        # print((theta_2 - theta_1), collision_vector_new[idx])



    # print(np.sum(changed_differences))
    # print(changed_indices)
    # print(np.rad2deg(collision_vector_new.reshape((20, -1))))
    # print(np.rad2deg(collision_vector_old.reshape((20, -1))))
    # print()
    Omega_difference_update = Omega_differences_old.copy()
    Omega_difference_update[changed_differences] = Omega_differences_new[changed_differences]
    return collision_vector_new, Omega_difference_update


def difference(x_1, x_2) -> float:
    return x_2[0] - x_1[0]