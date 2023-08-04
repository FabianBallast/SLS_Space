import random
import Dynamics.SystemDynamics as SysDyn
from Dynamics.DifferentialDragDynamics import DifferentialDragDynamics
import numpy as np


def generate_anomalies_and_longitudes(number_of_dropouts: int, longitude_list: list[float], number_of_systems: int,
                                      dynamics: SysDyn.GeneralDynamics) -> tuple[list[float], list[float]]:
    """
    Generate the initial true anomalies and longitudes for the satellites.

    :param number_of_dropouts: Number of dropouts.
    :param longitude_list: List with the longitudinal angles to divide the satellites into.
    :param number_of_systems: Number of (controlled) satellites.
    :param dynamics: The dynamics used for the controller.

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

    if number_of_planes == 1:
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

        return angles_sorted[:, 0].tolist(), longitude_list * number_of_systems

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
        return angles_sorted[:, 0].tolist(), longitudes_selected.tolist()


def generate_reference(number_of_systems: int, anomaly_list: list[float],
                       longitude_list: list[float]):
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

    satellites_per_plane = number_of_systems // number_of_planes
    reference = np.tile(np.linspace(0, 2 * np.pi, satellites_per_plane, endpoint=False), number_of_planes)

    anomalies_rad = np.deg2rad(anomaly_list)

    reference[reference - anomalies_rad > np.pi] -= 2 * np.pi
    reference[reference - anomalies_rad < -np.pi] += 2 * np.pi

    reference -= np.mean(reference - anomalies_rad)

    return reference.reshape((-1, 1))
