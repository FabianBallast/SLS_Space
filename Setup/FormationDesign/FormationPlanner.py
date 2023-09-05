import numpy as np
import matplotlib.pyplot as plt
import random


def plan_formation(satellites_per_plane: int, number_of_planes: int,
                   theta_spacing: int = 5) -> (np.ndarray, np.ndarray):
    """
    Plan a formation given a number of satellites per plane and the number of planes.

    :param satellites_per_plane: The number of satellites per plane.
    :param number_of_planes: Number of planes.
    :param theta_spacing: Spacing between theta values of each plane in degrees.
    :return: Two arrays with the respective thetas and Omegas in rad in the range (0, 2 * np.pi)
    """
    Omega = np.kron(np.linspace(0, 2 * np.pi, number_of_planes, endpoint=False), np.ones((satellites_per_plane,)))
    theta = np.zeros_like(Omega)

    for i in range(number_of_planes):
        thetas_raw = np.linspace(0, 2 * np.pi, satellites_per_plane, endpoint=False) + np.deg2rad(theta_spacing) * i
        theta[i * satellites_per_plane: (i + 1) * satellites_per_plane] = thetas_raw % (2 * np.pi)

    return theta, Omega


def formation_after_dropouts(theta: np.ndarray, Omega: np.ndarray, dropouts: int) -> (np.ndarray, np.ndarray):
    """
    Find the formation after a given number of dropouts.

    :param theta: The theta values in rad to plot.
    :param Omega: The Omega values in rad to plot.
    :param dropouts: Number of dropouts.
    :return: Formation after dropouts.
    """
    total_number_of_satellites = len(theta)

    random.seed(129)
    survivors = random.sample(range(total_number_of_satellites), total_number_of_satellites - dropouts)

    return theta[survivors], Omega[survivors]

def plot_formation(theta: np.ndarray, Omega: np.ndarray, figure: plt.figure = None,
                   dropouts: int = 0) -> plt.figure:
    """
    Give a visualisation of a formation in 2D.

    :param theta: The theta values in rad to plot.
    :param Omega: The Omega values in rad to plot.
    :param figure: Figure to plot the results into.
    :param dropouts: Number of dropouts
    :return: Two arrays with the respective thetas and Omegas in rad.
    """
    if figure is None:
        figure = plt.figure()

    plt.scatter(np.rad2deg(theta), np.rad2deg(Omega))
    plt.xlabel("Theta [deg]")
    plt.ylabel("Omega [deg]")

    return figure


if __name__ == '__main__':
    plot_formation(*plan_formation(5, 2))
    fig = plot_formation(*formation_after_dropouts(*plan_formation(6, 6), 6))
    plt.show()
