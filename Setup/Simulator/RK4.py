import numpy as np
from Scenarios.MainScenarios import Scenario


def RK4_integration(x0: np.ndarray, inputs: np.ndarray, scenario: Scenario):
    """
    Simulate the dynamics of the system using RK4

    :param x0: The starting state in shape (6 * number_of_satellites, )
    :param inputs: The control inputs in shape (t, 3 * number_of_satellites) in N.
    :param scenario: The scenario that is running.
    :return: States over time in shape (t, 6 * number_of_satellites).
    """
    timestep = scenario.simulation.simulation_timestep
    simulation_length = scenario.simulation.simulation_duration

    number_of_iterations = int((simulation_length + 0.001) / timestep)
    x = np.zeros((number_of_iterations + 1, x0.shape[0]))
    x[0] = x0

    for t in range(number_of_iterations):
        k1 = find_xdot(x[t], inputs[t], scenario)
        k2 = find_xdot(x[t] + 0.5 * timestep * k1, inputs[t], scenario)
        k3 = find_xdot(x[t] + 0.5 * timestep * k2, inputs[t], scenario)
        k4 = find_xdot(x[t] + timestep * k3, inputs[t], scenario)

        x[t + 1] = x[t] + timestep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return x


def find_xdot(x0: np.ndarray, input: np.ndarray, scenario: Scenario) -> np.ndarray:
    """
    Find the derivative of the state given its current state and the control inputs.

    :param x0: Current state in shape (6 * number_of_satellites, )
    :param input: Current inputs in shape (3 * number_of_satellites in N.
    :param scenario: The scenario that is running.
    :return: Derivative of x0 in shape (6 * number_of_satellites, )
    """
    grav_param = scenario.physics.gravitational_parameter_Earth

    r = x0[0::6]
    theta = x0[1::6]
    ex = x0[2::6]
    ey = x0[3::6]
    i = x0[4::6]
    # Omega = x0[5::6]

    u_r = input[0::3] / scenario.physics.mass
    u_t = input[1::3] / scenario.physics.mass
    u_n = input[2::3] / scenario.physics.mass

    e = np.sqrt(ex**2 + ey**2)

    f = np.zeros_like(e)
    f[e > 0] = np.arctan2(ey[e > 0], ex[e > 0])

    eta = np.sqrt(1 - e**2)
    kappa = 1 + e * np.cos(f)
    a = r * kappa / eta**2
    n = np.sqrt(grav_param / a**3)

    J2_value = scenario.physics.J2_value
    J2_scaling_factor = 0.75 * J2_value * (scenario.physics.radius_Earth / (a * eta**2)) ** 2 * n

    if scenario.physics.J2_perturbation:
       Omega_dot_j2 = -2 * J2_scaling_factor * np.cos(i)
       omega_dot_j2 = J2_scaling_factor * (5 * np.cos(i) ** 2 - 1)
       f_dot_j2 = J2_scaling_factor * eta * (3 * np.cos(i) ** 2 - 1)
    else:
        f_dot_j2 = 0
        omega_dot_j2 = 0
        Omega_dot_j2 = 0

    r_dot = n * a / eta * ey + e * np.sin(f) * r / kappa * f_dot_j2
    theta_dot = np.sqrt(grav_param * kappa / r**3) - eta * np.sin(theta) * u_n / (a * n * kappa * np.tan(i)) + omega_dot_j2 + f_dot_j2
    ex_dot = 2 * eta * u_t / (a * n) - n * kappa**2 * ey / eta**3 - f_dot_j2 * e * np.sin(f)
    ey_dot = eta * u_r / (a * n) + eta * ey * u_t / (a * n * kappa) + n * a**2 * eta * ex / r**2 + f_dot_j2 * e * np.cos(f)
    i_dot = eta * np.cos(theta) * u_n / (a * n * kappa)
    Omega_dot = eta * np.sin(theta) * u_n / (a * n * kappa * np.sin(i)) + Omega_dot_j2

    return np.concatenate((r_dot, theta_dot, ex_dot, ey_dot, i_dot, Omega_dot)).reshape((6, -1)).T.flatten()
