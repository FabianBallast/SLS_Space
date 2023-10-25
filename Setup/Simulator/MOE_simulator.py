from Simulator.RK4 import RK4_integration
import numpy as np
from Scenarios.MainScenarios import Scenario


def mean_orbital_elements_simulator(x0: np.ndarray, inputs: np.ndarray, scenario: Scenario, simulation_length: int,
                                    disturbance_limit: np.ndarray = None):
    """
    Simulate the dynamics of the system using RK4

    :param x0: The starting state in shape (6 * number_of_satellites, )
    :param inputs: The control inputs in shape (t, 3 * number_of_satellites) in N.
    :param scenario: Scenario that is running.
    :param simulation_length: Length of the simulation.
    :param disturbance_limit: Maximum value of the allowed disturbance.
    :return: States over time in shape (t, 6 * number_of_satellites).
    """
    # Convert mean orbital elements to state that is used for simulation
    a = x0[0::6]
    e = x0[1::6]
    i = x0[2::6]
    omega = x0[3::6]
    Omega = x0[4::6]
    f = x0[5::6]

    r = a * (1 - e**2) / (1 + e * np.cos(f))
    theta = f + omega
    ex = e * np.cos(f)
    ey = e * np.sin(f)

    sim_state = np.concatenate((r, theta, ex, ey, i, Omega)).reshape((6, -1)).T.flatten()

    # Simulate
    states = RK4_integration(sim_state, inputs, scenario, simulation_length, disturbance_limit)

    # Convert back
    r = states[:, 0::6]
    theta = states[:, 1::6]
    ex = states[:, 2::6]
    ey = states[:, 3::6]
    i = states[:, 4::6]
    Omega = states[:, 5::6]

    e = np.sqrt(ex ** 2 + ey ** 2)
    f = np.zeros_like(e)
    f[e > 0] = np.arctan2(ey[e > 0], ex[e > 0])
    eta = np.sqrt(1 - e ** 2)
    kappa = 1 + e * np.cos(f)
    a = r * kappa / eta ** 2
    omega = theta - f

    mean_states = np.concatenate((a, e, i, omega, Omega, f)).reshape((6, -1)).T.reshape((a.shape[0], -1))
    return mean_states




if __name__ == '__main__':
    from Scenarios.MainScenarios import ScenarioEnum

    number_of_satellites = 10
    x0_test = np.array([55, 0, 0.7, 0.1, 0.1, 0.3] * number_of_satellites)
    inputs = np.zeros((3600, 3 * number_of_satellites))

    res = mean_orbital_elements_simulator(x0_test, inputs, ScenarioEnum.simple_scenario_translation_ROE_scaled.value, 10)
