import numpy as np
from Simulator.MOE_simulator import mean_orbital_elements_simulator
from Scenarios.MainScenarios import Scenario


class DynamicsSimulator:
    """
    A dummy Dynamics Simulator class.
    """
    def __init__(self, x0, inputs: np.ndarray, scenario: Scenario, simulation_length: int,
                 disturbance_limit: np.ndarray = None):
        """
        Custom equivalent of the same function of Tudatpy.

        :param x0: Initial state
        :param inputs: The control inputs over time in the shape (t, 3 * number_of_satellites).
        :param scenario: The scenario that is running.
        :param simulation_length: Length of the simulation.
        :param disturbance_limit: The maximum disturbance in simulation parameters.
        """
        self.x0 = x0
        self.inputs = inputs.reshape((-1, inputs.shape[2])).T
        self.scenario = scenario
        self.simulation_length = simulation_length
        self.state_history = None
        self.dependent_variable_history = None
        self.disturbance_limit = disturbance_limit

    def run(self):
        oe_states = mean_orbital_elements_simulator(x0=self.x0, inputs=self.inputs,
                                                    scenario=self.scenario, simulation_length=self.simulation_length,
                                                    disturbance_limit=self.disturbance_limit)
        self.state_history = {index: oe_states[index, :] for index in range(oe_states.shape[0])}
        self.dependent_variable_history = {index: oe_states[index, :] for index in range(oe_states.shape[0])}


def create_dynamics_simulator(x0: np.ndarray, inputs: np.ndarray, scenario: Scenario,
                              simulation_length: int, disturbance_limit: np.ndarray = None) -> DynamicsSimulator:
    """
    Custom equivalent of the same function of Tudatpy.

    :param x0: Initial state in MOE.
    :param inputs: The control inputs over time in the shape (t, 3 * number_of_satellites).
    :param scenario: The scenario that is running.
    :param simulation_length: Length of the simulation.
    :param disturbance_limit: The maximum disturbance in simulation parameters.
    :return: Simplified dynamics simulator.
    """
    simulator = DynamicsSimulator(x0, inputs, scenario, simulation_length, disturbance_limit)
    simulator.run()
    return simulator
