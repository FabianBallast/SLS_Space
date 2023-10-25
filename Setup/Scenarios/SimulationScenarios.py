from enum import Enum

import numpy as np


class Simulation:
    """
    A class with important variables regarding simulations, all in seconds.
    """
    def __init__(self, start_epoch=0, simulation_duration: int | float = 60, simulation_timestep: int | float = 1):
        """
        :param start_epoch: Epoch at which simulation starts in s.
        :param simulation_duration: Duration of simulation in s.
        :param simulation_timestep: Timestep during each simulation step in s.
        """
        self.start_epoch = start_epoch
        self.simulation_duration = simulation_duration
        self.simulation_timestep = simulation_timestep

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Simulation with {self.start_epoch=}, {self.simulation_duration=}, {self.simulation_timestep}"


class SimulationScenarios(Enum):
    """
    Simple enum that provides different simulation scenarios.
    """
    sim_1_minute = Simulation(simulation_timestep=0.1)
    sim_2_minute = Simulation(simulation_duration=120)
    sim_6_minute = Simulation(simulation_duration=360)
    sim_10_minute = Simulation(simulation_duration=600)
    sim_15_minute = Simulation(simulation_duration=900)
    sim_20_minute = Simulation(simulation_duration=1200)
    sim_30_minute = Simulation(simulation_duration=1800)
    sim_45_minute = Simulation(simulation_duration=2700)
    sim_1_hour = Simulation(simulation_duration=3600)
    sim_1_5_hour = Simulation(simulation_duration=5400)
    sim_2_hour = Simulation(simulation_duration=7200)
    sim_6_hour = Simulation(simulation_duration=6 * 3600)
    sim_12_hour = Simulation(simulation_duration=12 * 3600)
    sim_24_hour = Simulation(simulation_duration=24 * 3600)
    sim_orbital_period = Simulation(simulation_duration=int(2 * np.pi / np.sqrt(100 / 55**3)))
    sim_2_orbital_period = Simulation(simulation_duration=int(4 * np.pi / np.sqrt(100 / 55 ** 3)))
    sim_5_orbital_period = Simulation(simulation_duration=int(10 * np.pi / np.sqrt(100 / 55 ** 3)))
    sim_10_orbital_period = Simulation(simulation_duration=int(20 * np.pi / np.sqrt(100 / 55 ** 3)))


if __name__ == '__main__':
    print(list(SimulationScenarios))

