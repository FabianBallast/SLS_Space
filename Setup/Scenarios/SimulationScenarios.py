from enum import Enum


class Simulation:
    """
    A class with important variables regarding simulations, all in seconds.
    """
    def __init__(self, start_epoch=0, simulation_duration: int | float = 60, simulation_timestep: int | float = 1):
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
    sim_1_minute = Simulation()
    sim_10_minute = Simulation(simulation_duration=600)
    sim_30_minute = Simulation(simulation_duration=1800)
    sim_1_hour = Simulation(simulation_duration=3600)
    sim_6_hour = Simulation(simulation_duration=6 * 3600)
    sim_12_hour = Simulation(simulation_duration=12 * 3600)
    sim_24_hour = Simulation(simulation_duration=24 * 3600)


if __name__ == '__main__':
    print(list(SimulationScenarios))

