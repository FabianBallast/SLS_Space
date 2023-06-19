from typing import Callable
import control as ct
import numpy as np
from matplotlib import pyplot as plt
from Dynamics.SystemDynamics import GeneralDynamics
from Scenarios.MainScenarios import Scenario


class CombinedDynamics(GeneralDynamics):

    def __init__(self, scenario: Scenario):
        super().__init__(scenario)

    def create_model(self, sampling_time: float, **kwargs) -> ct.LinearIOSystem:
        pass

    def create_initial_condition(self, **kwargs) -> np.ndarray[float]:
        pass

    def create_reference(self, **kwargs) -> np.ndarray[float]:
        pass

    def get_plot_method(self) -> Callable[..., plt.figure]:
        pass

    def get_state_constraint(self) -> list[int, float]:
        pass

    def get_input_constraint(self) -> list[int, float]:
        pass

    def get_state_cost_matrix_sqrt(self) -> np.ndarray:
        pass

    def get_input_cost_matrix_sqrt(self) -> np.ndarray:
        pass