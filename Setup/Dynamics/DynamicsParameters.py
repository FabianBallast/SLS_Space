import numpy as np


class DynamicParameters:
    """
    Class to select different sets of dynamic parameters.
    """

    def __init__(self, state_limit: list, input_limit: list, q_sqrt: np.ndarray, r_sqrt_scalar: float | int,
                 slack_variable_length: int = 0, slack_variable_costs: list[int] = None,
                 planetary_distance: float | int = -np.inf, inter_planetary_distance: float | int = -np.inf,
                 radial_distance: float | int = -np.inf):
        self.state_limit = state_limit
        self.input_limit = input_limit
        self.Q_sqrt = q_sqrt
        self.R_sqrt = r_sqrt_scalar * np.array([[0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0],
                                                [1, 0, 0],
                                                [0, 1, 0],
                                                [0, 0, 1]])
        self.slack_variable_length = slack_variable_length

        if slack_variable_costs is not None:
            self.slack_variable_costs = slack_variable_costs
        else:
            self.slack_variable_costs = [0] * len(state_limit)

        self.planetary_distance = planetary_distance
        self.inter_planetary_distance = inter_planetary_distance
        self.radial_distance = radial_distance

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Dynamic set of parameters with {self.state_limit=}, " \
               f"{self.input_limit=}, {self.Q_sqrt=}, {self.R_sqrt=}"
