import numpy as np


class DynamicParameters:
    """
    Class to select different sets of dynamic parameters.
    """

    def __init__(self, state_limit: list, input_limit: list, q_sqrt: np.ndarray, r_sqrt_scalar: float | int):
        self.state_limit = state_limit
        self.input_limit = input_limit
        self.Q_sqrt = q_sqrt
        self.R_sqrt = r_sqrt_scalar * np.array([[0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0],
                                                [1, 0, 0],
                                                [0, 1, 0],
                                                [0, 0, 1]])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Dynamic set of parameters with {self.state_limit=}, " \
               f"{self.input_limit=}, {self.Q_sqrt=}, {self.R_sqrt=}"
