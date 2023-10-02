from typing import Callable
import numpy as np
import control as ct
from abc import ABC, abstractmethod
from scipy.linalg import block_diag
from matplotlib import pyplot as plt
from Scenarios.MainScenarios import Scenario
from Scenarios.PhysicsScenarios import ScaledPhysics


class GeneralDynamics(ABC):
    """
    Abstract base class to deal with different dynamical models.
    """

    def __init__(self, scenario: Scenario):
        """
        Initialising the object for a translational model.

        :param scenario: The scenario which has been selected, and which should include at least:
                             - orbital_height: Height above the surface of the Earth in m.
                             - radius_Earth: The radius of the Earth in m.
                             - gravitational_parameter_Earth: The gravitational parameter of the Earth in m^3/s^2
        """
        if not 500e3 < scenario.physics.orbital_height < 5000e3 and scenario.physics.orbital_height > 100:
            raise Exception("Orbital height should be in meters!")

        self.orbital_height = scenario.physics.orbital_height  # m
        self.earth_gravitational_parameter = scenario.physics.gravitational_parameter_Earth  # m^3 s^-2
        self.earth_radius = scenario.physics.radius_Earth  # m
        self.orbit_radius = (self.orbital_height + self.earth_radius)  # m
        self.mean_motion = np.sqrt(self.earth_gravitational_parameter / self.orbit_radius ** 3) * (1 + 1.9359e-8)  # rad/s
        self.is_LTI = True
        self.state_size = 6  # Default state size
        self.input_size = 3  # Default input size
        self.is_scaled = isinstance(scenario.physics, ScaledPhysics)

        # For updating ROE
        self.J2_active = scenario.physics.J2_perturbation
        self.J2_value = scenario.physics.J2_value
        self.inclination = np.deg2rad(scenario.orbital.inclination)
        self.eccentricity = scenario.orbital.eccentricity
        self.periapsis = np.deg2rad(scenario.orbital.argument_of_periapsis)
        self.J2_scaling_factor = 0.75 * self.J2_value * (self.earth_radius / (self.orbit_radius * (1-self.eccentricity**2)))**2 * self.mean_motion

    @abstractmethod
    def create_model(self, sampling_time: float, **kwargs) -> ct.LinearIOSystem:
        """
        Create a discrete-time model for a single satellite with an A, B, C and D matrix.

        :param sampling_time: Sampling time in s.
        :param kwargs: Possible keyword argument for model. Depends on subclass that is using it (and might be unused).
        :return: A linear system object.
        """
        pass

    def create_multi_satellite_model(self, sampling_time: float, **kwargs) -> ct.LinearIOSystem:
        """
        Create a discrete-time model for multiple satellites with an A, B, C and D matrix.

        :param sampling_time: Sampling time in s.
        :param kwargs: Is used to determine to number of satellites, and possible more specific required information.
                       If the system is LTI, adding 'number_of_systems' suffices for this function. However, if it is
                       LTV, the model depends on variables such as the argument of latitude. In that case, add a keyword
                       argument with the name '${keyword_name}_list' and a list of desired values.
        :return: A linear system object.
        """
        if self.is_LTI and "number_of_systems" in kwargs:
            model = self.create_model(sampling_time)
            A = [model.A] * kwargs["number_of_systems"]
            B = [model.B] * kwargs["number_of_systems"]
            C = [model.C] * kwargs["number_of_systems"]
            D = [model.D] * kwargs["number_of_systems"]
        else:
            A = []
            B = []
            C = []
            D = []

            keyword_name_0 = list(kwargs.keys())[0].replace("_list", "")
            keyword_name_1 = list(kwargs.keys())[1].replace("_list", "")

            for idx, keyword_value in enumerate(list(kwargs.values())[0]):
                model = self.create_model(sampling_time, **{keyword_name_0: keyword_value},
                                          **{keyword_name_1: list(kwargs.values())[1][idx]})
                A.append(model.A)
                B.append(model.B)
                C.append(model.C)
                D.append(model.D)

        return ct.ss(block_diag(*A), block_diag(*B), block_diag(*C), block_diag(*D))

    @abstractmethod
    def create_initial_condition(self, **kwargs) -> np.ndarray[float]:
        """
        Create an array with the initial conditions of this system.

        :param kwargs: Keyword argument to set a specific initial condition.
        :return: An array with the initial conditions.
        """
        pass

    @abstractmethod
    def create_reference(self, **kwargs) -> np.ndarray[float]:
        """
        Create an array with the reference of this system.

        :param kwargs: Keyword argument to set a specific reference.
        :return: An array with the reference.
        """
        pass

    @abstractmethod
    def get_plot_method(self) -> Callable[..., plt.figure]:
        """
        Return the method that can be used to plot this dynamical model.

        :return: Callable with arguments:
                 states: np.ndarray, timestep: float, name: str = None, figure: plt.figure = None, kwargs
        """
        pass

    @abstractmethod
    def get_state_constraint(self) -> list[int, float]:
        """
        Return the vector x_lim such that -x_lim <= x <= x_lim

        :return: List with maximum state values
        """
        pass

    @abstractmethod
    def get_input_constraint(self) -> list[int, float]:
        """
        Return the vector u_lim such that -u_lim <= u <= u_lim

        :return: List with maximum input values
        """
        pass

    @abstractmethod
    def get_state_cost_matrix_sqrt(self) -> np.ndarray:
        """
        Provide the matrix Q_sqrt

        :return: An nxn dimensional matrix representing Q_sqrt
        """
        pass

    @abstractmethod
    def get_input_cost_matrix_sqrt(self) -> np.ndarray:
        """
        Provide the matrix R_sqrt

        :return: An nxm dimensional matrix representing R_sqrt
        """
        pass

    def get_orbital_differentiation(self) -> np.ndarray:
        """
        Return the derivative of orbital elements.
        Order: [a, e, i, Omega, omega, M)

        :return: Derivatives in shape (6,).
        """
        if self.J2_active:
            or_dif = self.J2_scaling_factor * np.array([0, 0, 0, -2 * np.cos(self.inclination), 5 * np.cos(self.inclination)**2 - 1, np.sqrt(1-self.eccentricity**2) * (3 * np.cos(self.inclination)**2 - 1)])
            # Small corrections
            # or_dif[2] = -2.95e-8
            # or_dif[4:6] *= (1 + 0.0018996)
            # or_dif[3] *= 1.0052669
            return or_dif
        else:
            return np.array([0 * -3.3178e-10, 0, 0, 0, 0, 0])

    @abstractmethod
    def get_slack_variable_length(self) -> int:
        """
        Get the time horizon for which to use slack variables.

        :return: Time for which slack variables are used.
        """
        pass

    @abstractmethod
    def get_slack_costs(self) -> list[int]:
        """
        Find the states for which to use slack variables and their costs

        :return: List with positive cost for states where slack variables should be applied.
        """
        pass


class TranslationalDynamics(GeneralDynamics, ABC):
    """
    Abstract base class to deal with different translational models.
    """

    def __init__(self, scenario: Scenario):
        """
        Initialising the object for a translational model.

        :param scenario: The scenario which has been selected, and which should include at least:
                             - orbital_height: Height above the surface of the Earth in m.
                             - radius_Earth: The radius of the Earth in m.
                             - gravitational_parameter_Earth: The gravitational parameter of the Earth in m^3/s^2
                             - mass: Mass of the satellite in kg.
        """
        super().__init__(scenario)
        self.satellite_mass = scenario.physics.mass  # kg

    @abstractmethod
    def get_positional_angles(self) -> np.ndarray[bool]:
        """
        Find the position of the relative angle in the model.

        :return: Array of bools with True on the position of the relative angle.
        """
        pass

    @abstractmethod
    def create_initial_condition(self, relative_angle: float) -> np.ndarray[float]:
        """
        Create an array with the initial conditions of this system.

        :param relative_angle: The relative angle with respect to a reference in degrees.
        :return: An array with the initial conditions.
        """
        pass

    @abstractmethod
    def create_reference(self, relative_angle: float) -> np.ndarray[float]:
        """
        Create an array with the reference of this system.

        :param relative_angle: The relative angle with respect to a reference in degrees.
        :return: An array with the reference.
        """
        pass

    @abstractmethod
    def get_angles_list(self) -> list[bool]:
        """
        Find all the values that represent an angle.

        :return: Return a list with True for every state that represents an angle.
        """
        pass

    @abstractmethod
    def get_orbital_parameter(self) -> list[bool]:
        """
        Find the parameter that determines if two orbits are close.

        :return: List with bool with True for the orbital parameter.
        """
        pass

    @abstractmethod
    def get_planetary_distance(self) -> int | float:
        """
        Find the minimum planetary distance.

        :return: The minimum planetary distance
        """
        pass

    @abstractmethod
    def get_inter_planetary_distance(self) -> int | float:
        """
        Find the minimum inter_planetary distance.

        :return: The minimum inter_planetary distance
        """
        pass


class AttitudeDynamics(GeneralDynamics, ABC):
    """
    Abstract base class to deal with different attitude models.
    """

    def __init__(self, scenario: Scenario):
        """
        Initialising the object for a translational model.

        :param scenario: The scenario which has been selected, and which should include at least:
                             - orbital_height: Height above the surface of the Earth in m.
                             - radius_Earth: The radius of the Earth in m.
                             - gravitational_parameter_Earth: The gravitational parameter of the Earth in m^3/s^2
                             - inertia_tensor: Moment of inertia tensor of the satellite in kg m^2.
        """
        super().__init__(scenario)
        self.satellite_moment_of_inertia = scenario.physics.inertia_tensor  # kg m^2

    @abstractmethod
    def create_initial_condition(self, quaternion: np.ndarray[float]) -> np.ndarray[float]:
        """
        Create an array with the initial conditions of this system.

        :param quaternion: The initial state as a quaternion.
        :return: An array with the initial conditions.
        """
        pass

    @abstractmethod
    def create_reference(self, quaternion: np.ndarray[float]) -> np.ndarray[float]:
        """
        Create an array with the reference of this system.

        :param quaternion: The reference state as a quaternion.
        :return: An array with the reference.
        """
        pass

