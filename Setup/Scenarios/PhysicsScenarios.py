from enum import Enum
import numpy as np
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import environment_setup

# Find default physical parameters
spice.load_standard_kernels()
body_settings = environment_setup.get_default_body_settings(['Earth'], 'Earth', 'J2000')


class Physics:
    """
    Create the parameters related to the physics.
    """

    def __init__(self, radius_earth=body_settings.get('Earth').shape_settings.radius,
                 grav_param_earth=body_settings.get('Earth').gravity_field_settings.gravitational_parameter,
                 orbital_height=750e3, atmosphere_density=1e-13, mass=400, inertia_tensor=np.array([[10, 1, -1], [1, 10, 1], [-1, 1, 10]]),
                 specific_impulse=1e10, j2_perturbation=False, j2_value = 1.08262668e-3, aerodynamic_forces=False,
                 third_bodies=False, second_order_grav_torques=False):
        """
        :param radius_earth: Radius of the Earth in m.
        :param grav_param_earth: Gravitational parameter of the Earth in m^2/s^3
        :param orbital_height: Height above the surface of the Earth in m.
        :param atmosphere_density: Density of the air in kg/m^3
        :param mass: Mass of the satellite in kg.
        :param inertia_tensor: Mass moment of inertia tensor in kg m^2
        :param specific_impulse: Specific impulse in N/s
        :param j2_perturbation: Whether to simulate J2 perturbations.
        :param aerodynamic_forces: Whether to simulate aerodynamic forces.
        :param third_bodies: Whether to simulate third body gravitational forces.
        :param second_order_grav_torques: Whether to simulate second order gravitational torques.
        """
        self.radius_Earth = radius_earth
        self.gravitational_parameter_Earth = grav_param_earth
        self.orbital_height = orbital_height
        self.atmosphere_density = atmosphere_density
        self.mass = mass
        self.inertia_tensor = inertia_tensor
        self.specific_impulse = specific_impulse
        self.J2_value = j2_value
        self.J2_perturbation = j2_perturbation
        self.aerodynamic_forces = aerodynamic_forces
        self.third_bodies = third_bodies
        self.second_order_grav_torques = second_order_grav_torques

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Physics with {self.radius_Earth=}, {self.gravitational_parameter_Earth=}, {self.orbital_height=}," \
               f"{self.mass=}, {self.inertia_tensor=}, {self.specific_impulse=}, {self.J2_perturbation=}, " \
               f"{self.aerodynamic_forces=}, {self.third_bodies=} and {self.second_order_grav_torques=}"


class ScaledPhysics(Physics):
    """
    Create the parameters related to the scaled physics.
    """

    def __init__(self, radius_earth=40, grav_param_earth=100, orbital_height=15, atmosphere_density=1e-6, **kwargs):
        super().__init__(radius_earth, grav_param_earth, orbital_height, atmosphere_density, **kwargs)


class PhysicsScenarios(Enum):
    """
    An Enum for different sets of physical parameters.
    """
    basic_physics = Physics()
    basic_physics_scaled = ScaledPhysics()

    advanced_grav_physics = Physics(j2_perturbation=True, second_order_grav_torques=True)
    advanced_grav_physics_scaled = ScaledPhysics(j2_perturbation=True, second_order_grav_torques=True)

    full_physics = Physics(j2_perturbation=True, aerodynamic_forces=True, third_bodies=True,
                           second_order_grav_torques=True)
    full_physics_scaled = ScaledPhysics(j2_perturbation=True, aerodynamic_forces=True, third_bodies=True,
                                        second_order_grav_torques=True)


if __name__ == '__main__':
    print(list(PhysicsScenarios))
