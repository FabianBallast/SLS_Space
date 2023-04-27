import numpy as np
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import environment_setup

# Find default physical parameters
spice.load_standard_kernels()
body_settings = environment_setup.get_default_body_settings(['Earth'], 'Earth', 'J2000')
bodies = environment_setup.create_system_of_bodies(body_settings)
radius_Earth_default = bodies.get('Earth').shape_model.average_radius
grav_param_Earth_default = bodies.get('Earth').gravitational_parameter


satellite_mass_default = 400
satellite_inertia_tensor_default = np.array([[10, 1, -1],
                                             [1, 10, 1],
                                             [-1, 1, 10]])
specific_impulse = 1e10


basic_physics = {'radius_Earth': radius_Earth_default,
                 'gravitational_parameter_Earth': grav_param_Earth_default,
                 'mass': satellite_mass_default,
                 'inertia_tensor': satellite_inertia_tensor_default,
                 'specific_impulse': specific_impulse,
                 'J2_perturbations': False,
                 'aerodynamic_forces': False,
                 'third_bodies': False,
                 'second_order_grav_torques': False}

basic_physics_scaled = {'radius_Earth': 55,
                        'gravitational_parameter_Earth': 100,
                        'mass': satellite_mass_default,
                        'inertia_tensor': satellite_inertia_tensor_default,
                        'specific_impulse': specific_impulse,
                        'J2_perturbations': False,
                        'aerodynamic_forces': False,
                        'third_bodies': False,
                        'second_order_grav_torques': False}

advanced_grav_physics = {'radius_Earth': radius_Earth_default,
                         'gravitational_parameter_Earth': grav_param_Earth_default,
                         'mass': satellite_mass_default,
                         'inertia_tensor': satellite_inertia_tensor_default,
                         'specific_impulse': specific_impulse,
                         'J2_perturbations': True,
                         'aerodynamic_forces': False,
                         'third_bodies': False,
                         'second_order_grav_torques': True}

advanced_grav_physics_scaled = {'radius_Earth': 55,
                                'gravitational_parameter_Earth': 100,
                                'mass': satellite_mass_default,
                                'inertia_tensor': satellite_inertia_tensor_default,
                                'specific_impulse': specific_impulse,
                                'J2_perturbations': True,
                                'aerodynamic_forces': False,
                                'third_bodies': False,
                                'second_order_grav_torques': True}

full_physics = {'radius_Earth': radius_Earth_default,
                'gravitational_parameter_Earth': grav_param_Earth_default,
                'mass': satellite_mass_default,
                'inertia_tensor': satellite_inertia_tensor_default,
                'specific_impulse': specific_impulse,
                'J2_perturbations': True,
                'aerodynamic_forces': True,
                'third_bodies': True,
                'second_order_grav_torques': True}

full_physics_scaled = {'radius_Earth': 55,
                       'gravitational_parameter_Earth': 100,
                       'mass': satellite_mass_default,
                       'inertia_tensor': satellite_inertia_tensor_default,
                       'specific_impulse': specific_impulse,
                       'J2_perturbations': True,
                       'aerodynamic_forces': True,
                       'third_bodies': True,
                       'second_order_grav_torques': True}

physics_scenarios = {'basic_physics': basic_physics,
                     'basic_physics_scaled': basic_physics_scaled,
                     'advanced_grav_physics': advanced_grav_physics,
                     'advanced_grav_physics_scaled': advanced_grav_physics_scaled,
                     'full_physics': full_physics,
                     'full_physics_scaled': full_physics_scaled}


def print_physics_scenarios() -> None:
    """
    Print the available scenarios regarding the physics.
    """
    print(f"The available physics scenarios are: {list(physics_scenarios.keys())}")


if __name__ == '__main__':
    print_physics_scenarios()
