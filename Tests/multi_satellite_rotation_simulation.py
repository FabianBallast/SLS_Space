# Load standard modules
import numpy as np
from matplotlib import pyplot as plt
from tudatpy.kernel import constants
from Space.OrbitalMechanics import OrbitalMechSimulator
from SLS.SLS_setup import SLSSetup

# Global parameters
number_of_satellites = 3
satellite_mass = 400  # kg
satellite_moment_of_inertia = 10 * np.eye(3)  # kg m^2
# satellite_moment_of_inertia[0, 1] = satellite_moment_of_inertia[1, 0] = 5
orbital_sim = OrbitalMechSimulator()

# Create satellites
orbital_sim.create_bodies(number_of_satellites=number_of_satellites, satellite_mass=satellite_mass,
                          satellite_inertia=satellite_moment_of_inertia, add_reference_satellite=True)

# Create thrust models
control_timestep = 10  # s
control_inputs = np.zeros((orbital_sim.number_of_controlled_satellites, 3, 10000))
specific_impulse = 1E10  # Really high to make mass constant
orbital_sim.create_engine_models_thrust(control_timestep=control_timestep,
                                        thrust_inputs=control_inputs,
                                        specific_impulse=specific_impulse)

# Create torque models
control_timestep = 10  # s
control_inputs = np.zeros((orbital_sim.number_of_controlled_satellites, 3, 10000))
control_inputs[0, 0, 0:10] = 0*0.001
specific_impulse = 1E10  # Really high to make mass constant
orbital_sim.create_engine_models_torque(control_timestep=control_timestep,
                                        torque_inputs=control_inputs,
                                        specific_impulse=specific_impulse)

# Add acceleration to the model
orbital_sim.create_acceleration_model(order_of_zonal_harmonics=0)

# Add torques to the satellites. Currently only adds 2nd degree grav torque.
orbital_sim.create_torque_model()

# Set the initial position of the satellites
true_anomalies = [0, 90, 180]
orbit_height = 750

orbital_sim.set_initial_position(orbital_sim.convert_orbital_elements_to_cartesian(true_anomalies=true_anomalies,
                                                                                   orbit_height=orbit_height,
                                                                                   inclination=0))

# Set the initial rotations
initial_quaternion = np.array([1, 0, 0, 0])  # Scalar part at the front
# angle = np.deg2rad(180)  # rad
# initial_rot = np.array([[np.cos(angle), np.sin(angle), 0],
#                         [-np.sin(angle), np.cos(angle), 0],
#                         [0, 0, 1]])
initial_omega = np.array([0, 0, 0])
orbital_sim.set_initial_orientation(initial_quaternion, initial_omega)

# Add dependent variables to track
orbital_sim.set_dependent_variables_translation(add_keplerian_state=True,
                                                add_rsw_rotation_matrix=True,
                                                add_thrust_accel=True)
orbital_sim.set_dependent_variables_rotation(add_torque_norm=True)

# Create propagation settings
start_epoch = 0
end_epoch = start_epoch + 6 * constants.JULIAN_DAY / 24
simulation_timestep = 2  # s
orbital_sim.create_propagation_settings(start_epoch=start_epoch,
                                        end_epoch=end_epoch,
                                        simulation_step_size=simulation_timestep)

# Simulate system
states_array, dep_vars_array = orbital_sim.simulate_system()

# Plot results
# orbital_sim.plot_cylindrical_states()
orbital_sim.plot_3d_orbit()
# anim = orbital_sim.create_animation()
orbital_sim.plot_keplerian_states(plot_argument_of_latitude=True)
# orbital_sim.plot_inputs()
orbital_sim.plot_quaternions()
orbital_sim.plot_quaternions_rsw()
orbital_sim.plot_euler_angles()
plt.show()
