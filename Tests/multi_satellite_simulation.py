# Load standard modules
import numpy as np
from matplotlib import pyplot as plt
from tudatpy.kernel.interface import spice
from Space.OrbitalMechanics import OrbitalMechSimulator
from tudatpy.kernel import constants

# Find default physical parameters
spice.load_standard_kernels()

# Create simulator
orbital_sim = OrbitalMechSimulator()

# Create satellites
satellite_mass = 400.0  # [kg]
inertia_tensor = 0.01 * np.eye(3)
orbital_sim.create_bodies(number_of_satellites=3,
                          satellite_mass=satellite_mass,
                          satellite_inertia=inertia_tensor,
                          add_reference_satellite=True)

# Create thruster models
control_timestep = 10  # s
control_inputs = np.zeros((orbital_sim.number_of_controlled_satellites, 3, 10000))
control_inputs[0, 0] = 10
specific_impulse = 1E10  # Really high to make mass constant
orbital_sim.create_engine_models_thrust(control_timestep=control_timestep,
                                        thrust_inputs=control_inputs,
                                        specific_impulse=specific_impulse)

# Add forces to the satellites.
orbital_sim.create_acceleration_model(order_of_zonal_harmonics=0)

# Add mass rate model to the satellites
orbital_sim.create_mass_rate_model()

# Set the initial position of the satellites
true_anomalies = [0, 90, 180]
orbit_height = 750
orbital_sim.set_initial_position(orbital_sim.convert_orbital_elements_to_cartesian(true_anomalies=true_anomalies,
                                                                                   orbit_height=orbit_height,
                                                                                   inclination=45))

# Add dependent variables to track
orbital_sim.set_dependent_variables_translation(add_keplerian_state=True,
                                                add_thrust_accel=True,
                                                add_rsw_rotation_matrix=True)
orbital_sim.set_dependent_variables_mass(add_mass=True)

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
orbital_sim.plot_cylindrical_states()
orbital_sim.plot_3d_orbit()
anim = orbital_sim.create_animation()
orbital_sim.plot_keplerian_states(plot_argument_of_latitude=True)
orbital_sim.plot_thrusts()
plt.show()
