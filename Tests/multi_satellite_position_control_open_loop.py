import numpy as np
from matplotlib import pyplot as plt
from tudatpy.kernel.interface import spice
from Dynamics.HCWDynamics import RelCylHCW
from Dynamics.ROEDynamics import QuasiROE
from Space.OrbitalMechanics import OrbitalMechSimulator
from SLS.SLS_setup import SLSSetup

# Find default physical parameters
spice.load_standard_kernels()

# Global parameters
satellite_mass = 400  # kg
control_timestep = 10  # s
orbital_height = 750  # km
number_of_satellites = 3  # -
simulation_duration = 0.1  # hours

# Create prediction using HCW model
# First, create system basics
# control_model = RelCylHCW(orbital_height=orbital_height, satellite_mass=satellite_mass)
control_model = QuasiROE(orbital_height=orbital_height, satellite_mass=satellite_mass)
sls_setup = SLSSetup(sampling_time=control_timestep, system_dynamics=control_model, tFIR=20)
sls_setup.create_system(number_of_systems=number_of_satellites)

# Add cost matrices
# Q_matrix_sqrt = 1 * np.array([4, 8, 4, 0, 0, 0])
# R_matrix_sqrt = 1e-4 * 1 * np.array([[0, 0, 0],
#                                      [0, 0, 0],
#                                      [0, 0, 0],
#                                      [1, 0, 0],
#                                      [0, 1, 0],
#                                      [0, 0, 1]])
Q_matrix_sqrt = 1 * np.diag(np.array([100, 10, 100, 100, 0, 0]))
R_matrix_sqrt = 1e-3 * 1 * np.array([[0, 0, 0],
                                     [0, 0, 0],
                                     [0, 0, 0],
                                     [1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])
sls_setup.create_cost_matrices(Q_matrix_sqrt=Q_matrix_sqrt, R_matrix_sqrt=R_matrix_sqrt)

# Create x0 and x_ref
sls_setup.create_x0(number_of_dropouts=1, seed=129)
sls_setup.create_reference()

# Start simulation
t_horizon = int(simulation_duration * 3600 / control_timestep) + 1
sls_setup.simulate_system(t_horizon=t_horizon, noise=None, progress=True)

# Create simulator
orbital_sim = OrbitalMechSimulator()

# Create satellites
satellite_mass = satellite_mass  # [kg]
inertia_tensor = 0.01 * np.eye(3)
orbital_sim.create_bodies(number_of_satellites=number_of_satellites,
                          satellite_mass=satellite_mass,
                          satellite_inertia=inertia_tensor,
                          add_reference_satellite=True)

# Create thruster models
control_inputs = sls_setup.u_inputs.reshape((number_of_satellites, 3, -1))
control_inputs_sim = control_inputs.copy()
# control_inputs_sim[:, 0, :] *= 0.85
specific_impulse = 1E10  # Really high to make mass constant
orbital_sim.create_engine_models_thrust(control_timestep=control_timestep,
                                        thrust_inputs=control_inputs_sim,
                                        specific_impulse=specific_impulse)

# Add forces to the satellites.
orbital_sim.create_acceleration_model(order_of_zonal_harmonics=0)

# Add mass rate model to the satellites
orbital_sim.create_mass_rate_model()

# Set the initial position of the satellites
true_anomalies = [0, 90, 180]
orbital_sim.set_initial_position(orbital_sim.convert_orbital_elements_to_cartesian(true_anomalies=true_anomalies,
                                                                                   orbit_height=orbital_height))

# Add dependent variables to track
orbital_sim.set_dependent_variables_translation(add_keplerian_state=True,
                                                add_thrust_accel=True,
                                                add_rsw_rotation_matrix=True)
orbital_sim.set_dependent_variables_mass(add_mass=True)

# Create propagation settings
start_epoch = 0
end_epoch = start_epoch + simulation_duration * 3600
simulation_timestep = 2  # s
orbital_sim.create_propagation_settings(start_epoch=start_epoch,
                                        end_epoch=end_epoch,
                                        simulation_step_size=simulation_timestep)

# Simulate system
states_array, dep_vars_array = orbital_sim.simulate_system()

# Find cylindrical coordinates
cylindrical_coordinates = orbital_sim.convert_to_cylindrical_coordinates()


# Plot results
orbital_sim.plot_3d_orbit()
anim = orbital_sim.create_animation()
fig_rel_states = orbital_sim.plot_states_of_dynamical_model(dynamical_model=control_model)
sls_setup.plot_states(figure=fig_rel_states)
orbital_sim.plot_cylindrical_states()
fig_inputs = orbital_sim.plot_thrusts()
sls_setup.plot_inputs(figure=fig_inputs)
plt.show()
