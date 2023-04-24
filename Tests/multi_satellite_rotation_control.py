# Load standard modules
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from tudatpy.kernel import constants
from Space.OrbitalMechanics import OrbitalMechSimulator
from SLS.SLS_setup import SLSSetup
from Dynamics.AttitudeDynamics import LinAttModel

# Global parameters
satellite_mass = 400  # kg
satellite_moment_of_inertia = 10 * np.eye(3)
control_timestep = 5  # s
orbital_height = 750  # km
number_of_satellites = 3  # -
simulation_duration = 0.01  # hours
simulation_timestep = 1  # s
start_epoch = 0
end_epoch = start_epoch + simulation_duration * 3600
specific_impulse = 1E10  # Really high to make mass constant
inclination = 30
longitude = 10
offset = Rotation.from_euler('XYZ', [30, -41, 22], degrees=True)


control_model = LinAttModel(orbital_height=orbital_height*1000, satellite_moment_of_inertia=satellite_moment_of_inertia)
sls_setup = SLSSetup(sampling_time=control_timestep, system_dynamics=control_model, tFIR=10)
sls_setup.create_system(number_of_systems=number_of_satellites)

# Add cost matrices
sls_setup.create_cost_matrices()

# Create x0 and x_ref
sls_setup.create_x0()
sls_setup.create_reference()

# Create loop
t_horizon_control = int(np.ceil(simulation_duration * 3600 / control_timestep)) + 1
control_simulation_ratio = int(control_timestep / simulation_timestep)
t_horizon_simulation = control_simulation_ratio * t_horizon_control

sls_states = np.zeros((t_horizon_simulation + 1, len(sls_setup.x0)))
pos_states = np.zeros((t_horizon_simulation + 1, (number_of_satellites + 1) * 6))  # Also include reference
rot_states = np.zeros((t_horizon_simulation + 1, (number_of_satellites + 1) * 7))  # Also include reference

dep_vars = None
dep_var_dict = None

# Small simulation to initialise all parameters
orbital_sim = OrbitalMechSimulator()
orbital_sim.create_bodies(number_of_satellites=number_of_satellites, satellite_mass=satellite_mass,
                          satellite_inertia=satellite_moment_of_inertia, add_reference_satellite=True)

# Create thrust models
control_inputs = np.zeros((orbital_sim.number_of_controlled_satellites, 3, 2))
orbital_sim.create_engine_models_thrust(control_timestep=control_timestep,
                                        thrust_inputs=control_inputs,
                                        specific_impulse=specific_impulse)

# Create torque models
orbital_sim.create_engine_models_torque(control_timestep=control_timestep,
                                        torque_inputs=control_inputs,
                                        specific_impulse=specific_impulse)

# Add acceleration to the model
orbital_sim.create_acceleration_model(order_of_zonal_harmonics=0)

# Add torques to the satellites. Currently only adds 2nd degree grav torque.
orbital_sim.create_torque_model()

# Set the initial position of the satellites
true_anomalies = list(np.linspace(0, 360, number_of_satellites, endpoint=False))

pos_states[0:1] = orbital_sim.convert_orbital_elements_to_cartesian(true_anomalies=true_anomalies,
                                                                    orbit_height=orbital_height,
                                                                    inclination=inclination,
                                                                    longitude=longitude)


angular_vel = np.array([0, 0, -1*sls_setup.dynamics.mean_motion]) + 1 * np.array([0.01, -0.05, 0.05])
rot_states[0:1] = orbital_sim.convert_orbital_elements_to_quaternion(true_anomalies=true_anomalies,
                                                                     initial_angular_velocity=angular_vel,
                                                                     inclination=inclination,
                                                                     longitude=longitude,
                                                                     initial_offset=offset)

orbital_sim.set_initial_position(pos_states[0:1].T)
orbital_sim.set_initial_orientation(rot_states[0:1].T)

# Add dependent variables to track
orbital_sim.set_dependent_variables_translation(add_keplerian_state=False,
                                                add_rsw_rotation_matrix=True,
                                                add_thrust_accel=False)
orbital_sim.set_dependent_variables_rotation(add_control_torque=False, add_torque_norm=False)

# Create propagation settings
orbital_sim.create_propagation_settings(start_epoch=0,
                                        end_epoch=1,
                                        simulation_step_size=1)

# Simulate system
orbital_sim.simulate_system()
orbital_sim.set_mean_motion(sls_setup.dynamics.mean_motion)
control_states = orbital_sim.convert_to_euler_state()
sls_states[0:1] = control_states[1:control_simulation_ratio + 1]

progress = 0
for t in range(t_horizon_control):
    if t / t_horizon_control * 100 > progress:
        print(f"Progress: {int(t / t_horizon_control * 100)}%")
        progress = int(t / t_horizon_control * 100) + 1

    # Start simulation
    sls_setup.set_initial_conditions(sls_states[t * control_simulation_ratio:t * control_simulation_ratio + 1].T)
    sls_setup.simulate_system(t_horizon=1, noise=None, inputs_to_store=1)
    control_inputs = sls_setup.u_inputs.reshape((number_of_satellites, 3, -1))

    # Create satellites
    orbital_sim = OrbitalMechSimulator()
    orbital_sim.create_bodies(number_of_satellites=number_of_satellites, satellite_mass=satellite_mass,
                              satellite_inertia=satellite_moment_of_inertia, add_reference_satellite=True)

    # Create thrust models
    control_inputs_thrust = np.zeros((orbital_sim.number_of_controlled_satellites, 3, 2))
    orbital_sim.create_engine_models_thrust(control_timestep=control_timestep,
                                            thrust_inputs=control_inputs_thrust,
                                            specific_impulse=specific_impulse)

    # Create torque models
    orbital_sim.create_engine_models_torque(control_timestep=control_timestep,
                                            torque_inputs=control_inputs,
                                            specific_impulse=specific_impulse)

    # Add acceleration to the model
    orbital_sim.create_acceleration_model(order_of_zonal_harmonics=0)

    # Add torques to the satellites. Currently only adds 2nd degree grav torque.
    orbital_sim.create_torque_model()

    # Set initial position/orientation
    orbital_sim.set_initial_position(pos_states[t * control_simulation_ratio:t * control_simulation_ratio + 1].T)
    orbital_sim.set_initial_orientation(rot_states[t * control_simulation_ratio:t * control_simulation_ratio + 1].T)

    # Add dependent variables to track
    orbital_sim.set_dependent_variables_translation(add_keplerian_state=True,
                                                    add_rsw_rotation_matrix=True,
                                                    add_thrust_accel=True)
    orbital_sim.set_dependent_variables_rotation(add_control_torque=True, add_torque_norm=True)

    # Create propagation settings
    start_epoch = t * control_timestep
    end_epoch = (t + 1) * control_timestep
    orbital_sim.create_propagation_settings(start_epoch=start_epoch,
                                            end_epoch=end_epoch,
                                            simulation_step_size=simulation_timestep)

    # Simulate system
    index_selection = np.arange(t * control_simulation_ratio + 1, (t+1) * control_simulation_ratio + 1)
    orbital_sim.simulate_system()
    dep_vars_array = orbital_sim.dep_vars
    pos_states[index_selection] = orbital_sim.translation_states[1:]
    rot_states[index_selection] = orbital_sim.rotation_states[1:]

    if dep_vars is None:
        dep_vars = np.zeros((t_horizon_simulation + 1, dep_vars_array.shape[1]))
        dep_vars[0] = dep_vars_array[0]
        dep_var_dict = orbital_sim.dependent_variables_dict
    dep_vars[index_selection] = dep_vars_array[1:]

    # Find states for SLS model
    orbital_sim.set_mean_motion(sls_setup.dynamics.mean_motion)
    control_states = orbital_sim.convert_to_euler_state()
    sls_states[index_selection] = control_states[1:control_simulation_ratio + 1]


orbital_sim = OrbitalMechSimulator()
orbital_sim.number_of_controlled_satellites = number_of_satellites
orbital_sim.number_of_total_satellites = number_of_satellites + 1
orbital_sim.simulation_timestep = simulation_timestep
orbital_sim.translation_states = pos_states  # np.concatenate((abs_states[:, 0:1] * 0, abs_states), axis=1)
orbital_sim.rotation_states = rot_states
orbital_sim.dep_vars = dep_vars
orbital_sim.dependent_variables_dict = dep_var_dict
orbital_sim.controlled_satellite_names = ["Satellite_0", "Satellite_1", "Satellite_2"]
orbital_sim.satellite_mass = satellite_mass
orbital_sim.set_mean_motion(sls_setup.dynamics.mean_motion)

# print(inputs)

# Plot results
# orbital_sim.plot_cylindrical_states()
# orbital_sim.plot_3d_orbit()
# anim = orbital_sim.create_animation()
# orbital_sim.plot_keplerian_states(plot_argument_of_latitude=True)
# orbital_sim.plot_thrusts()
# orbital_sim.plot_quaternions()
# orbital_sim.plot_quaternions_rsw()
orbital_sim.plot_euler_angles()
orbital_sim.plot_angular_velocities()
orbital_sim.plot_torques()
plt.show()
