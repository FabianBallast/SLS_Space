# Load standard modules
import numpy as np
from matplotlib import pyplot as plt

# Load tudatpy modules
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment, environment_setup, propagation_setup
from tudatpy.kernel.astro import element_conversion, frame_conversion
from tudatpy.kernel import constants
from tudatpy.util import result2array

# Load spice kernels
spice.load_standard_kernels()

# Set simulation start and end epochs
simulation_start_epoch = 0.0
simulation_end_epoch = constants.JULIAN_DAY

# Create default body settings for "Earth"
bodies_to_create = ["Earth"]

# Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

# Create system of bodies (in this case only Earth)
bodies = environment_setup.create_system_of_bodies(body_settings)

# Add vehicle object to system of bodies
bodies.create_empty_body("Delfi-C3")
initial_mass = 400.0
inertia_tensor = 0.01 * np.eye(3)
# inertia_tensor[0,2] = 0.005
bodies.get("Delfi-C3").mass = initial_mass
bodies.get("Delfi-C3").inertia_tensor = inertia_tensor

# Define bodies that are propagated
bodies_to_propagate = ["Delfi-C3"]

# Define central bodies of propagation
central_bodies = ["Earth"]


# Define the thrust magnitude function
def thrust_magnitude_function(time):
    return 0  # N


# Define a specific impulse constant at 350s
specific_impulse = 1E9  # Really high to make mass constant

# Define the custom thrust magnitude settings based on the pre-defined functions
engine_settings = propagation_setup.thrust.custom_thrust_magnitude_fixed_isp(thrust_magnitude_function,
                                                                             specific_impulse)

environment_setup.add_engine_model("Delfi-C3", "x-axis", engine_settings, bodies, np.array([1., 0., 0.]))


def inertial_direction(time):
    # Get aerodynamic angle calculator
    #     aerodynamic_angle_calculator = bodies.get("Delfi-C3").flight_conditions.aerodynamic_angle_calculator

    # Set thrust in RSW (LVLH) frame and transpose it
    thrust_direction_rsw_frame = np.array([[0, 1, 0]]).T

    current_state = bodies.get("Delfi-C3").state
    rsw_to_inertial_frame = frame_conversion.rsw_to_inertial_rotation_matrix(current_state)

    # Compute the thrust in the inertial frame
    thrust_inertial_frame = np.dot(rsw_to_inertial_frame, thrust_direction_rsw_frame)

    # Return the thrust direction in the inertial frame
    return thrust_inertial_frame


rotation_model = environment_setup.rotation_model.custom_inertial_direction_based(inertial_direction, "J2000",
                                                                                  "VehicleFixed")
# rotation_model = environment_setup.rotation_model.orbital_state_direction_based("Earth", False, True,  "J2000", "VehicleFixed")
# rotation_model = environment_setup.rotation_model.synchronous("Earth", "J2000", "VehicleFixed")  # Always crashes
# print(bodies.get("Earth").body_fixed_frame_name())
environment_setup.add_rotation_model(bodies, "Delfi-C3", rotation_model)

# Define torque models
# Define torque settings acting on spacecraft
torque_settings_spacecraft = dict(Earth=[propagation_setup.torque.second_degree_gravitational()])
torque_settings = {"Delfi-C3": torque_settings_spacecraft}

# Create torque models.
torque_models = propagation_setup.create_torque_models(bodies, torque_settings, bodies_to_propagate)

# Below, we define the initial state in a somewhat trivial manner (body axes along global frame
# axes; no initial rotation). A real application should use a more realistic initial rotational state
# Set initial rotation matrix (identity matrix)
initial_rotation_matrix = np.eye(3)
# Set initial orientation by converting a rotation matrix to a Tudat-compatible quaternion
initial_state_rot = element_conversion.rotation_matrix_to_quaternion_entries(initial_rotation_matrix)
# Complete initial state by adding angular velocity vector (zero in this case)
initial_state_rot = np.concatenate((initial_state_rot, np.array([1, 0, 0])))

# Define propagator type
propagator_type = propagation_setup.propagator.modified_rodrigues_parameters

# Define accelerations acting on Delfi-C3
acceleration_settings_delfi_c3 = {
    "Earth": [propagation_setup.acceleration.point_mass_gravity()],
    "Delfi-C3": [propagation_setup.acceleration.thrust_from_all_engines()]
}

acceleration_settings = {"Delfi-C3": acceleration_settings_delfi_c3}

# Create acceleration models
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)

mass_rate_settings = {"Delfi-C3": [propagation_setup.mass_rate.from_thrust()]}
mass_rate_models = propagation_setup.create_mass_rate_models(
    bodies,
    mass_rate_settings,
    acceleration_models
)

# Set initial conditions for the satellite that will be
# propagated in this simulation. The initial conditions are given in
# Keplerian elements and later on converted to Cartesian elements
earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter
initial_state_trans = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=earth_gravitational_parameter,
    semi_major_axis=7500.0e3,
    eccentricity=0.0,
    inclination=0,  # np.deg2rad(85.3),
    argument_of_periapsis=0,  # np.deg2rad(235.7),
    longitude_of_ascending_node=0,  # np.deg2rad(23.4),
    true_anomaly=0,  # np.deg2rad(139.87),
)

# Create termination settings
termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

# Create numerical integrator settings
fixed_step_size = 10.0
integrator_settings = propagation_setup.integrator.runge_kutta_4(fixed_step_size)

# Also save orbital elements
dependent_variables_to_save_trans = [
    propagation_setup.dependent_variable.keplerian_state("Delfi-C3", "Earth"),
    propagation_setup.dependent_variable.single_acceleration(
        propagation_setup.acceleration.AvailableAcceleration.point_mass_gravity_type, "Delfi-C3", "Earth"),
    propagation_setup.dependent_variable.single_acceleration(
        propagation_setup.acceleration.AvailableAcceleration.thrust_acceleration_type, "Delfi-C3", "Delfi-C3")
]

# Create propagation settings translational part
translational_propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state_trans,
    simulation_start_epoch,
    integrator_settings,
    termination_settings,
    output_variables=dependent_variables_to_save_trans
)

# Define rotational propagator settings
dependent_variables_to_save_rot = [propagation_setup.dependent_variable.total_torque_norm("Delfi-C3")]
rotational_propagator_settings = propagation_setup.propagator.rotational(
    torque_models,
    bodies_to_propagate,
    initial_state_rot,
    simulation_start_epoch,
    integrator_settings,
    termination_settings,
    # propagator=propagator_type,
    # output_variables=dependent_variables_to_save_rot
)

# Create propagation settings for the mass
dependent_variables_to_save_mass = [propagation_setup.dependent_variable.body_mass("Delfi-C3")]
mass_propagator_settings = propagation_setup.propagator.mass(
    bodies_to_propagate,
    mass_rate_models,
    [initial_mass],
    simulation_start_epoch,
    integrator_settings,
    termination_settings,
    output_variables=dependent_variables_to_save_mass)

propagator_settings_list = [translational_propagator_settings, rotational_propagator_settings, mass_propagator_settings]

# Define settings for multi-type propagator
propagator_settings = propagation_setup.propagator.multitype(
    propagator_settings_list,
    integrator_settings,
    simulation_start_epoch,
    termination_settings,
    output_variables=dependent_variables_to_save_trans + dependent_variables_to_save_rot + dependent_variables_to_save_mass)

# Create simulation object and propagate the dynamics
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings
)

# Extract the resulting state history and convert it to an ndarray
states = dynamics_simulator.state_history
states_array = result2array(states)

# For the orbital elements
dep_vars = dynamics_simulator.dependent_variable_history
dep_vars_array = result2array(dep_vars)
print(dep_vars_array.shape)

print(
    f"""
Single Earth-Orbiting Satellite Example.
The initial position vector of Delfi-C3 is [km]: \n{
    states[simulation_start_epoch][:3] / 1E3} 
The initial velocity vector of Delfi-C3 is [km/s]: \n{
    states[simulation_start_epoch][3:6] / 1E3}
\nAfter {simulation_end_epoch} seconds the position vector of Delfi-C3 is [km]: \n{
    states[simulation_end_epoch][:3] / 1E3}
And the velocity vector of Delfi-C3 is [km/s]: \n{
    states[simulation_end_epoch][3:6] / 1E3}
    """
)

# Define a 3D figure using pyplot
fig = plt.figure(figsize=(6, 6), dpi=150)
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Delfi-C3 trajectory around Earth')

# Plot the positional state history
ax.plot(states_array[:, 1] / 1E3, states_array[:, 2] / 1E3, states_array[:, 3] / 1E3, label=bodies_to_propagate[0],
        linestyle='-.')
ax.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue')

# Add the legend and labels, then show the plot
ax.legend()
ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')
plt.show()

# Plot Kepler elements as a function of time
time_hours = dep_vars_array[:, 0] / 3600
kepler_elements = dep_vars_array[:, 1:7]
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(9, 12))
fig.suptitle('Evolution of Kepler elements over the course of the propagation.')

# Semi-major Axis
semi_major_axis = kepler_elements[:, 0] / 1e3
ax1.plot(time_hours, semi_major_axis)
ax1.set_ylabel('Semi-major axis [km]')

# Eccentricity
eccentricity = kepler_elements[:, 1]
ax2.plot(time_hours, eccentricity)
ax2.set_ylabel('Eccentricity [-]')

# Inclination
inclination = np.rad2deg(kepler_elements[:, 2])
ax3.plot(time_hours, inclination)
ax3.set_ylabel('Inclination [deg]')

# Argument of Periapsis
argument_of_periapsis = np.rad2deg(kepler_elements[:, 3])
ax4.plot(time_hours, argument_of_periapsis)
ax4.set_ylabel('Argument of Periapsis [deg]')

# Right Ascension of the Ascending Node
raan = np.rad2deg(kepler_elements[:, 4])
ax5.plot(time_hours, raan)
ax5.set_ylabel('RAAN [deg]')

# True Anomaly
true_anomaly = np.rad2deg(kepler_elements[:, 5])
ax6.scatter(time_hours, true_anomaly, s=1)
ax6.set_ylabel('True Anomaly [deg]')
ax6.set_yticks(np.arange(0, 361, step=60))

for ax in fig.get_axes():
    ax.set_xlabel('Time [hr]')
    ax.set_xlim([min(time_hours), max(time_hours)])
    ax.grid()
plt.tight_layout()

forces = dep_vars_array[:, 7:13] * initial_mass
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(9, 12))
fig.suptitle('Force in different directions over time.')

# X-axis grav
force_x = forces[:, 0]
ax1.plot(time_hours, force_x)
ax1.set_ylabel('Grav. Force x-axis [N]')

# Y-axis grav
force_y = forces[:, 1]
ax3.plot(time_hours, force_y)
ax3.set_ylabel('Grav. Force y-axis [N]')

# Z-axis grav
force_z = forces[:, 2]
ax5.plot(time_hours, force_z)
ax5.set_ylabel('Grav. Force z-axis [N]')

# X-axis input
force_x = forces[:, 3]
ax2.plot(time_hours, force_x)
ax2.set_ylabel('Input Force x-axis [N]')

# Y-axis input
force_y = forces[:, 4]
ax4.plot(time_hours, force_y)
ax4.set_ylabel('Input Force y-axis [N]')

# Z-axis input
force_z = forces[:, 5]
ax6.plot(time_hours, force_z)
ax6.set_ylabel('Input Force z-axis [N]')

for ax in fig.get_axes():
    ax.set_xlabel('Time [hr]')
    ax.set_xlim([min(time_hours), max(time_hours)])
    ax.grid()
plt.tight_layout()

print(np.linalg.norm(forces, axis=1))

# Define a 3D figure using pyplot
fig = plt.figure(figsize=(6, 6), dpi=100)
ax = fig.add_subplot(111)
ax.set_title(f'Mass over time')

# Plot the positional state history
ax.plot(time_hours, dep_vars_array[:, -1])

# Add the legend and labels, then show the plot
# ax.legend()
ax.set_xlabel('time [hr]')
ax.set_ylabel('mass [kg]')
plt.show()
