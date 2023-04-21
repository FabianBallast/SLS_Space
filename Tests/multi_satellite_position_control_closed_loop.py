import numpy as np
from matplotlib import pyplot as plt
from Space.OrbitalMechanics import OrbitalMechSimulator
from SLS.SLS_setup import SLSSetup
from Dynamics.HCWDynamics import RelCylHCW
from Dynamics.ROEDynamics import QuasiROE

# Global parameters
satellite_mass = 400  # kg
inertia_tensor = 0.01 * np.eye(3)
control_timestep = 50  # s
orbital_height = 750  # km
number_of_satellites = 5  # -
simulation_duration = 1  # hours
simulation_timestep = 1  # s

# Create prediction using HCW model
# First, create system basics
control_model = QuasiROE(orbital_height=orbital_height*1000, satellite_mass=satellite_mass)
# control_model = RelCylHCW(orbital_height=orbital_height*1000, satellite_mass=satellite_mass)
sls_setup = SLSSetup(sampling_time=control_timestep, system_dynamics=control_model, tFIR=10)
sls_setup.create_system(number_of_systems=number_of_satellites)

# Add cost matrices
sls_setup.create_cost_matrices()

# Create x0 and x_ref
sls_setup.create_x0(number_of_dropouts=1, seed=129)
sls_setup.create_reference()

# print(sls_setup.x0)
# print(sls_setup.x_ref)

# Create loop
t_horizon_control = int(np.ceil(simulation_duration * 3600 / control_timestep)) + 1
control_simulation_ratio = int(control_timestep / simulation_timestep)
t_horizon_simulation = control_simulation_ratio * t_horizon_control
rel_states = np.zeros((t_horizon_simulation + 1, len(sls_setup.x0)))
rel_states[0:1] = sls_setup.x0.T

abs_states = np.zeros((t_horizon_simulation + 1, (number_of_satellites + 1) * 6))  # Also include reference
dep_vars = None
dep_var_dict = None

inputs = np.zeros((number_of_satellites, 3, t_horizon_control))

progress = 0
for t in range(t_horizon_control):
    if t / t_horizon_control * 100 > progress:
        print(f"Progress: {int(t / t_horizon_control * 100)}%")
        progress = int(t / t_horizon_control * 100) + 1

    # Start simulation
    sls_setup.set_initial_conditions(rel_states[t*control_simulation_ratio:t*control_simulation_ratio+1].T)
    sls_setup.simulate_system(t_horizon=1, noise=None, inputs_to_store=1)
    control_inputs = sls_setup.u_inputs.reshape((number_of_satellites, 3, -1))

    # if np.linalg.norm(control_inputs[0, :]) < 0.1:
    #     raise Exception("No (significant) input provided for the first satellite. ")
    inputs[:, :, t] = control_inputs[:, :, 0]

    # Create new orbital sim
    orbital_sim = OrbitalMechSimulator()
    orbital_sim.create_bodies(number_of_satellites=number_of_satellites,
                              satellite_mass=satellite_mass,
                              satellite_inertia=inertia_tensor,
                              add_reference_satellite=True)

    # Create thruster models
    specific_impulse = 1E10  # Really high to make mass constant
    orbital_sim.create_engine_models_thrust(control_timestep=control_timestep,
                                            thrust_inputs=control_inputs,
                                            specific_impulse=specific_impulse)

    # Add forces to the satellites.
    orbital_sim.create_acceleration_model(order_of_zonal_harmonics=0)

    # Add mass rate model to the satellites
    orbital_sim.create_mass_rate_model()

    # Set the initial position of the satellites
    if t == 0:
        true_anomalies = sls_setup.get_relative_angles()[:, 0]
        abs_states[t:t+1] = orbital_sim.convert_orbital_elements_to_cartesian(true_anomalies=true_anomalies,
                                                                              orbit_height=orbital_height,
                                                                              inclination=45)
    orbital_sim.set_initial_position(abs_states[t*control_simulation_ratio:t*control_simulation_ratio+1].T)

    # Add dependent variables to track
    orbital_sim.set_dependent_variables_translation(add_keplerian_state=True,
                                                    add_thrust_accel=True,
                                                    add_rsw_rotation_matrix=True)
    orbital_sim.set_dependent_variables_mass(add_mass=True)

    # Create propagation settings
    start_epoch = t * control_timestep
    end_epoch = (t + 1) * control_timestep
    orbital_sim.create_propagation_settings(start_epoch=start_epoch,
                                            end_epoch=end_epoch,
                                            simulation_step_size=simulation_timestep)

    # Simulate system
    orbital_sim.simulate_system()
    states_array, dep_vars_array = orbital_sim.translation_states, orbital_sim.dep_vars
    abs_states[t*control_simulation_ratio+1:(t+1)*control_simulation_ratio+1] = states_array[1:, :]

    if dep_vars is None:
        dep_vars = np.zeros((t_horizon_simulation + 1, dep_vars_array.shape[1]))
        dep_vars[0] = dep_vars_array[0]
        dep_var_dict = orbital_sim.dependent_variables_dict
    dep_vars[t*control_simulation_ratio+1:(t+1)*control_simulation_ratio+1] = dep_vars_array[1:]

    # Find states for SLS model
    control_states = orbital_sim.get_states_for_dynamical_model(control_model)
    rel_states[t*control_simulation_ratio+1:(t+1)*control_simulation_ratio+1] = \
        control_states[1:control_simulation_ratio+1]

    # Unwrap to prevent weird position
    rel_states[t*control_simulation_ratio:(t+1)*control_simulation_ratio + 1, sls_setup.angle_states] = \
        np.unwrap(rel_states[t*control_simulation_ratio:(t+1)*control_simulation_ratio + 1, sls_setup.angle_states],
                  axis=0)

# Plot results
orbital_sim = OrbitalMechSimulator()
orbital_sim.number_of_controlled_satellites = number_of_satellites
orbital_sim.number_of_total_satellites = number_of_satellites + 1
orbital_sim.simulation_timestep = simulation_timestep
orbital_sim.translation_states = abs_states  # np.concatenate((abs_states[:, 0:1] * 0, abs_states), axis=1)
orbital_sim.dep_vars = dep_vars
orbital_sim.dependent_variables_dict = dep_var_dict
orbital_sim.controlled_satellite_names = ["Satellite_0", "Satellite_1", "Satellite_2", "Satellite_3", "Satellite_4"]
orbital_sim.satellite_mass = satellite_mass

orbital_sim.plot_cylindrical_states(reference_angles=sls_setup.x_ref[sls_setup.angle_states])
orbital_sim.plot_quasi_roe_states()
input_fig = orbital_sim.plot_thrusts()  # Fix first
# anim = orbital_sim.create_animation()
plt.show()


# # Possibly save results
# with open('kepler_5_satellites.npy', 'wb') as f:
#     np.save(f, abs_states)
#     np.save(f, rel_states)
#     np.save(f, inputs)
