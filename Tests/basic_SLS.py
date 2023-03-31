from matplotlib import pyplot as plt

from SLS.SLS_setup import SLSSetup
import numpy as np

# Create system basics
sls_setup = SLSSetup(sampling_time=10)
sls_setup.create_system(number_of_systems=3, orbital_height=750, satellite_mass=400)

# Add cost matrices
Q_matrix_sqrt = np.array([4, 4, 4, 2, 2, 2])
R_matrix_sqrt = 1e-5 * 1 * np.array([[0, 0, 0],
                                     [0, 0, 0],
                                     [0, 0, 0],
                                     [1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])
sls_setup.create_cost_matrices(Q_matrix_sqrt=Q_matrix_sqrt, R_matrix_sqrt=R_matrix_sqrt)

# Create x0 and x_ref
sls_setup.create_spaced_x0(number_of_dropouts=1, seed=129, add_small_velocity=False)
sls_setup.create_reference()

# Start simulation
sls_setup.simulate_system(t_horizon=50, t_FIR=15, noise=None)

objective_value = sls_setup.find_latest_objective_value()
print(f"Latest cost: {objective_value}")
print(f"Input: {sls_setup.u_inputs.shape}")
print(f"Final states: {sls_setup.x_states[:, -1]}")

sls_setup.plot_latest_results()
plt.show()
