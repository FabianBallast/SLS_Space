import numpy as np
from Visualisation.Plotting import plot_3d_position
import matplotlib.pyplot as plt
from tudatpy.kernel.astro import element_conversion
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix

Omegas = np.linspace(-np.pi, np.pi, 60)
thetas = np.linspace(-np.pi, np.pi, 60)

pos = np.zeros((len(Omegas), len(thetas), 3))
problems = np.zeros((len(Omegas), len(thetas)), dtype=bool)

for i, Omega in enumerate(Omegas):
    for j, theta in enumerate(thetas):
        pos[i, j, :] = element_conversion.keplerian_to_cartesian(np.array([55, 0, np.deg2rad(45), 0.1, Omega, theta]), 100)[:3]
        problems[i, j] = np.abs(theta + np.cos(np.deg2rad(45)) * Omega) < np.deg2rad(5)

fig = None
for i in range(len(Omegas)):
    if np.sum(problems[i]) > 0:
        fig = plot_3d_position(pos[i][problems[i]], state_label_name=None, figure=fig, c=np.tile(np.array([1, 0, 0]),
                                                                                                 (np.sum(problems[i]), 1)))
    fig = plot_3d_position(pos[i][~problems[i]], state_label_name=None, figure=fig, c=np.tile(np.array([0, 1, 0]),
                                                                                             (np.sum(~problems[i]), 1)))

plt.figure()
problems_sparse = coo_matrix(problems)
fine_sparse = coo_matrix(~problems)
plt.scatter(thetas[problems_sparse.col], Omegas[problems_sparse.row], c=np.tile(np.array([1, 0, 0]), (np.sum(problems), 1)))
plt.scatter(thetas[fine_sparse.col], Omegas[fine_sparse.row], c=np.tile(np.array([0, 1, 0]), (np.sum(fine_sparse), 1)))
plt.xlabel('Theta')
plt.ylabel('Omega')
plt.show()
