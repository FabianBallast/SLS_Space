import numpy as np
from tudatpy.kernel.astro import element_conversion
import matplotlib.pyplot as plt
from Utils.CollisionAngles import find_collision_angles

mu = 100
orbit_param = [55, 0, np.deg2rad(45)] # a, e, i

Omegas = np.deg2rad(np.linspace(0, 360, num=7, endpoint=False))

theta_arr = np.linspace(0, 2 * np.pi, endpoint=True, num=100)

pos = np.zeros((len(Omegas), len(theta_arr), 3))

for i, plane in enumerate(Omegas):
    for j, theta in enumerate(theta_arr):
        pos[i, j] = element_conversion.keplerian_to_cartesian(np.array(orbit_param + [0, plane, theta]), mu)[:3]

collision_angles = np.zeros((len(Omegas), len(Omegas) - 1))
for i, Omega_i in enumerate(Omegas):
    for j, Omega_j in enumerate(Omegas):
        if i != j:
            collision_angles[i, j - (j > i)] = find_collision_angles(orbit_param[2], Omega_i, Omega_j)[0]

specific_points = np.zeros((len(Omegas), len(Omegas) - 1, 3))
for i, plane in enumerate(Omegas):
    for j in range(len(Omegas) - 1):
        specific_points[i, j] = element_conversion.keplerian_to_cartesian(np.array(orbit_param + [0, plane, collision_angles[i, j]]), mu)[:3]


figure = plt.figure(figsize=(6, 6), dpi=150)
ax = figure.add_subplot(111, projection='3d')
# ax.set_title(f'Satellite trajectories around Earth')
ax.set_xlabel(f'x [m]')
ax.set_ylabel(f'y [m]')
ax.set_zlabel(f'z [m]')
# ax.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue')
ax.set_xlim(-60, 60)
ax.set_ylim(-60, 60)
ax.set_zlim(-60, 60)

for i, plane in enumerate(Omegas):
    ax.plot(pos[i, :, 0], pos[i, :, 1], pos[i, :, 2], 'b-', label=f"Omega={np.round(np.rad2deg(plane),0)}")

for i, plane in enumerate(Omegas):
    for j in range(len(Omegas) - 1):
        ax.plot(specific_points[i, j, 0], specific_points[i, j, 1], specific_points[i, j, 2], 'ro')

# plt.legend()
figure.savefig('orbits_7.eps')
plt.show()
