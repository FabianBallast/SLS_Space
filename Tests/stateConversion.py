import matplotlib.pyplot as plt
import numpy as np

theta_list = np.linspace(0, 2 * np.pi, endpoint=False, num=100)
delta_Omega_list = np.deg2rad([-20, 0, 20])


ix = np.zeros((len(delta_Omega_list), len(theta_list)))
iy = np.zeros_like(ix)

for i, delta_Omega in enumerate(delta_Omega_list):
    for j, theta in enumerate(theta_list):
        ix[i, j] = np.cos(theta) - np.cos(theta + delta_Omega)
        iy[i, j] = np.sin(theta) - np.sin(theta + delta_Omega)

fig, axes = plt.subplots(2, 1, sharex=True)

axes = list(axes)
theta_deg = np.rad2deg(theta_list)
for i, delta_Omega in enumerate(delta_Omega_list):
    axes[0].plot(theta_deg, ix[i], label=f'Omega={np.rad2deg(delta_Omega)}')
axes[0].set_ylabel('ix')
plt.grid(True)

for i, delta_Omega in enumerate(delta_Omega_list):
    axes[1].plot(theta_deg, iy[i], label=f'Omega={np.rad2deg(delta_Omega)}')
axes[1].set_ylabel('iy')
plt.legend()
# plt.grid(True)
plt.show()