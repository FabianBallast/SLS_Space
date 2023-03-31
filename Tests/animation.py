import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3

filename = "kepler_5_satellites.npy"
number_of_satellites = 5
simulation_timestep = 0.5  # s

with open(filename, 'rb') as f:
    abs_states = np.load(f)
    abs_states = np.concatenate((abs_states[:, 0:1] * 0, abs_states), axis=1)
    rel_states = np.load(f)
    inputs = np.load(f)



fig = plt.figure(figsize=(6, 6), dpi=150)
ax = p3.Axes3D(fig, auto_add_to_figure=False)  # 3D place for drawing
fig.add_axes(ax)
ax.set_title(f'Satellite trajectories around Earth')

# Plot the positional state history
idx_base = np.arange(0, number_of_satellites) * 6

pos, = ax.plot(abs_states[0, idx_base + 1] / 1E3, abs_states[0, idx_base + 2] / 1E3,
               abs_states[0, idx_base + 3] / 1E3, 'o', label="Satellites")
ax.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue')

# Add the legend and labels, then show the plot
ax.legend()
ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')
ax.set_xlim(-7000, 7000)
ax.set_ylim(-7000, 7000)


def animation_function(i):
    pos.set_xdata(abs_states[i, idx_base + 1] / 1E3)
    pos.set_ydata(abs_states[i, idx_base + 2] / 1E3)
    pos.set_3d_properties(abs_states[i, idx_base + 3] / 1E3)
    return pos,


animation = FuncAnimation(fig,
                          func=animation_function,
                          frames=np.linspace(1, len(abs_states[:, 0]), int(len(abs_states[:, 0]) / 2),
                                             endpoint=False, dtype=int),
                          interval=10,
                          repeat=True,
                          blit=False)

plt.show()
