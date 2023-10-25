import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams["mathtext.fontset"] = 'cm'

data_time_to_plot = 3

with open(f'../Data/lumped_data_{data_time_to_plot}.npy', 'rb') as f:
    states_lumped = np.load(f)
    lim_x_lumped = np.load(f)
    inputs_lumped = np.load(f)
    lim_u_lumped = np.load(f)

with open(f'../Data/nominal_data_{data_time_to_plot}.npy', 'rb') as f:
    states_nominal = np.load(f)
    lim_x_nominal = np.load(f)
    inputs_nominal = np.load(f)
    lim_u_nominal = np.load(f)


fig, _ = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
axes = list(fig.get_axes())

is_angle_list = [False, False]
y_label_list = [r'$\delta r\mathrm{\;[m]}$',  r'$\delta e_x^f \mathrm{\;[-]}$']

states2plot = [0, 2]

for idx, axes in enumerate(axes):
    idx_state = states2plot[idx]
    state_lumped = states_lumped[:, idx_state]
    constraint_lumped = lim_x_lumped[:, idx_state]

    state_nominal = states_nominal[:, idx_state]
    constraint_nominal = lim_x_nominal[:, idx_state]

    # If it is an angle, unwrap and convert to deg
    if is_angle_list[idx]:
        state_lumped = np.rad2deg(state_lumped)
        constraint_lumped = np.rad2deg(constraint_lumped)

        state_nominal = np.rad2deg(state_nominal)
        constraint_nominal = np.rad2deg(constraint_nominal)

    # Plot data
    axes.plot(state_lumped, 'b', label='state lumped SLS')
    axes.plot(state_nominal, 'r', label='state nominal SLS')

    axes.plot(constraint_lumped, 'b--', label='constraint lumped SLS')
    # print(state, constraint)
    axes.plot(-constraint_lumped, 'b--')


    axes.plot(constraint_nominal, 'r--', label='constraint nominal SLS')
    # print(state, constraint)
    axes.plot(-constraint_nominal, 'r--')

    if idx in [1]:
        axes.set_xlabel(r'$\mathrm{Prediction \;Step\;[-]}$', fontsize=14)
    axes.set_ylabel(y_label_list[idx], fontsize=14)
    axes.set_xlim([0, 6])
    axes.grid(True)

    # if legend_names[state_idx] is not None:
    axes.legend(fontsize=12, loc='upper right')

plt.tight_layout()

fig.savefig(f'../Figures/states_comparison_lumped_{data_time_to_plot}.eps')
fig.savefig(f'../Figures/states_comparison_lumped_{data_time_to_plot}.png')

fig, _ = plt.subplots(1, 1, figsize=(16, 3), sharex=True)
axes = list(fig.get_axes())

y_label_list = [r'$u_t\mathrm{\;[N]}$']
inputs2plot = [1]
for idx, axes in enumerate(axes):
    idx_state = inputs2plot[idx]
    input_lumped = inputs_lumped[:, idx_state]
    constraint_lumped = lim_u_lumped[:, idx_state]

    input_nominal = inputs_nominal[:, idx_state]
    constraint_nominal = lim_u_nominal[:, idx_state]


    # Plot data
    axes.plot(input_lumped, 'b', label='input lumped SLS')
    axes.plot(input_nominal, 'r', label='input nominal SLS')
    axes.plot(constraint_lumped, 'b--', label='constraint lumped SLS')
    # print(state, constraint)
    axes.plot(-constraint_lumped, 'b--')


    axes.plot(constraint_nominal, 'r--', label='constraint nominal SLS')
    # print(state, constraint)
    axes.plot(-constraint_nominal, 'r--')

    if idx in [0]:
        axes.set_xlabel(r'$\mathrm{Prediction \;Step\;[-]}$', fontsize=14)
    axes.set_ylabel(y_label_list[idx], fontsize=14)
    axes.set_xlim([0, 5])
    axes.grid(True)

    # if legend_names[state_idx] is not None:
    axes.legend(fontsize=12, loc='upper right')

plt.tight_layout()

fig.savefig(f'../Figures/input_comparison_lumped_{data_time_to_plot}.eps')
fig.savefig(f'../Figures/input_comparison_lumped_{data_time_to_plot}.png')

plt.show()
