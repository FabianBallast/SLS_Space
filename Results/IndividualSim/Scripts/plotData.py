import matplotlib.pyplot as plt
import os
import pickle
import numpy as np



linestyle_dict = {'safe': {'color': 'red', 'linestyle': 'dashed', 'marker': 'None'},
                  'simple': {'color': 'blue', 'linestyle': 'solid', 'marker': 'None'}}

plot_duration = {'Omega_lim_full': 390,
                 'Omega_lim_part': 110,
                 'roe_comparison': 60}

satellite_selection = {'Omega_lim_full': np.linspace(0, 23, dtype=int).tolist(),
                       'Omega_lim_part': [8, 9, 10, 11, 12, 13, 14, 15, 16],
                       'roe_comparison': [1, 3, 9]}

legend_dict = {'nominal': 'nominal',
               'safe': 'safe',
               'simple': 'naive'}


def plot_theta_Omega(plot_name: str) -> None:
    """
    Plot the theta-Omega plot for the different sats.

    :param plot_name: The name of the plot with corresponding data names.
    """
    for file in os.listdir("../Data"):
        if file.startswith(plot_name):
            with open(os.path.join("../Data", file), 'rb') as f:
                orbital_sim = pickle.load(f)
                fig = orbital_sim.plot_theta_Omega()
                # orbital_sim.plot_out_of_plane_constraints()
                # orbital_sim.plot_in_plane_constraints()
                fig.savefig(f'../Figures/{plot_name}_theta_omega_full.eps')

def plot_main_states(plot_name: str) -> None:
    """
    Plot the main states.
    :param plot_name: Name of the scenario to plot.
    """
    for file in os.listdir("../Data"):
        if file.startswith(plot_name):
            with open(os.path.join("../Data", file), 'rb') as f:
                orbital_sim = pickle.load(f)
                fig = orbital_sim.plot_main_states_theta_Omega(satellite_indices=satellite_selection['Omega_lim_full'],
                                                   plot_duration=plot_duration['Omega_lim_full'])
                # orbital_sim.plot_out_of_plane_constraints()
                # orbital_sim.plot_in_plane_constraints()
                fig.savefig(f'../Figures/{plot_name}_main_states_full.eps')

                fig = orbital_sim.plot_main_states_theta_Omega(satellite_indices=satellite_selection['Omega_lim_part'],
                                                   plot_duration=plot_duration['Omega_lim_part'])
                # orbital_sim.plot_out_of_plane_constraints()
                # orbital_sim.plot_in_plane_constraints()
                fig.savefig(f'../Figures/{plot_name}_main_states_part.eps')

def plot_radius_theta(plot_name: str) -> None:

    fig = None
    for file in os.listdir("../Data"):
        if file.startswith(plot_name):
            method = file.removeprefix(plot_name + "_")
            with open(os.path.join("../Data", file), 'rb') as f:
                orbital_sim = pickle.load(f)
                fig = orbital_sim.plot_radius_theta(satellite_indices=satellite_selection[plot_name],
                                                    plot_duration=plot_duration[plot_name],
                                                    **linestyle_dict[method],
                                                    legend_name=legend_dict[method], figure=fig)

                fig.savefig(f'../Figures/{plot_name}_radius_theta.eps')

def plot_all() -> None:
    """
    Plot and save all plots.
    """
    for file in os.listdir("../Data"):
        # plot_theta_Omega(file)
        plot_main_states(file)
        # plot_constraints(file)


if __name__ == '__main__':
    # plot_theta_Omega('Omega_lim_part')
    # plot_individual_results('large_sim_nominal_ca_yes_noise_no')
    # plot_constraints('large_sim_nominal_ca_no_noise_no')
    # plot_radius_theta('roe_comparison')
    plot_main_states("blend_model_Omega_lim_all")
    # plot_all()
    plt.show()
