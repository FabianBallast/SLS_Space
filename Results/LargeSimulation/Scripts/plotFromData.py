import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

linestyle_dict = {'nominal': {}}

plot_duration = {'nominal': 35}

legend_dict = {'nominal': 'nominal'}


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
                fig.savefig(f'../Figures/{plot_name}_theta_omega.eps')


def plot_individual_results(plot_name: str) -> None:
    """
    Plot the individual results of the simulation for the appendix.

    :param plot_name: The name of the plot (with corresponding data names).
    """
    fig = None

    for file in os.listdir("../Data"):
        if file.startswith(plot_name):
            method = file.removeprefix(plot_name + "_")
            with open(os.path.join("../Data", file), 'rb') as f:
                orbital_sim = pickle.load(f)
                # fig = orbital_sim.plot_controller_states()
                # fig.savefig(f'../Figures/{plot_name}_states.eps')
                #
                # fig = orbital_sim.plot_inputs()
                # fig.savefig(f'../Figures/{plot_name}_inputs.eps')

                fig = orbital_sim.plot_radius()
                fig.savefig(f'../Figures/{plot_name}_radius.eps')


def plot_constraints(plot_name: str) -> None:
    """
    Plot the theta-Omega plot for the different sats.

    :param plot_name: The name of the plot with corresponding data names.
    """
    for file in os.listdir("../Data"):
        if file.startswith(plot_name):
            with open(os.path.join("../Data", file), 'rb') as f:
                orbital_sim = pickle.load(f)

                fig = orbital_sim.plot_in_plane_constraints()
                fig.savefig(f'../Figures/{plot_name}_in_plane_full.eps')

                fig = orbital_sim.plot_in_plane_constraints(y_lim=[0, 15])
                fig.savefig(f'../Figures/{plot_name}_in_plane_zoom.eps')

                fig = orbital_sim.plot_out_of_plane_constraints()
                fig.savefig(f'../Figures/{plot_name}_out_of_plane_full.eps')

                fig = orbital_sim.plot_out_of_plane_constraints(y_lim=[0, 0.02])
                fig.savefig(f'../Figures/{plot_name}_out_of_plane_zoom.eps')


def plot_all() -> None:
    """
    Plot and save all plots.
    """
    for file in os.listdir("../Data"):
        plot_theta_Omega(file)
        # plot_individual_results(file)
        # plot_constraints(file)


if __name__ == '__main__':
    # plot_theta_Omega('large_sim_nominal_ca_no_noise_no')
    plot_individual_results('large_sim_nominal_ca_yes_noise_no')
    # plot_constraints('large_sim_nominal_ca_no_noise_no')
    # plot_all()
    plt.show()
