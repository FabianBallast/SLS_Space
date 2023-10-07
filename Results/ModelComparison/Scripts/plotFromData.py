import matplotlib.pyplot as plt
import os
import pickle


linestyle_dict = {'HCW': {'color': 'red', 'linestyle': 'dashed', 'marker': 'None'},
                  'ROE': {'color': 'green', 'linestyle': 'dotted', 'marker': 'None'},
                  'BLEND': {'color': 'blue', 'linestyle': 'solid', 'marker': 'None'}}

satellite_dict = {'single_plane': [1, 3, 9],
                  'double_plane': [4, 6, 8],
                  'hex_plane': [5, 9, 12]}

plot_duration = {'single_plane': 45,
                 'double_plane': 70,
                 'hex_plane': 60}


def plot_data_comparison(plot_name: str) -> None:
    """
    Plot data for a given plot name.
    :param plot_name: The name of the plot (with corresponding data names)
    """
    fig_name_list = ['main_states', 'side_states', 'inputs']
    fig_list = [None] * len(fig_name_list)

    for file in os.listdir("../Data"):
        if file.startswith(plot_name):
            method = file.removeprefix(plot_name + "_")
            end = file.removeprefix(plot_name)
            type = file.removesuffix(end).removesuffix('_kepler').removesuffix('_j2')
            with open(os.path.join("../Data", file), 'rb') as f:
                orbital_sim = pickle.load(f)
                fig_list[0] = orbital_sim.plot_main_states(figure=fig_list[0], satellite_indices=satellite_dict[type],
                                                           **linestyle_dict[method], legend_name=method,
                                                           plot_duration=plot_duration[type])
                fig_list[1] = orbital_sim.plot_side_states(figure=fig_list[1], satellite_indices=satellite_dict[type],
                                                           **linestyle_dict[method], legend_name=method,
                                                           plot_duration=plot_duration[type])
                fig_list[2] = orbital_sim.plot_inputs(figure=fig_list[2], satellite_indices=satellite_dict[type],
                                                      **linestyle_dict[method], legend_name=method,
                                                      plot_duration=plot_duration[type])

    for idx, fig in enumerate(fig_list):
        fig.savefig('../Figures/' + plot_name + '_' + fig_name_list[idx] + '.eps')


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
                fig = orbital_sim.plot_controller_states()
                fig.savefig(f'../Figures/{plot_name}_states_{method}.eps')

                fig = orbital_sim.plot_inputs()
                fig.savefig(f'../Figures/{plot_name}_inputs_{method}.eps')


def plot_projection_comparison() -> None:
    """
    Plot the projection comparison plot.
    """
    for file in os.listdir("../Data"):
        if file.startswith('projection'):
            with open(os.path.join("../Data", file), 'rb') as f:
                orbital_sim = pickle.load(f)
                fig = orbital_sim.plot_3d_orbit_projection()
                fig.savefig(f'../Figures/projection_comparison.eps', bbox_inches='tight')

                fig2 = orbital_sim.plot_relative_radius_and_height()
                fig2.savefig(f'../Figures/projection_comparison_states.eps')


def plot_all() -> None:
    """
    Plot and save all plots.
    """
    for file in os.listdir("../Figures"):
        if file.endswith('_inputs.eps'):
            plot_data_comparison(file.removesuffix('_inputs.eps'))
            plot_individual_results(file.removesuffix('_inputs.eps'))

    plot_projection_comparison()


if __name__ == '__main__':
    # plot_data_comparison('double_plane_kepler')
    # plot_individual_results('single_plane_kepler')
    # plot_projection_comparison()
    plot_all()
    # plt.show()
