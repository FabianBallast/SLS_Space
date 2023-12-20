import matplotlib.pyplot as plt
import os
import pickle


linestyle_dict = {'HCW': {'color': 'orange', 'linestyle': 'dashdot', 'marker': 'None'},
                  'ROE': {'color': 'green', 'linestyle': 'dotted', 'marker': 'None'},
                  'ROE(NO J2)': {'color': 'green', 'linestyle': 'dotted', 'marker': 'None'},
                  'ROE(J2)': {'color': 'green', 'linestyle': 'dashed', 'marker': 'None'},
                  'BLEND': {'color': 'blue', 'linestyle': 'solid', 'marker': 'None'},
                  'BLEND(NO J2)': {'color': 'blue', 'linestyle': 'dotted', 'marker': 'None'},
                  'BLEND(J2)': {'color': 'blue', 'linestyle': 'dashed', 'marker': 'None'}}

satellite_dict = {'single_plane': [1, 3, 9],
                  'double_plane': [4, 6, 8],
                  'hex_plane': [5, 9, 12],
                  'j2_comparison': [0,1,2,3]}

plot_duration = {'single_plane': 60,
                 'double_plane': 150,
                 'hex_plane': 120,
                 'j2_comparison': 24*60}

legend_dict = {'BLEND' : 'Blend',
               'HCW' : 'HCW',
               'ROE' : 'ROE'}


def plot_data_comparison(plot_name: str) -> None:
    """
    Plot data for a given plot name.
    :param plot_name: The name of the plot (with corresponding data names)
    """
    fig_name_list = ['main_states', 'side_states', 'inputs', 'radius_theta', 'inputs_planar']
    fig_list = [None] * len(fig_name_list)

    for file in os.listdir("../Data"):
        if file.startswith(plot_name):
            method = file.removeprefix(plot_name + "_")
            end = file.removeprefix(plot_name)
            type = file.removesuffix(end).removesuffix('_kepler').removesuffix('_j2')
            with open(os.path.join("../Data", file), 'rb') as f:
                orbital_sim = pickle.load(f)
                fig_list[0] = orbital_sim.plot_main_states(figure=fig_list[0], satellite_indices=satellite_dict[type],
                                                           **linestyle_dict[method], legend_name=legend_dict[method],
                                                           plot_duration=plot_duration[type], plot_radial_constraint=False)
                fig_list[1] = orbital_sim.plot_side_states(figure=fig_list[1], satellite_indices=satellite_dict[type],
                                                           **linestyle_dict[method], legend_name=legend_dict[method],
                                                           plot_duration=plot_duration[type])
                # if method == 'HCW':
                #     continue
                fig_list[2] = orbital_sim.plot_inputs(figure=fig_list[2], satellite_indices=satellite_dict[type],
                                                      **linestyle_dict[method], legend_name=legend_dict[method],
                                                      plot_duration=plot_duration[type])

                fig_list[3] = orbital_sim.plot_radius_theta(figure=fig_list[3], satellite_indices=satellite_dict[type],
                                                      **linestyle_dict[method], legend_name=legend_dict[method],
                                                      plot_duration=plot_duration[type])

                fig_list[4] = orbital_sim.plot_planar_inputs(figure=fig_list[4], satellite_indices=satellite_dict[type],
                                                      **linestyle_dict[method], legend_name=legend_dict[method],
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


def plot_j2_comparison() -> None:
    """
    Plot the j2 comparison plot.
    """
    fig = None
    for file in os.listdir("../Data"):
        if file.startswith('j2_comparison'):
            method = file.removeprefix("j2_comparison_")
            with open(os.path.join("../Data", file), 'rb') as f:
                orbital_sim = pickle.load(f)
                x_states = pickle.load(f)

                fig = orbital_sim.plot_model_errors(x_states, figure=fig, legend_name=method, **linestyle_dict[method])

    fig.savefig(f'../Figures/j2_comparison.eps')


def plot_all() -> None:
    """
    Plot and save all plots.
    """
    for file in os.listdir("../Figures"):
        if file.endswith('_inputs.eps'):
            plot_data_comparison(file.removesuffix('_inputs.eps'))
            plot_individual_results(file.removesuffix('_inputs.eps'))

    plot_projection_comparison()


def plot_presentation(plot_name: str) -> None:
    """
    Plot for presentation.

    :param plot_name: The name of the plot (with corresponding data names).
    """
    fig_name_list = ['radius', 'radial_input']
    fig_list = [None] * len(fig_name_list)

    for file in os.listdir("../Data"):
        if file.startswith(plot_name):
            method = file.removeprefix(plot_name + "_")
            end = file.removeprefix(plot_name)
            type = file.removesuffix(end).removesuffix('_kepler').removesuffix('_j2')
            with open(os.path.join("../Data", file), 'rb') as f:
                orbital_sim = pickle.load(f)
                fig_list[0] = orbital_sim.plot_radius(figure=fig_list[0], satellite_indices=[4],
                                                           **linestyle_dict[method], legend_name=legend_dict[method],
                                                           plot_duration=600)

                fig_list[1] = orbital_sim.plot_radial_input(figure=fig_list[1], satellite_indices=[4],
                                                      **linestyle_dict[method], legend_name=legend_dict[method],
                                                      plot_duration=600)


if __name__ == '__main__':
    # plot_data_comparison('single_plane_j2')
    # plot_individual_results('double_plane_j2')
    # plot_projection_comparison()
    # plot_j2_comparison()
    # plot_all()
    plot_presentation('double_plane_j2')
    plt.show()
