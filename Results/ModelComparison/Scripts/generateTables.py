import os
import pickle
import numpy as np

table_name_legend = {'BLEND': 'Blend',
              'HCW': 'HCW',
              'ROE': 'ROE',
                     "BLEND(J2)": 'Blend(J2)',
                     'BLEND(NO J2)': 'Blend(No J2)',
                     'ROE(J2)': "ROE(J2)",
                     "ROE(NO J2)": "ROE(No J2)"}

def generate_table(table_name: str) -> None:
    """
    Generate table for a given table name.
    :param table_name: The name of the table (with corresponding data names)
    """
    table_data = {}

    for file in os.listdir("../Data"):
        if file.startswith(table_name):
            method = file.removeprefix(table_name + "_")
            with open(os.path.join("../Data", file), 'rb') as f:
                orbital_sim = pickle.load(f)
                table_data[method] = orbital_sim.find_metric_values()
                number_of_iterations = int(np.ceil(orbital_sim.scenario.simulation.simulation_duration / orbital_sim.scenario.control.control_timestep)) + 1
                table_data[method] += [orbital_sim.solver_time / number_of_iterations]

    with open(f"..\\Tables\\{table_name}.txt", "w") as text_file:
        print("\\begin{table}[!hbt]\n\\centering", file=text_file)
        print(f"\\caption{{Metric during}}\n\\label{{tab: met_{table_name}_orbit_mech}}", file=text_file)
        print("\\begin{tabular}{c| c c c c c c c c} \n Model & $\\Bar{|r|}$ [m] & $\\Bar{|\\theta|}$ [deg] & $\\Bar{|\\Omega|}$ [deg] & "
              "$\\Bar{|u_r|}$ [N]& $\\Bar{|u_t|}$ [N]& $\\Bar{|u_n|}$ [N]& $\\Bar{\\norm{\\mathbf{u}}}_2$ [N] & $T_\\mathrm{sol}$ [s]\\\\ \\hline", file=text_file)

        for method in table_data:
            print(f"{table_name_legend[method]} & {'&'.join(f'{x:.2g}' for x in table_data[method])} \\\\", file=text_file)

        print("\\end{tabular}\n\\end{table}", file=text_file)


def generate_small_table(table_name: str) -> None:
    """
    Generate table for a given table name.
    :param table_name: The name of the table (with corresponding data names)
    """
    table_data = {}

    for file in os.listdir("../Data"):
        if file.startswith(table_name):
            method = file.removeprefix(table_name + "_")
            with open(os.path.join("../Data", file), 'rb') as f:
                orbital_sim = pickle.load(f)
                table_data[method] = orbital_sim.find_metric_values()
                number_of_iterations = int(np.ceil(orbital_sim.scenario.simulation.simulation_duration / orbital_sim.scenario.control.control_timestep)) + 1
                table_data[method] += [orbital_sim.solver_time / number_of_iterations]

    data_array = np.zeros((len(table_data), 4))
    for idx, method in enumerate(table_data):
        data_array[idx, 0:2] = table_data[method][6]
        data_array[idx, 2:] = table_data[method][-1]

    data_array[:, 1] /= np.min(data_array[:, 1])
    data_array[:, 3] /= np.min(data_array[:, 3])


    with open(f"..\\Tables\\{table_name}.txt", "w") as text_file:
        print("\\begin{table}[!hbt]\n\\centering", file=text_file)
        print(f"\\caption{{Metric during}}\n\\label{{tab: met_{table_name}_orbit_mech}}", file=text_file)
        print("\\begin{tabular}{|c| c c | c c|} \\hline \n \\rule{0pt}{3ex} Model & $\\Bar{\\norm{\\mathbf{u}}}_2$ [N] & $\\Bar{\\norm{\\mathbf{u}}}_2^\\mathrm{norm}$ [-] & $T_\\mathrm{sol}$ [s] & $T_\\mathrm{sol}^\\mathrm{norm}$ [-]\\\\ \\hline", file=text_file)

        for idx, method in enumerate(table_data):
            print(f"{table_name_legend[method]} & {'&'.join(f'{x:.4f}' for x in data_array[idx])} \\\\", file=text_file)

        print("\\hline \\end{tabular}\n\\end{table}", file=text_file)


def generate_all() -> None:
    """
    Generate all tables
    """
    for file in os.listdir("../Figures"):
        if file.endswith('_inputs.eps'):
            generate_table(file.removesuffix('_inputs.eps'))


if __name__ == '__main__':
    # plot_data('double_plane_kepler')
    # generate_all()
    generate_small_table('hex_plane_j2')
    # plt.show()
