from scipy import sparse, linalg
import numpy as np


def get_masks(A_matrix: sparse.coo_matrix, B_matrix: sparse.coo_matrix) -> (np.ndarray, np.ndarray):
    """
    Find the masks used for the computations.

    :param A_matrix: The A matrix of the system.
    :param B_matrix: The B matrix of the system.
    :return: A tuple with the A and B mask
    """
    mask_A_temp = (A_matrix != 0) ** 6
    mask_B = np.array(((B_matrix.T @ mask_A_temp) != 0).todense())
    mask_A = np.array((mask_A_temp + B_matrix @ mask_B) != 0)

    return mask_A, mask_B


def get_sigma_matrix(mask_A: np.ndarray) -> np.ndarray:
    """
    Find the sigma matrix such that phi_x = sigma_matrix @ sigma <=> Phi_x = Sigma

    :param mask_A: Mask of the A matrix
    :return: sigma matrix
    """
    mask_sparse = sparse.coo_matrix(mask_A)
    mask_sparse.data = np.arange(0, np.sum(mask_A))
    sigma_indices = np.array(mask_sparse.todense().T, dtype=int)[np.eye(mask_A.shape[0]) > 0]
    sigma_sparse = sparse.coo_matrix((np.ones(mask_A.shape[0]), (sigma_indices, np.arange(mask_A.shape[0], dtype=int))))

    return np.array(sigma_sparse.todense())


def sparse_state_matrix_replacement(A_matrix: sparse.coo_matrix, B_matrix: sparse.coo_matrix, mask_A: np.ndarray,
                                    mask_B: np.ndarray) -> (sparse.coo_matrix, sparse.coo_matrix):
    """
    Create a matrix to move from Phi_x[k+1] = A * Phi_x[k] + B * Phi_u[k] to phi_x[k+1] = A_new * phi_x[k] + B_new * phi_u[k],
    with phi_x being the vector form of the non-zero elements in Phi_x (and phi_u similarly).

    :param A_matrix: The A matrix of the system
    :param B_matrix: The B matrix of the system
    :param mask_A: Mask of the A matrix.
    :param mask_B: Mask of the B matrix.
    :return: A_new: new A_matrix for new vector and B_new: new B_matrix for the new vector
    """
    A_dense = np.array(A_matrix.todense())
    B_dense = np.array(B_matrix.todense())

    mask_B_dense = B_dense != 0

    rows, columns = mask_A.shape
    row_indices_A = np.arange(0, rows)
    row_indices_B = np.arange(0, mask_B.shape[0])

    blk_diag_list_A = []
    blk_diag_list_B = []

    for i in range(columns):
        row_selection = row_indices_A[mask_A[:, i]]
        blk_diag_list_A.append(A_dense[row_selection][:, mask_A[:, i]])

        column_selection = row_indices_B[mask_B[:, i]]
        row_selection = np.any(mask_B_dense[:, column_selection], axis=1)
        blk_diag_list_B.append(B_dense[:, column_selection][row_selection])

    return sparse.coo_matrix(linalg.block_diag(*blk_diag_list_A)), sparse.coo_matrix(
        linalg.block_diag(*blk_diag_list_B))


def find_fx_and_fu(mask_A: np.ndarray, mask_B: np.ndarray, x0: np.ndarray) -> (sparse.coo_matrix, sparse.coo_matrix):
    """
    Find Fx and Fu such that Phi_x x == Fx phi_x and Phi_u x == F_u phi_u

    :param mask_A: Mask of the A matrix
    :param mask_B: Mask of the B matrix
    :param x0: Current initial state
    :return: Fx and Fu
    """
    order_matrix_A = sparse.coo_matrix(mask_A.T)
    order_matrix_A.data = np.arange(np.sum(mask_A))
    order_matrix_A = np.array(order_matrix_A.todense()).T

    order_matrix_B = sparse.coo_matrix(mask_B.T)
    order_matrix_B.data = np.arange(np.sum(mask_B))
    order_matrix_B = np.array(order_matrix_B.todense()).T

    # print(order_matrix_A, order_matrix_B)

    F_x = np.zeros((mask_A.shape[0], np.sum(mask_A)))
    F_u = np.zeros((mask_B.shape[0], np.sum(mask_B)))

    for i in range(F_x.shape[0]):
        F_x[i, order_matrix_A[i, mask_A[i, :]]] = x0[mask_A[i, :]]

    for i in range(F_u.shape[0]):
        F_u[i, order_matrix_B[i, mask_B[i, :]]] = x0[mask_B[i, :]]

    return sparse.coo_matrix(F_x), sparse.coo_matrix(F_u)


def update_fx_and_fu(Fx: sparse.csc_matrix, Fu: sparse.csc_matrix, x0: np.ndarray,
                     sparse: bool = True) -> (np.ndarray, np.ndarray):
    """
    Find the data values for the full sparse Fx and Fu matrices.

    :param Fx: Fx matrix.
    :param Fu: Fu matrix.
    :param x0: Current state.
    :param sparse: Whether the data is sparse
    :return: Tuple with data values for Fx and Fu.
    """

    mask_row = np.ones(x0.shape, bool)
    mask_row[2::6] = False
    mask_row[5::6] = False

    data_selection_fx = np.ones(len(Fx.data), bool)
    data_selection_fx[8::10] = False
    data_selection_fx[9::10] = False

    data_selection_fu = np.ones(len(Fu.data), bool)
    data_selection_fu[4::5] = False

    new_data_fx = np.zeros_like(Fx.data)
    new_data_fu = np.zeros_like(Fu.data)

    if sparse:
        new_data_fx[data_selection_fx] = np.kron(x0[mask_row], np.ones(4))
        new_data_fx[~data_selection_fx] = np.kron(x0[~mask_row], np.ones(2))

        new_data_fu[data_selection_fu] = np.kron(x0[mask_row], np.ones(2))
        new_data_fu[~data_selection_fu] = x0[~mask_row]
    else:
        new_data_fx = np.kron(x0, np.ones(x0.shape[0]))
        new_data_fu = np.kron(x0, np.ones(len(Fu.data) // x0.shape[0]))

    return new_data_fx, new_data_fu


def find_indices(problem: dict, number_of_blocks: int, x_vars: int, u_vars: int) -> dict:
    """
    Find the indices when using with OSQP.

    :param problem: The problem dict.
    :param number_of_blocks: Number of blocks in the closed-loop transfer matrix.
    :param x_vars: Number of non-zero elements in Phi_x
    :param u_vars: Number of non-zero elements in Phi_u

    :return: Dict with the indices of each subvector.
    """
    indices_dict = {'x': np.arange(problem['nx'] * problem['N'])}
    indices_dict['u'] = np.arange(indices_dict['x'][-1] + 1,
                                  indices_dict['x'][-1] + 1 + problem['nu'] * problem['N'])
    indices_dict['phi_x'] = np.arange(indices_dict['u'][-1] + 1,
                                      indices_dict['u'][-1] + 1 + x_vars * number_of_blocks)
    indices_dict['phi_u'] = np.arange(indices_dict['phi_x'][-1] + 1,
                                      indices_dict['phi_x'][-1] + 1 + u_vars * number_of_blocks)
    indices_dict['sigma'] = np.arange(indices_dict['phi_u'][-1] + 1,
                                      indices_dict['phi_u'][-1] + 1 + problem['N'] * problem['nx'])

    indices_dict['phi_x_pos'] = np.arange(indices_dict['sigma'][-1] + 1,
                                          indices_dict['sigma'][-1] + 1 + x_vars * number_of_blocks)
    indices_dict['phi_x_neg'] = np.arange(indices_dict['phi_x_pos'][-1] + 1,
                                          indices_dict['phi_x_pos'][-1] + 1 + x_vars * number_of_blocks)
    indices_dict['phi_x_abs'] = np.arange(indices_dict['phi_x_neg'][-1] + 1,
                                          indices_dict['phi_x_neg'][-1] + 1 + x_vars * number_of_blocks)

    indices_dict['phi_u_pos'] = np.arange(indices_dict['phi_x_abs'][-1] + 1,
                                          indices_dict['phi_x_abs'][-1] + 1 + u_vars * number_of_blocks)
    indices_dict['phi_u_neg'] = np.arange(indices_dict['phi_u_pos'][-1] + 1,
                                          indices_dict['phi_u_pos'][-1] + 1 + u_vars * number_of_blocks)
    indices_dict['phi_u_abs'] = np.arange(indices_dict['phi_u_neg'][-1] + 1,
                                          indices_dict['phi_u_neg'][-1] + 1 + u_vars * number_of_blocks)

    indices_dict['x_pos'] = np.arange(indices_dict['phi_u_abs'][-1] + 1,
                                      indices_dict['phi_u_abs'][-1] + 1 + problem['nx'] * problem['N'])
    indices_dict['x_neg'] = np.arange(indices_dict['x_pos'][-1] + 1,
                                      indices_dict['x_pos'][-1] + 1 + problem['nx'] * problem['N'])
    indices_dict['x_abs'] = np.arange(indices_dict['x_neg'][-1] + 1,
                                      indices_dict['x_neg'][-1] + 1 + problem['nx'] * problem['N'])

    indices_dict['u_pos'] = np.arange(indices_dict['x_abs'][-1] + 1,
                                      indices_dict['x_abs'][-1] + 1 + problem['nu'] * problem['N'])
    indices_dict['u_neg'] = np.arange(indices_dict['u_pos'][-1] + 1,
                                      indices_dict['u_pos'][-1] + 1 + problem['nu'] * problem['N'])
    indices_dict['u_abs'] = np.arange(indices_dict['u_neg'][-1] + 1,
                                      indices_dict['u_neg'][-1] + 1 + problem['nu'] * problem['N'])

    indices_dict['x_max'] = np.arange(indices_dict['u_abs'][-1] + 1,
                                      indices_dict['u_abs'][-1] + 1 + problem['N'])
    indices_dict['phi_x_max'] = np.arange(indices_dict['x_max'][-1] + 1,
                                          indices_dict['x_max'][-1] + 1 + number_of_blocks)

    indices_dict['u_max'] = np.arange(indices_dict['phi_x_max'][-1] + 1,
                                      indices_dict['phi_x_max'][-1] + 1 + problem['N'])
    indices_dict['phi_u_max'] = np.arange(indices_dict['u_max'][-1] + 1,
                                          indices_dict['u_max'][-1] + 1 + number_of_blocks)

    return indices_dict


if __name__ == '__main__':
    import control as ct

    A_matrix = np.array([[0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1],
                         [3 * 0.0245 ** 2, 0, 0, 0, 2 * 55 * 0.0245, 0],
                         [0, 0, 0, -2 * 0.0245 / 55, 0, 0],
                         [0, 0, -0.0245 ** 2, 0, 0, 0]])

    B_matrix = np.array([[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0],
                         [1, 0, 0],
                         [0, 1 / 55, 0],
                         [0, 0, 1]]) / 400

    system_continuous = ct.ss(A_matrix, B_matrix, np.eye(6), 0)

    # Find discrete system
    system_discrete = ct.sample_system(system_continuous, 20)

    A_test = sparse.coo_matrix(system_discrete.A)
    B_test = sparse.coo_matrix(system_discrete.B)
    x0 = np.arange(12)

    mask_A, mask_B = get_masks(sparse.block_diag([A_test, A_test]), sparse.block_diag([B_test, B_test]))
    # A_new, B_new = sparse_state_matrix_replacement(A_test, B_test, mask_A, mask_B)
    Fx, Fu = find_fx_and_fu(mask_A, mask_B, x0)

    # sparse_state_matrix_replacement(A_test, B_test, x0)
