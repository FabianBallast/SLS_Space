import osqp
import numpy as np
from scipy import sparse

# Define problem data
P = 2 * sparse.csc_matrix([1 + 1e-8])
q = np.array([0])
A = sparse.csc_matrix([1])
l = np.array([1])
u = np.array([100])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace
prob.setup(P, q, A, l, u, eps_rel=1e-10, eps_abs=1e-10, verbose=True)

# Solve problem
res = prob.solve()
print(res.info.obj_val)
print(res.x[0])
# Update problem
# NB: Update only upper triangular part of P
# P_new = sparse.csc_matrix([[5, 1.5], [1.5, 1]])
# A_new = sparse.csc_matrix([[1.2, 1], [1, 0], [0, 1.3]])
# prob.update(Ax=A_new.data[[0, 3]], Ax_idx=np.array([0, 3]))
#
# # Solve updated problem
# res = prob.solve()
# print(res.x)
#
# prob = osqp.OSQP()
#
# # Setup workspace
# prob.setup(P, q, A_new, l, u, verbose=False)
# print(res.x)
