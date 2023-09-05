import osqp
import numpy as np
from scipy import sparse

# Define problem data
P = sparse.csc_matrix([[4, 1], [1, 2]])
q = np.array([1, 1])
A = sparse.csc_matrix([[1, 1], [1, 0], [0, 1]])
l = np.array([1, 0, 0])
u = np.array([1, 0.7, 0.7])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace
prob.setup(P, q, A, l, u, verbose=False)

# Solve problem
res = prob.solve()
print(res.x)
# Update problem
# NB: Update only upper triangular part of P
P_new = sparse.csc_matrix([[5, 1.5], [1.5, 1]])
A_new = sparse.csc_matrix([[1.2, 1], [1, 0], [0, 1.3]])
prob.update(Ax=A_new.data[[0, 3]], Ax_idx=np.array([0, 3]))

# Solve updated problem
res = prob.solve()
print(res.x)

prob = osqp.OSQP()

# Setup workspace
prob.setup(P, q, A_new, l, u, verbose=False)
print(res.x)
