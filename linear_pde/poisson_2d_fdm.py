# %% import packages
import matplotlib.pyplot as plt
import numpy as np


# %% solve 2d poisson equation
def solve_poisson(L: int, c: float, boundary_condition: float):
    """
    Solve -\Delta u = c with Dirichlet BC in [-1, 1] x [-1, 1]
    """
    N = L**2
    A = np.zeros([N, N])  # LHS matrix
    b = np.zeros(N)  # RHS BC

    h = 2 / L
    # interior
    for i in range(1, L - 1):
        for j in range(1, L - 1):
            A[i * L + j, i * L + j] = 4
            A[i * L + j, i * L + j - 1], A[i * L + j, i * L + j + 1] = -1, -1
            A[i * L + j, (i - 1) * L + j], A[i * L + j, (i + 1) * L + j] = -1, -1
    # boundary
    for j in range(1, L - 1):
        A[j, j] = 4
        A[j, j - 1], A[j, j + 1], A[j, j + L] = -1, -1, -1
        b[j] = 1
    for j in range(N - L, N - 1):
        A[j, j] = 4
        A[j, j - 1], A[j, j + 1], A[j, j - L] = -1, -1, -1
        b[j] = 1
    for i in range(1, L - 1):
        A[i * L, i * L] = 4
        A[i * L, (i - 1) * L], A[i * L, (i + 1) * L], A[i * L, i * L + 1] = -1, -1, -1
        b[i * L] = 1
    for i in range(1, L - 1):
        A[(i + 1) * L - 1, (i + 1) * L - 1] = 4
        (
            A[(i + 1) * L - 1, i * L - 1],
            A[(i + 1) * L - 1, (i + 2) * L - 1],
            A[(i + 1) * L - 1, (i + 1) * L - 2],
        ) = (-1, -1, -1)
        b[(i + 1) * L - 1] = 1
    # corner
    A[0, 0], A[0, 1], A[0, L] = 4, -1, -1
    A[L - 1, L - 1], A[L - 1, L - 2], A[L - 1, 2 * L - 1] = 4, -1, -1
    A[N - L, N - L], A[N - L, N - L + 1], A[N - L, N - 2 * L] = 4, -1, -1
    A[N - 1, N - 1], A[N - 1, N - 2], A[N - 1, N - L - 1] = 4, -1, -1
    b[0] = b[L - 1] = b[N - L] = b[N - 1] = 2

    # solve linear system
    A *= 1 / h**2
    b *= 1 / h**2 * boundary_condition
    b = c + b
    u = np.linalg.solve(A, b)

    return u.reshape(L, -1)


# %% example
L = 20
c = 1
boundary_condition = 0
u = solve_poisson(L, c, boundary_condition)
plt.imshow(u, cmap="jet")
