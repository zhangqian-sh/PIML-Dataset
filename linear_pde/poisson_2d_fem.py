# %% import packages
import matplotlib.pyplot as plt
import numpy as np
from netgen.geom2d import unit_square
from ngsolve import *

# %% create mesh
ngsglobals.msg_level = 1

# generate a triangular mesh of mesh-size 0.01
mesh = Mesh(unit_square.GenerateMesh(maxh=0.01))
h = 1

# H1-conforming finite element space
fes = H1(mesh, order=3, dirichlet=[1, 2, 3, 4])

# define trial- and test-functions
u = fes.TrialFunction()
v = fes.TestFunction()

evaluation_points = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101))
evaluation_points = np.array(evaluation_points).transpose(1, 2, 0).reshape(-1, 2)

# %% assembly
# the bilinear-form
a = BilinearForm(fes, symmetric=True)
sigma = 0.5
k = exp(-0.5 * ((x - 0.5) ** 2 + (y - 0.5) ** 2) / sigma**2)
# a += k**2 * grad(u) * grad(v) * dx
a += grad(u) * grad(v) * dx

# the right hand side
f = LinearForm(fes)
f_rhs = 0
basis = [
    lambda x: 1,
    lambda x: (x - 0.5 * h) / (0.5 * h),
    lambda x: (3 * ((x - 0.5 * h) / (0.5 * h)) ** 2 - 1) / 2,
    lambda x: (5 * ((x - 0.5 * h) / (0.5 * h)) ** 3 - 3 * ((x - 0.5 * h) / (0.5 * h))) / 2,
    lambda x: (35 * ((x - 0.5 * h) / (0.5 * h)) ** 4 - 30 * ((x - 0.5 * h) / (0.5 * h)) ** 2 + 3)
    / 8,
]  # legendre
basis = [
    lambda x: 1,
    lambda x: (x - 0.5 * h) / (0.5 * h),
    lambda x: 2 * ((x - 0.5 * h) / (0.5 * h)) ** 2 - 1,
    lambda x: 4 * ((x - 0.5 * h) / (0.5 * h)) ** 3 - 3 * ((x - 0.5 * h) / (0.5 * h)),
    lambda x: 8 * ((x - 0.5 * h) / (0.5 * h)) ** 4 - 8 * ((x - 0.5 * h) / (0.5 * h)) ** 2 + 1,
]  # 1st Chebyshev
rng = np.random.default_rng(2023)
coef = rng.uniform(-1, 1, size=(5, 5))
# for i in range(5):
#     for j in range(5):
#         f_rhs += coef[i, j] * basis[i](x) * basis[j](y)
# f += f_rhs * v * dx
f += 1 * v * dx

a.Assemble()
f.Assemble()

# %% solve the linear system
gfu = GridFunction(fes, "u")
gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec
# %% visualization
vals = [gfu(mesh(*p)) for p in evaluation_points]
plt.tricontourf(evaluation_points[:, 0], evaluation_points[:, 1], vals, levels=50)
plt.colorbar()

# %%
