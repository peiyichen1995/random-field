import fenics as fe
import matplotlib.pyplot as plt
import numpy as np


# --------------------
# Functions and classes
# --------------------
# Bottom boundary
def bottom(x, on_boundary):
    return (on_boundary and fe.near(x[1], 0.0))

# Strain function
def epsilon(u):
    return fe.sym(fe.grad(u))

# Stress function
def sigma(u):
    return lambda_*fe.div(u)*fe.Identity(2) + 2*mu*epsilon(u)
# lambda is a reserved python keyword, naming convention recommends
# using a single trailing underscore for such cases

model = "plane_strain"

if model == "plane_stress":
    lambda_ = 2*mu*lambda_/(lambda_+2*mu)

mesh = fe.RectangleMesh(fe.Point(0.0, 0.0), fe.Point(l_x, l_y), n_x, n_y)

fe.plot(mesh)
plt.savefig('mesh.png')

# --------------------
# Parameters
# --------------------

# Density
rho = fe.Constant(200.0)

# Young's modulus and Poisson's ratio
E = 0.02e9
nu = 0.0

# Lame's constants
lambda_ = E*nu/(1+nu)/(1-2*nu)
mu = E/2/(1+nu)

l_x, l_y = 5.0, 5.0  # Domain dimensions
n_x, n_y = 20, 20  # Number of elements

# Load
g_z = -2.9575e5
b_z = -10.0
g = fe.Constant((0.0, g_z))
b = fe.Constant((0.0, b_z))

# --------------------
# Function spaces
# --------------------
V = fe.VectorFunctionSpace(mesh, "CG", 1)
u_tr = fe.TrialFunction(V)
u_test = fe.TestFunction(V)

# --------------------
# Boundary conditions
# --------------------
bc = fe.DirichletBC(V, fe.Constant((0.0, 0.0)), bottom)

top = fe.AutoSubDomain(lambda x: fe.near(x[1], l_y))

# Definition of Neumann boundary condition domain
boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)

top.mark(boundaries, 1)
ds = fe.ds(subdomain_data=boundaries)

# --------------------
# Weak form
# --------------------
a = fe.inner(sigma(u_tr), epsilon(u_test))*fe.dx
l = rho*fe.dot(b, u_test)*fe.dx + fe.inner(g, u_test)*ds(1)


# --------------------
# Solver
# --------------------
u = fe.Function(V)
A, L = fe.assemble_system(a, l, bc)

problem = fe.LinearVariationalProblem(a,l,u,bc)
solver = fe.LinearVariationalSolver(problem)

fe.solve(A, u.vector(), L)


fe.plot(u, mode="displacement")
plt.savefig('displacement.png')
