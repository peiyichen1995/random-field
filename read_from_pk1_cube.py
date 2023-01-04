from gstools import SRF, Exponential, Gaussian
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
from ufl import cofac


# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

N = 5
mesh = UnitCubeMesh(N, N, N)

V = VectorFunctionSpace(mesh, 'CG',1)
VVV = TensorFunctionSpace(mesh, 'CG', 1)

left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)
top =  CompiledSubDomain("near(x[1], side) && on_boundary", side = 1.0)
bottom = CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)

c = Expression(("0.0", "0.0", "0.0"), element = V.ufl_element())
extention = Expression(("0.1", "0.0", "0.0"), element = V.ufl_element())

bcl = DirichletBC(V, c, left)
bcr = DirichletBC(V, extention, right)
bcs = [bcl, bcr]

du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
B  = Constant((0.0, 0.0, 0.0))  # Body force per unit volume
T  = Constant((0.0, 0.0, 0.0))  # Traction force on the boundary


d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = variable(F.T*F)                   # Right Cauchy-Green tensor

# PK2_stress = Function(VVV, name='PK2')
# defF = Function(VVV, name='defF')
#
# #
# with XDMFFile("output/PK2_defF_Tensor.xdmf") as infile:
#     infile.read_checkpoint(PK2_stress, "PK2")
#     infile.read_checkpoint(defF, "defF")
#
# # use pk1, F after solving
# P = F * PK2_stress


PK1_stress = Function(VVV, name='PK1')

with XDMFFile("output/PK1Tensor.xdmf") as infile:
    infile.read_checkpoint(PK1_stress, "PK1")

P = PK1_stress

# Total potential energy
Pi = inner(P, grad(u))*dx - dot(B, u)*dx - dot(T, u)*ds


# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(F, u, du)

problem = NonlinearVariationalProblem(F, u, bcs, J)
solver  = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-8
prm['newton_solver']['relative_tolerance'] = 1E-7
prm['newton_solver']['maximum_iterations'] = 100
prm['newton_solver']['relaxation_parameter'] = 1.0
solver.solve()

plot(u, mode="displacement")
plt.savefig('figures/square_read_pk2.png')
plt.close()
