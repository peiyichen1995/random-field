from __future__ import division
from gstools import SRF, Exponential
from dolfin import *
import scipy.sparse.linalg as spla
import numpy as np
import scipy.linalg as dla
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from scipy.stats import gamma
from scipy.stats import norm

import math
import ufl
from ufl import cofac

def set_fem_fun(vec, fs):
    retval = Function(fs)
    retval.vector().set_local(vec)
    return retval

N = 10
mesh = UnitSquareMesh(N, N)

xs = mesh.coordinates()[:,0]
ys = mesh.coordinates()[:,1]

model = Exponential(dim=2, var=1, len_scale=[0.5, 0.5], angles=np.pi/4.)
srf = SRF(model, seed=20170519)
randomField = srf([xs, ys])

rF = set_fem_fun(randomField, FunctionSpace(mesh, 'CG', 1))

file = File("output/3D_Random.pvd")
file << rF

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}


V = VectorFunctionSpace(mesh, 'CG',1)

left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)

c = Expression(('0', '0'), element = V.ufl_element())

bcl = DirichletBC(V, c, left)
bcs = [bcl]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration

# Kinematics


d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Body forces
T  = Constant((0.0, 0.0))  # Traction force on the boundary
# B  = Expression(('0.0', '0.0'), element = V.ufl_element())  # Body force per unit volume
B  = Constant((0.0, 0.0))

# Invariants of deformation tensors
I1 = tr(C)
# I2 = 1/2*(tr(C)*tr(C) - tr(C*C))
I2 = tr(cofac(C))
I3 = det(C)

# eta1 = 141
eta1 = 141*rF
eta2 = 160
eta3 = 3100
delta = 2*eta1 + 4*eta2 + 2*eta3

# compressible Mooney-Rivlin model
psi_MR = eta1*I1 + eta2*I2 + eta3*I3 - delta*ln(sqrt(I3))

psi = psi_MR
# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds


# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(F, u, du)

# Solve variational problem

problem = NonlinearVariationalProblem(F, u, bcs, J)
solver  = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-8
prm['newton_solver']['relative_tolerance'] = 1E-7
prm['newton_solver']['maximum_iterations'] = 100
prm['newton_solver']['relaxation_parameter'] = 1.0
solver.solve()

plot(u, mode="displacement")
plt.savefig('figures/square.png')

file = File("output/displacement_rf_2D.pvd")
file << u
