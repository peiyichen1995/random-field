from gstools import SRF, Exponential, Gaussian
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
from ufl import cofac

def set_fem_fun(vec, fs):
    retval = Function(fs)
    retval.vector().set_local(vec)
    return retval

N = 5
mesh = UnitCubeMesh(N, N, N)

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
F = variable(I + grad(u))             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
I1 = tr(C)
I2 = tr(cofac(C))
I3 = det(C)

eta1 = 141
# eta1 = rF
eta2 = 160
eta3 = 3100
delta = 2*eta1 + 4*eta2 + 2*eta3

# compressible Mooney-Rivlin model
psi = eta1*I1 + eta2*I2 + eta3*I3 - delta*ln(sqrt(I3))

# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds

# Compute first variation of Pi (directional derivative about u in the direction of v)
F_ = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(F_, u, du)

problem = NonlinearVariationalProblem(F_, u, bcs, J)
solver  = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-8
prm['newton_solver']['relative_tolerance'] = 1E-7
prm['newton_solver']['maximum_iterations'] = 100
prm['newton_solver']['relaxation_parameter'] = 1.0
solver.solve()

VVV = TensorFunctionSpace(mesh, 'CG', 1)
PK1_stress = Function(VVV, name='PK1')
PK1 = diff(psi,F)
PK1Project = project(PK1, VVV)
PK1_stress.assign(PK1Project)

with XDMFFile("output/PK1Tensor.xdmf") as infile:
    infile.write_checkpoint(PK1_stress, "PK1", 0)
