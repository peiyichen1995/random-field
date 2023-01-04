from gstools import SRF, Exponential, Gaussian
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

def set_fem_fun(vec, fs):
    retval = Function(fs)
    retval.vector().set_local(vec)
    return retval

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
l_x = 1.0
l_y = 1.0
n_x = 5
n_y = 5
# mesh = RectangleMesh(Point(0.0, 0.0), Point(l_x, l_y), n_x, n_y)
mesh = UnitSquareMesh(30, 30)
V = VectorFunctionSpace(mesh, "CG", 1)

# plot(mesh)
# plt.savefig('mesh.png')
# plt.close()

xs = mesh.coordinates()[:,0]
ys = mesh.coordinates()[:,1]

# model = Exponential(dim=2, var=1, len_scale=[0.5, 0.5], angles=np.pi/2.)
model = Gaussian(dim=2, var=1, len_scale=0.5)
srf = SRF(model, seed=20170519)
randomField = srf([xs, ys])

rF = set_fem_fun(randomField, FunctionSpace(mesh, 'CG', 1))

srf.plot()
plt.savefig('figures/Gaussian_var_1_len_0.5.png')
plt.show()
plt.close()

file = File("2D_Random.pvd")
file << rF

# Mark boundary subdomians
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)

# Define Dirichlet boundary (x = 0 or x = 1)
c = Expression(("0.0", "0.0"), element = V.ufl_element())
extention = Expression(("0.5", "0.0"), element = V.ufl_element())

bcl = DirichletBC(V, c, left)
bcr = DirichletBC(V, extention, right)
bcs = [bcl, bcr]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
B  = Constant((0.0, 0.0))  # Body force per unit volume
T  = Constant((0.0, 0.0))  # Traction force on the boundary

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = variable(F.T*F)                  # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J  = det(F)

# Elasticity parameters
mu = 10
lmbda = 0.3

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(F, u, du)

# Solve variational problem
# solve(F == 0, u, bcs, J=J,
#       form_compiler_parameters=ffc_options)

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
plt.close()

VVV = TensorFunctionSpace(mesh, 'DG', 1)
defF = I + grad(u)
defFProject = project(defF, VVV)
file = XDMFFile("output/defFTensor.xdmf")
file.write(defFProject,0)

PK2 = 2.0*diff(psi,C)
PK2Project = project(PK2, VVV)

file = XDMFFile("output/PK2Tensor.xdmf")
file.write(PK2Project,0)

# Save solution in VTK format
file = File("output/2d_displacement.pvd");
file << u;
