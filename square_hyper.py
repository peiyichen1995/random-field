from dolfin import *
import matplotlib.pyplot as plt

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
mesh = UnitSquareMesh(5, 5)
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Mark boundary subdomians
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)

# Define Dirichlet boundary (x = 0 or x = 1)
c = Expression(("0.0", "0.0"), element = V.ufl_element())

bcl = DirichletBC(V, c, left)
bcs = [bcl]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
B  = Constant((0.0, 0.0))  # Body force per unit volume
T  = Constant((0.2, 0.0,))  # Traction force on the boundary

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = variable(F.T*F)                  # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J  = det(F)

# Elasticity parameters
E, nu = 10.0, 0.3
mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(F, u, du)

# Solve variational problem
solve(F == 0, u, bcs, J=J,
      form_compiler_parameters=ffc_options)

plot(u, mode="displacement")
plt.savefig('square_hyper.png')

VVV = TensorFunctionSpace(mesh, 'DG', 1)
defF = I + grad(u)
defFProject = project(defF, VVV)
file = XDMFFile("defFTensor.xdmf")
file.write(defFProject,0)

PK2 = 2.0*diff(psi,C)
PK2Project = project(PK2, VVV)

file = XDMFFile("PK2Tensor.xdmf")
file.write(PK2Project,0)

# Save solution in VTK format
file = File("2d_displacement.pvd");
file << u;
