from __future__ import division
from dolfin import *
import scipy.sparse.linalg as spla
import numpy as np
import scipy.linalg as dla
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from scipy.stats import gamma
from scipy.stats import norm

import ufl


# exponential covariance function
def cov_exp(r, rho, sigma2=1.0):
    return sigma2 * np.exp(-r*r/2.0/rho/rho)

def solve_covariance_EVP(cov, N, degree=1):
    def setup_FEM(N):
        mesh = UnitSquareMesh(N,N)
        V = FunctionSpace(mesh, 'CG', degree)
        u = TrialFunction(V)
        v = TestFunction(V)
        return mesh, V, u, v
    # construct FEM space
    mesh, V, u, v = setup_FEM(N)

    # dof to vertex map
    dof2vert = dof_to_vertex_map(V)
    # coords will be used for interpolation of covariance kernel
    coords = mesh.coordinates()
    # but we need degree of freedom ordering of coordinates
    coords = coords[dof2vert]
    # assemble mass matrix and convert to scipy
    M = assemble(u*v*dx)
    M = M.array()


    # evaluate covariance matrix
    L = coords.shape[0]
    C = np.zeros([L,L])

    for i in range(L):
        for j in range(L):
            if j <= i:
                v = cov(np.linalg.norm(coords[i]-coords[j]))
                C[i,j] = v
                C[j,i] = v

    # solve eigenvalue problem
    A = np.dot(M, np.dot(C, M))

    w, v = dla.eigh(A, b=M)


    return w, v, V, mesh, coords



def set_fem_fun(vec, fs):
    retval = Function(fs)
    retval.vector().set_local(vec)
    return retval



w, v, V, mesh, coords = solve_covariance_EVP(lambda r : cov_exp(r, rho=0.1, sigma2=1.0), N = 21, degree = 1)

idx = w.argsort()[::-1]
w = w[idx]
v = v[:,idx]

randomField = np.zeros(v[:, 0].shape)

gauss = np.random.normal(loc=0.0, scale=1.0, size=(len(w), 1))


for i in range(len(w)):
    randomField = randomField + sqrt(w[i]) * v[:,i] * gauss[i]

for i in range(len(w)):
    randomField[i] = norm.cdf(randomField[i])
    randomField[i] = gamma.ppf(randomField[i],1.9,scale=1.5)

rF = set_fem_fun(randomField, FunctionSpace(mesh, 'CG', 1))


plt.figure()
im = plot(rF)
plt.colorbar(im)
plt.title("Non-Gaussian Field")
plt.savefig('Non-Gaussian-Field.png')
plt.show()


## start for model

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
#mesh = UnitSquareMesh(10, 10)
V = VectorFunctionSpace(mesh, 'CG',1)

# Mark boundary subdomians
bottom =  CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)
top = CompiledSubDomain("near(x[1], side) && on_boundary", side = 1.0)
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)

# Define Dirichlet boundary (x = 0 or x = 1)
c_b = Expression(('0', '0'), element = V.ufl_element())

bc_b = DirichletBC(V, c_b, bottom)
bcs = [bc_b]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
# C = F.T*F                   # Right Cauchy-Green tensor
C = variable(F.T*F)

T  = Constant((0.0, 0.0))  # Traction force on the boundary

# Invariants of deformation tensors
I1 = tr(C)
I2 = 1/2*(tr(C)*tr(C) - tr(C*C))
I3 = det(C)

#eta1 = 141
eta1 = 14

eta2 = 16
eta3 = 31
delta = 2*eta1 + 4*eta2 + 2*eta3


# compressible Mooney-Rivlin model
psi_MR = eta1*I1 + eta2*I2 + eta3*I3 - delta*ln(sqrt(I3))

psi = psi_MR
# Total potential energy
Pi = psi*dx - dot(T, u)*ds

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
prm['newton_solver']['maximum_iterations'] = 1000
prm['newton_solver']['relaxation_parameter'] = 1.0
solver.solve()


file = File("displacement_rf.pvd")
file << u

plot(u, mode="displacement")
plt.savefig('random_square.png')


# VVV = TensorFunctionSpace(mesh, 'DG', 1)
# defF = I + grad(u)
# defFProject = project(defF, VVV)
#
# #
# PK2 = 2.0*diff(psi,C)
# PK2Project = project(PK2, VVV)

# import pdb
# pdb.set_trace()

# file = XDMFFile("PK2Tensor.xdmf")
# file.write(PK2Project,0)
#
# file = XDMFFile("defFTensor.xdmf")
# file.write(defF,0)
