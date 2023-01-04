from __future__ import division
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
import pdb


# exponential covariance function
def cov_exp(r, rho, sigma=1.0):
    return sigma * np.exp(-math.pi * r * r / 2.0 / rho / rho)

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

    return w, v, V, mesh, C, M



def set_fem_fun(vec, fs):
    retval = Function(fs, name='gamma')
    retval.vector().set_local(vec)
    return retval



w, v, V, mesh, C, M = solve_covariance_EVP(lambda r : cov_exp(r, rho=0.1, sigma=1.0), N = 21, degree = 1)



idx = w.argsort()[::-1]
w = w[idx]
v = v[:,idx]

# pdb.set_trace()

randomField = np.zeros(v[:, 0].shape)

gauss = np.random.normal(loc=0.0, scale=1.0, size=(len(w), 1))


for i in range(len(w)):
    randomField = randomField + sqrt(w[i]) * v[:,i] * gauss[i]

for i in range(len(w)):

    randomField[i] = norm.cdf(randomField[i])
    print(randomField[i])
    randomField[i] = gamma.ppf(randomField[i],1.9,scale=1.5)

rF = set_fem_fun(randomField, FunctionSpace(mesh, 'CG', 1))

vtkfile = File('output/gamma.pvd')
vtkfile << rF

plt.figure()
im = plot(rF)
plt.colorbar(im)
plt.title("Non-Gaussian Field")
plt.savefig('Non-Gaussian-Field.png')
# plt.show()
