# sample usage
# python non_gaussian_fields_given_etas.py 64 40000 4000000 etas_5.csv
# script name, N, mean, variance, file name of etas



from dolfin import *
import meshio
import numpy as np
import math
import scipy.linalg as dla
from scipy.stats import gamma
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

import sys
import csv

import pdb

def cov_exp(r, rho, sigma=1.0):
    return sigma * np.exp(-math.pi * r * r / 2.0 / rho / rho)

def solve_covariance_EVP(cov, mesh, V):
    u = TrialFunction(V)
    v = TestFunction(V)

    # dof to vertex map
    dof2vert = dof_to_vertex_map(V)
    # coords will be used for interpolation of covariance kernel
    coords = mesh.coordinates()
    # but we need degree of freedom ordering of coordinates
    coords = coords[dof2vert]

    # assemble mass matrix and convert to scipy
    M = assemble(u * v * dx)
    M = M.array()

    # evaluate covariance matrix
    L = coords.shape[0]
    C = np.zeros([L, L])

    for i in range(L):
        for j in range(L):
            if j <= i:
                v = cov(np.linalg.norm(coords[i] - coords[j]))
                C[i, j] = v
                C[j, i] = v

    # solve eigenvalue problem
    A = np.dot(M, np.dot(C, M))

    # w, v = spla.eigsh(A, k, M)
    w, v = dla.eigh(A, b=M)

    return w, v, C, M, coords

# order eigenvalues and eigen Vectors
def order_eig(w, v):
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:, idx]
    return w, v

def trun_order(err, C, M, w):
    e = 0
    eig = 0
    trCM = np.trace(np.dot(C, M))
    while 1 - eig / trCM > err:
        eig = eig + w[e]
        e = e + 1
    error = (1 - eig / trCM)
    return e, error

def nonGauss(w, v, etas, a, loc, scale):

    randomField = np.zeros(v[:, 0].shape)
    for i in range(len(etas)):
        randomField = randomField + sqrt(w[i]) * v[:, i] * etas[i]
    # pdb.set_trace()
    for i in range(len(w)):
        randomField[i] = norm.cdf(randomField[i], loc=0.0,scale=1.0)
        randomField[i] = gamma.ppf(randomField[i], a, loc=loc, scale=scale)
    return randomField

def set_fem_fun(vec, fs):
    retval = Function(fs, name='gamma')
    retval.vector().set_local(vec)
    return retval

N = int(sys.argv[1])
mean = float(sys.argv[2])
var = float(sys.argv[3])
start_sample = 0
df = pd.read_csv(sys.argv[4], header=None)
etas = df.to_numpy()
end_sample = etas.shape[1]

# square mesh
mesh = UnitSquareMesh(N,N)

# function space
V = FunctionSpace(mesh, 'CG', 1)

print("Generating Gaussian random filed...")
w, v, C, M, coords = solve_covariance_EVP(
    lambda r: cov_exp(r, rho=0.28295, sigma=1.0), mesh, V)

w, v = order_eig(w, v)

# for i in range(10):
#     mode = set_fem_fun(v[:,i], V)
#     plt.figure()
#     im = plot(mode)
#     plt.title("mode_" + str(i))
#     plt.savefig('figures/modes/mode_' + str(i) + '.png')
#     plt.show()



# import pdb
# pdb.set_trace()

loc = 0
scale = var / mean
a = mean / scale


for i in range(start_sample, end_sample):
    sample = str(i)
    print("Generating non-Gaussian random filed {:}...".format(i))
    randomField = nonGauss(w, v, etas[:, i], a, loc, scale)

    # pdb.set_trace()

    rF = set_fem_fun(randomField, FunctionSpace(mesh, 'CG', 1))


    print("Saving non-Gaussian random filed sample: " + sample + "...")
    plt.figure()
    im = plot(rF)
    plt.colorbar(im)
    plt.title("Non-Gaussian Field")
    plt.savefig('figures_given_etas/2D/NonGaussianField_' + sample + '.png')

    print("Saving non-Gaussian random filed sample: " + sample + " (3D surface)...")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(coords[:, 0], coords[:, 1], randomField, cmap=cm.jet, linewidth=0)
    plt.title("Non-Gaussian Field (3D surface)")
    plt.savefig('figures_given_etas/3D/NonGaussianField3DSurf_' + sample + '.png')

    print("Saving non-Gaussian random filed sample: " + sample + " into pvd...")
    vtkfile = File('output_given_etas/randomField/gamma_' + sample + '.pvd')
    vtkfile << rF
