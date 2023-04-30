import numpy as np
import os

# np.random.seed(0)
np.random.seed(42)

def cov_exp(r, rho, sigma=1.0):
    return sigma * np.exp(-math.pi * r * r / 2.0 / rho / rho)

def cov_len(rho, sigma=1.0):
    return integrate.quad(lambda r: cov_exp(r, rho), 0, math.inf)

n_sample = 100

path = 'scene_1_suplement.txt'

if (os.path.isfile(path)):
    os.remove(path)

# length of covariance of different rhos
rhos = np.arange(1e-5, 1, 1e-5)
lcs = []
for i in rhos:
    lcs.append(abs(cov_len(i)[0])

with open(path, 'a') as the_file:
    for i in range(n_sample):
        cv = np.random.uniform(low=0.1, high=0.3)
        mean = np.random.uniform(low=40000*0.8, high=40000*1.2)
        # target length of covariance
        lc = np.random.uniform(low=0.2, high=0.4)

        var = (cv * mean)**2
        diff = 1000
        rho = 0
        for j in len(lcs):
            if (abs(lcs[j] - lc) < diff):
                rho = rhos[j]
                diff = abs(lcs[j] - lc)

        ptint('target: {:}, actual: {:}'.format(lc, cov_len(rho)))

        the_file.write(str(mean) + " " + str(var) + " " + str(lc) + " " + str(rho) + '\n')
