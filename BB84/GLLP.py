import math
import numpy as np
import scipy.special as sp
from matplotlib import pyplot as plt
from pylab import mpl

# Set Chinese display font
mpl.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

start = 0 / 0.2
Dis0 = 34 / 0.2
Step0 = 0.2

q = 0.5  # Probability of taking the correct basis
e0 = 0.5  # error rate when the number of photons is 0

f = 1  # Error correction efficiency
p_d = 2 * 10 ** -6  # Dark count rate of one detector
e_det = 0.015  # Optics induced error rate
alpha = 0.2  # Loss of fiber
eta_d = 0.15  # Detection efficiency
eta_b = 0.4  # Bob's transmittance

mu_values = [0.1, 0.05, 0.01, 0.005, 0.001]
mu = 0.001
nu = 0.2
nuV = 0.1

FER = 0.05
Overhead = 0.5

# Binary entropy function
def H2(x):
    if x > 0 and x < 1:
        return -x * math.log2(x) - (1 - x) * math.log2(1 - x)
    else:
        return 0

matDis = np.arange(start, Dis0, Step0)
R_GLLP = np.zeros(len(matDis))

def GLLP(L, FER, Overhead, q, f, e0, p_d, e_det, alpha, eta_d, eta_b, mu):
    eta_B = eta_b * eta_d
    Y0 = 2 * p_d - p_d ** 2
    eta_AB = 10 ** (-alpha * L / 10)
    eta = eta_AB * eta_B
    Q_mu = Y0 + 1 - math.exp(-eta * mu)
    E_mu = (e0 * Y0 + e_det * (1 - math.exp(-eta * mu))) / Q_mu  # Qubit error rate
    P_multi = 1 - math.exp(-mu) - mu * math.exp(-mu)
    Delta1 = 1 - P_multi / Q_mu
    e1 = E_mu / Delta1
    R = q * Q_mu * (-f * H2(E_mu) + Delta1 * (1 - H2(e1)))
    return (1 - FER) * (1 - Overhead) * R

# GLLP
if __name__ == "__main__":
    # code to be executed if the script is run directly
    print("Run GLLP.py directly")

    for mu in mu_values:
        R_GLLP = np.zeros(len(matDis))
        count = 0
        for L in matDis:
            R_GLLP[count] = GLLP(L, FER, Overhead, q, f, e0, p_d, e_det, alpha, eta_d, eta_b, mu)
            count += 1
        plt.semilogy(matDis, R_GLLP, label=f'μ={mu}')
    plt.xlabel('Transmission Distance (km)')
    plt.ylabel('Secure Key Rate (bit/pulse)')
    plt.title('GLLP Analysis of BB84 Protocol for Non-ideal Light Sources')
    plt.xlim(0, 50)
    plt.ylim(1e-7, 1e-3)
    plt.legend(loc='upper right')
    plt.show()
else:
    # code to be executed when imported
    print("GLLP.py imported")