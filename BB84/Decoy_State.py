import math
import numpy as np
from matplotlib import pyplot as plt
from cmath import sqrt, log
from numpy.linalg import det
from numpy import transpose
from scipy.special import comb
from pylab import mpl

# Set the font to display Chinese characters
mpl.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

start = 0 / 0.2
Dis0 = 34 / 0.2
Step0 = 0.2

q = 0.5  # Probability of taking the correct basis
e0 = 0.5  # error rate when the number of photons is 0

f = 1  # Error correction efficiency
p_d = 2 * 10 ** -6  # Dark count rate of one detector
e_det = 0.01  # Optics induced error rate
alpha = 0.2  # Loss of fiber
eta_d = 0.15  # Detection efficiency
eta_b = 0.4  # Bob's transmittance

mu = 0.1
nu = 0.05
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
R_DecoyInf = np.zeros(len(matDis))
R_Decoysingle = np.zeros(len(matDis))
R_Decoy_weak_plus_vacuum = np.zeros(len(matDis))

def DecoyInf(L, FER, Overhead, q, f, e0, p_d, e_det, alpha, eta_d, eta_b, mu):
    eta_B = eta_b * eta_d
    Y0 = 2 * p_d - p_d ** 2
    eta_AB = 10 ** (-alpha * L / 10)
    eta = eta_AB * eta_B
    Q_mu = Y0 + 1 - math.exp(-eta * mu)
    E_mu = (e0 * Y0 + e_det * (1 - math.exp(-eta * mu))) / Q_mu  # Qubit error rate
    Deltal = (mu * math.exp(-mu) / Q_mu) * (eta + Y0 - eta * Y0)
    e1 = (e0 * Y0 + e_det * eta) / (eta + Y0 - eta * Y0)
    R = q * Q_mu * (-f * H2(E_mu) + Deltal * (1 - H2(e1)))
    return (1 - FER) * (1 - Overhead) * R

def Decoysingle(L, FER, Overhead, q, f, e0, p_d, e_det, alpha, eta_d, eta_b, mu, nu):
    eta_B = eta_b * eta_d
    Y0 = 2 * p_d - p_d ** 2
    eta_AB = 10 ** (-alpha * L / 10)
    eta = eta_AB * eta_B
    Q_mu = Y0 + 1 - math.exp(-eta * mu)
    Q_nu = Y0 + 1 - math.exp(-eta * nu)
    E_mu = (e0 * Y0 + e_det * (1 - math.exp(-eta * mu))) / Q_mu  # Qubit error rate
    E_nu = (e0 * Y0 + e_det * (1 - math.exp(-eta * nu))) / Q_nu
    Y1L = (mu / (mu * nu - nu ** 2)) * (Q_nu * math.exp(nu) - nu ** 2 * Q_mu * math.exp(mu) / mu ** 2 - (mu ** 2 - nu ** 2) * E_mu * math.exp(mu) * Q_mu / (mu ** 2 * e0))
    Delta1 = Y1L * mu * math.exp(-mu) / Q_mu
    e1 = (E_mu * Q_mu * math.exp(mu) - E_nu * Q_nu * math.exp(nu)) / (Y1L * (mu - nu))
    R = q * Q_mu * (-f * H2(E_mu) + Delta1 * (1 - H2(e1)))
    return (1 - FER) * (1 - Overhead) * R

def Decoy_weak_plus_vacuum(L, FER, Overhead, q, f, e0, p_d, e_det, alpha, eta_d, eta_b, mu, nu):
    eta_B = eta_b * eta_d
    Y0 = 2 * p_d - p_d ** 2
    eta_AB = 10 ** (-alpha * L / 10)
    eta = eta_AB * eta_B
    Q_mu = Y0 + 1 - math.exp(-eta * mu)
    Q_nu = Y0 + 1 - math.exp(-eta * nu)
    E_mu = (e0 * Y0 + e_det * (1 - math.exp(-eta * mu))) / Q_mu  # Qubit error rate
    E_nu = (e0 * Y0 + e_det * (1 - math.exp(-eta * nu))) / Q_nu
    Delta1 = mu ** 2 * math.exp(-mu) * ((Q_nu * math.exp(nu)) - (nu ** 2 * Q_mu * math.exp(mu) / mu ** 2) - ((mu ** 2 - nu ** 2) * Y0 / mu ** 2)) / (Q_mu * (mu * nu - nu ** 2))
    e1 = (E_nu * Q_nu * math.exp(nu) - e0 * Y0) * mu * math.exp(-mu) / (Q_mu * nu * Delta1)
    R = q * Q_mu * (-f * H2(E_mu) + Delta1 * (1 - H2(e1)))
    return (1 - FER) * (1 - Overhead) * R

def GLLP(L, FER, Overhead, q, f, e0, p_d, e_det, alpha, eta_d, eta_b, mu):
    eta_B = eta_b * eta_d
    Y0 = 2 * p_d - p_d ** 2
    eta_AB = 10 ** (-alpha * L / 10)
    eta = eta_AB * eta_B
    Q_mu = Y0 + 1 - math.exp(-eta * mu)
    E_mu = (e0 * Y0 + e_det * (1 - math.exp(-eta * mu))) / Q_mu  # Qubit error rate
    P_multi = 1 - math.exp(-mu) - mu * math.exp(-mu)
    Deltal = 1 - P_multi / Q_mu
    e1 = E_mu / Deltal
    R = q * Q_mu * (-f * H2(E_mu) + Deltal * (1 - H2(e1)))
    return (1 - FER) * (1 - Overhead) * R

if __name__ == "__main__":
    count = 0

    for L in matDis:
        R_GLLP[count] = GLLP(L, FER, Overhead, q, f, e0, p_d, e_det, alpha, eta_d, eta_b, mu)
        R_DecoyInf[count] = DecoyInf(L, FER, Overhead, q, f, e0, p_d, e_det, alpha, eta_d, eta_b, mu)
        R_Decoysingle[count] = Decoysingle(L, FER, Overhead, q, f, e0, p_d, e_det, alpha, eta_d, eta_b, mu, nu)
        R_Decoy_weak_plus_vacuum[count] = Decoy_weak_plus_vacuum(L, FER, Overhead, q, f, e0, p_d, e_det, alpha, eta_d, eta_b, mu, nu)
        count += 1

    plt.semilogy(matDis, R_GLLP, label='GLLP', color='tab:blue', linestyle='solid')
    plt.semilogy(matDis, R_Decoysingle, label='Single Decoy State', color='tab:orange', linestyle='solid')
    plt.semilogy(matDis, R_Decoy_weak_plus_vacuum, label='Weak + Vacuum Decoy State', color='lime', linestyle='solid')
    plt.semilogy(matDis, R_DecoyInf, label='Infinite Decoy State', color='m', linestyle='dashed')

    plt.xlabel('Transmission Distance (km)')
    plt.ylabel('Secure Key Rate (bit/pulse)')
    plt.title('Decoy State Protocol Analysis of the BB84 Protocol for Non-Ideal Light Sources')
    plt.xlim(0, 140)
    plt.ylim(1e-7, 1e-2)
    plt.legend(loc='upper right')
    plt.show()

else:
    # code to be executed when imported
    print("Decoy.py imported")