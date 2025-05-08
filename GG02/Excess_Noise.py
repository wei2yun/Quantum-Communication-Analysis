import numpy as np
import scipy as sc
import scipy.linalg
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]  # Set font to SimHei
plt.rcParams["axes.unicode_minus"] = False  # Resolve issue with negative sign display on axes

# Define global parameters
V_A = 999
beta = 1
alpha = 0.2
V = V_A + 1
# e = 0.3

# Todo: 1. Construct the gamma_AB_0 matrix before entering the quantum channel

# Calculate variances for A and B as two-dimensional identity matrices
va_A_1 = va_B_1 = np.eye(2) * V
sigma_z = np.array([[1, 0], [0, -1]])
# Calculate covariance for AB
cov_AB_0 = np.sqrt(V ** 2 - 1) * sigma_z
# Concatenate by rows
gamma_AB_0_1 = np.concatenate((va_A_1, cov_AB_0), axis=1)
gamma_AB_0_2 = np.concatenate((cov_AB_0, va_B_1), axis=1)
# Concatenate by columns
# gamma_AB_0 = np.concatenate((gamma_AB_0_1, gamma_AB_0_2), axis=0)
gamma_AB_0 = np.array([[1, 0], [0, 1]])


# Todo: 2. Define the g(x) function and the secure key rate function K(L)

# Define g(x) function
def g(x):
    return (x + 1) * np.log2(x + 1) - x * np.log2(x)

# Define secure key rate K function
def K(L, V, e, id):
    """
    Squeezed State Zero Dispersion -> 1
    Coherent State Zero Dispersion -> 2
    Squeezed State External Dispersion -> 3
    Coherent State External Dispersion -> 4
    """
    # Generate the gamma_AB matrix after entering the quantum channel
    T = 10 ** (- alpha * L / 10)
    chi = (1 - T) / T + e
    cov_AB = np.sqrt((V ** 2 - 1) * T) * sigma_z
    sigma_AB = cov_AB
    va_A_2 = va_A_1
    va_B_2 = T * (V + chi) * np.eye(2)
    gamma_AB_1 = np.concatenate((va_A_2, cov_AB), axis=1)
    gamma_AB_2 = np.concatenate((cov_AB, va_B_2), axis=1)
    gamma_AB = np.concatenate((gamma_AB_1, gamma_AB_2), axis=0)

    X = np.array([[1, 0], [0, 0]])

    # Calculate the pseudo-inverse matrix
    H = sc.linalg.pinv(np.dot(np.dot(X, va_B_2), X))
    gamma_X_BA = va_A_2 - np.dot(np.dot(sigma_AB.T, H), sigma_AB)

    # Different protocols have different formulas for I_AB and S_BE
    # Squeezed State Zero Dispersion
    I_AB1 = 0.5 * np.log2((V + chi) / (chi + 1 / V))
    lamda1 = np.sqrt(V * (V * chi + 1) / (V + chi))
    S_BE1 = sum(g((get_sym_val(gamma_AB) - 1) / 2)) - g((lamda1 - 1) / 2)

    # Coherent State Zero Dispersion
    I_AB2 = 0.5 * np.log2((V + chi) / (chi + 1))
    lamda2 = np.sqrt(V * (V * chi + 1) / (V + chi))
    S_BE2 = sum(g((get_sym_val(gamma_AB) - 1) / 2)) - g((lamda2 - 1) / 2)

    # Squeezed State External Dispersion
    I_AB3 = 0.5 * np.log2((T * (V + chi) + 1) / (T * (chi + 1 / V) + 1))
    S_BE3 = sum(g((get_sym_val(gamma_AB) - 1) / 2)) - g((get_sym_val(gamma_X_BA) - 1) / 2)

    # Coherent State External Dispersion
    lamda4 = (T * (V * chi + 1) + 1) / (T * (V + chi) + 1)
    I_AB4 = np.log2((T * (V + chi) + 1) / (T * (chi + 1) + 1))
    S_BE4 = sum(g((get_sym_val(gamma_AB) - 1) / 2)) - g((lamda4 - 1) / 2)

    if id == 1:
        K_val = 0.5 * (beta * I_AB1 - S_BE1)
    elif id == 2:
        K_val = beta * I_AB2 - S_BE2
    elif id == 3:
        K_val = beta * I_AB3 - S_BE3
    else:
        K_val = beta * I_AB4 - S_BE4
    return K_val

# Todo: 3. Define the symplectic eigenvalue function
def get_sym_val(gamma):
    # Construct omega matrix using Kronecker product
    omega = np.kron(np.eye(int(len(gamma) / 2)), np.array([[0, 1], [-1, 0]]))
    # Calculate the symplectic matrix
    sym = 1j * np.dot(omega, gamma)
    # Calculate eigenvalues and eigenvectors of the symplectic matrix
    sys_val = np.linalg.eig(sym)[0]
    val = []
    # Save eigenvalues > 1
    for j in sys_val:
        if np.real(j) > 1:
            val.append(np.real(j))
    return np.array(val)

l = np.linspace(0, 500, 500)
e_t1, e_t2, e_t3, e_t4 = [], [], [], []
for d in l:
    mark1, mark2, mark3, mark4 = False, False, False, False
    # Squeezed State Zero Dispersion -> 1
    for e in np.linspace(0, 1, 1000):
        if K(d, V, e, 1) < 0:
            e_t1.append(e)
            mark1 = True
            break
    if mark1 is False:
        e_t1.append(0)
    # Coherent State Zero Dispersion -> 2
    for e in np.linspace(0, 1, 1000):
        if K(d, V, e, 2) < 0:
            e_t2.append(e)
            mark2 = True
            break
    if mark2 is False:
        e_t2.append(0)
    # Squeezed State External Dispersion -> 3
    for e in np.linspace(0, 1, 1000):
        if K(d, V, e, 3) < 0:
            e_t3.append(e)
            mark3 = True
            break
    if mark3 is False:
        e_t3.append(0)
    # Coherent State External Dispersion -> 4
    for e in np.linspace(0, 1, 1000):
        if K(d, V, e, 4) < 0:
            e_t4.append(e)
            mark4 = True
            break
    if mark4 is False:
        e_t4.append(0)

x_ticks = np.linspace(0, 500, 6)
y_ticks = np.linspace(0, 1, 11)
plt.plot(l, e_t1, 'b:', label='Squeezed State Zero Dispersion')
plt.plot(l, e_t2, 'r-', label='Coherent State Zero Dispersion')
plt.plot(l, e_t3, 'y-.', label='Squeezed State External Dispersion')
plt.plot(l, e_t4, 'purple', label='Coherent State External Dispersion')
plt.title(f'Reverse Coordination, V={V}, β={beta}', fontsize=15)
plt.xlabel('Distance (km)', fontsize=15)
plt.ylabel('Excess Noise (SNU)', fontsize=15)
plt.legend()
plt.xticks(x_ticks)
plt.xlim(0, 500)
plt.show()