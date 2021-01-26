import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power as mp

nu = 1/4
gamma = 1/8
S_ref = 0.6 # NOT WHAT I WANT

A = np.matrix([ [ 0,  0,  0     ],
                [ 0, -nu, 0     ],
                [ 0,  nu, -gamma] ])
B = np.matrix([ [-1], [1], [0] ])
C = np.matrix([ [1, 0, 0] ]) # AGH, not what I want to track!
ctrb = np.hstack([B, np.dot(A,B), np.dot(A, np.dot(A,B))])

# With all three poles at -1, characteristic polynomial coefficients are:
_, alpha_1, alpha_2, alpha_3 = np.poly([-0.25,-0.25,-0.25])
alpha_c_F = mp(A, 3) + \
            alpha_1 * mp(A, 2) + \
            alpha_2 * mp(A, 1) + \
            alpha_3 * np.eye(3)
last_row = np.matrix([0, 0, 1])

# Ackermann's formula for pole placement
K = np.dot(last_row, np.dot(np.linalg.inv(ctrb), alpha_c_F))

blk = np.block([[A,B], [C,0]])
N = np.linalg.inv(blk) * np.matrix( [[0], [0], [0], [1]])
Nx = N[:-1]
Nu = N[-1]
Nbar = Nu + K*Nx


def sei(t,x,A,B,K):
    x = np.matrix(x).T
    u = -K*x + Nbar * S_ref
    dxdt = A*x + B*u
    return dxdt.T

t = np.linspace(0,10,100)
sln = spi.solve_ivp(sei, [0,20], [0.999, 0.001, 0], method='RK45', dense_output=False, max_step=0.1, args=(A,B,K))
plt.plot(sln.t, sln.y.T)
plt.legend(['Susceptible', 'Exposed', 'Infectious'])
plt.show()
