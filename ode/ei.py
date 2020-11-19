import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power as mp

nu = 1/4
gamma = 1/8
EI_ref = 0.02

A = np.matrix([ [ -nu, 0     ],
                [  nu, -gamma] ])
B = np.matrix([ [1], [0] ])
C = np.matrix([ [1, 1] ])
ctrb = np.hstack([B, np.dot(A,B)])

# With all three poles at -1, characteristic polynomial coefficients are:
_, alpha_1, alpha_2 = np.poly([-0.2,-0.2])
alpha_c_F = mp(A, 2) + \
            alpha_1 * mp(A, 1) + \
            alpha_2 * np.eye(2)
last_row = np.matrix([0, 1])

# Ackermann's formula for pole placement
K = np.dot(last_row, np.dot(np.linalg.inv(ctrb), alpha_c_F))

blk = np.block([[A,B], [C,0]])
N = np.linalg.inv(blk) * np.matrix( [[0], [0], [1]])
Nx = N[:-1]
Nu = N[-1]
Nbar = Nu + K*Nx

def ei(t,x,A,B,K):
    x = np.matrix(x).T
    u = -K*x + Nbar * EI_ref
    dxdt = A*x + B*u
    return dxdt.T

t = np.linspace(0,10,100)
sln = spi.solve_ivp(ei, [0,25], [0.001, 0], method='RK45', dense_output=False, max_step=0.1, args=(A,B,K))
plt.plot(sln.t, sln.y.T)
plt.legend(['Exposed', 'Infectious'])
plt.show()
