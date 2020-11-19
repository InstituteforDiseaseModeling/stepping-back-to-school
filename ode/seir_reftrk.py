import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power as mp

nu = 1/4
gamma = 1/8
EI_ref = 0.04

# Dynamics of E and I
A = np.matrix([ [ -nu, 0     ],
                [  nu, -gamma] ])
B = np.matrix([ [1], [0] ])
C = np.matrix([ [1, 1] ]) # y = E + I

# Error dynamics: e_dot, E_dot, I_dot
Ac = np.block( [[0, C], [np.zeros((2,1)), A]] )
Bc = np.vstack([0, B])

ctrb = np.hstack([Bc, Ac*Bc, Ac*Ac*Bc])

# With all three poles at -1, characteristic polynomial coefficients are:
_, alpha_1, alpha_2, alpha_3 = np.poly([-1, -.25, -.25])
alpha_c_F = mp(Ac, 3) + \
            alpha_1 * mp(Ac, 2) + \
            alpha_2 * mp(Ac, 1) + \
            alpha_3 * np.eye(3)
last_row = np.matrix([0, 0, 1])

# Ackermann's formula for pole placement
K = np.dot(last_row, np.dot(np.linalg.inv(ctrb), alpha_c_F))

# Full SEIR Dynamics
Af = np.matrix([[ 0,   0,      0,     0],
                [ 0, -nu,      0,     0],
                [ 0,  nu, -gamma,     0],
                [ 0,   0,  gamma,     0]    ])
Bf = np.matrix([ [-1], [1], [0], [0] ])
Cf = np.block([0,C,0])

def seir(t,X,A,B,C,K):
    X = np.matrix(X).T
    integrated_error = X[0]
    x = X[1:]

    u = -K*X[[0,2,3],:] # integrated error, E, I
    y = C*x
    d_integrated_error_dt = y - EI_ref

    dxdt = A*x + B*u
    return np.hstack( [d_integrated_error_dt, dxdt.T] )

t = np.linspace(0,10,100)
# Error dynamics
sln = spi.solve_ivp(seir, [0,25], [0, 0.999, 0.001, 0, 0], method='RK45', dense_output=False, max_step=0.1, args=(Af,Bf,Cf,K))

fig, axv = plt.subplots(1,3,figsize=(16,10))

axv[0].plot(sln.t, sln.y[1:,:].T)
axv[0].legend(['Susceptible', 'Exposed', 'Infectious', 'Recovered'])
axv[0].set_title('SEIR')

axv[1].plot(sln.t, sln.y[0,:].T)
axv[1].set_title('Integrated Tracking Error')

axv[2].plot(sln.t, np.asarray(Cf*sln.y[1:,:])[0])
axv[2].plot(sln.t, sln.y[[2,3],:].T)
axv[2].axhline(y=EI_ref, color='r', ls='--')
axv[2].legend(['Exposed+Infectious', 'Exposed', 'Infectious', 'Reference'])
axv[2].set_title('Tracking')

plt.show()
