import scipy.integrate as spi
import scipy.linalg as spl # for expm
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power as mp

nu = 1/4
gamma = 1/8
EI_ref = 0.04

dt = 1.0
#A = np.matrix([ [ -nu, 0     ],
#                [  nu, -gamma] ])
#print(spl.expm(A*dt))

# Dynamics of E and I
A = np.matrix([ [ 1-nu, 0     ],
                [  nu, 1-gamma] ])

B = np.matrix([ [1], [0] ])
C = np.matrix([ [1, 1] ]) # y = E + I

# Error dynamics: e_dot, E_dot, I_dot
Ac = np.block( [[1, C], [np.zeros((2,1)), A]] )
Bc = np.vstack([0, B])

ctrb = np.hstack([Bc, Ac*Bc, Ac*Ac*Bc])

# With all three poles at -1, characteristic polynomial coefficients are:
_, alpha_1, alpha_2, alpha_3 = np.poly([0.2, 0.6+0.3j, 0.6-0.3j])
alpha_c_F = mp(Ac, 3) + \
            alpha_1 * mp(Ac, 2) + \
            alpha_2 * mp(Ac, 1) + \
            alpha_3 * np.eye(3)
last_row = np.matrix([0, 0, 1])

# Ackermann's formula for pole placement
K = np.dot(last_row, np.dot(np.linalg.inv(ctrb), alpha_c_F))

# Full SEIR Dynamics
Af = np.matrix([[ 1,   0,      0,     0],
                [ 0, 1-nu,      0,     0],
                [ 0,  nu, 1-gamma,     0],
                [ 0,   0,  gamma,     1]    ])
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

def step(t, X, A, B, C, K):
    X = np.matrix(X).T
    integrated_error = X[0]
    x = X[1:]

    u = -K*X[[0,2,3]] # integrated error, E, I
    y = C*x
    new_integrated_error = integrated_error + y - EI_ref

    new_x = A*x + B*u
    new = np.vstack( [new_integrated_error, new_x] )

    return np.squeeze(np.asarray(new))

t = np.linspace(0,10,100)
steps = 15
X = np.zeros((5,steps+1))
X[:,0] = [0, 0.999, 0.001, 0, 0]
for k in range(steps):
    X[:,k+1] = step(k*dt, X[:,k], Af, Bf, Cf, K)

fig, axv = plt.subplots(1,4,figsize=(16,10))

axv[0].plot(X[1:,:].T)
axv[0].legend(['Susceptible', 'Exposed', 'Infectious', 'Recovered'])
axv[0].set_title('SEIR')

axv[1].plot(X[0,:].T)
axv[1].set_title('Integrated Tracking Error')

axv[2].plot(np.asarray(Cf*X[1:,:])[0])
axv[2].plot(X[[2,3],:].T)
axv[2].axhline(y=EI_ref, color='r', ls='--')
axv[2].legend(['Exposed+Infectious', 'Exposed', 'Infectious', 'Reference'])
axv[2].set_title('Tracking')

axv[3].plot(np.asarray(-K*X[[0,2,3],:])[0]) # TODO: Compute beta
axv[3].set_title('Control')

plt.show()
