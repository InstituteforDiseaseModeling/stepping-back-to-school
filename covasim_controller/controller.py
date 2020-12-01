import numpy as np
from numpy.linalg import matrix_power as mp

__all__ = ['Controller']


class Controller:

    def __init__(self, SEIR):
        self.SEIR = SEIR


    def design_controller(self, pole_loc=0.7):
        # Dynamics for feedback controller design
        Af = self.SEIR.A[1:-1,1:-1]
        Bf = self.SEIR.B[1:-1]
        Cf = self.SEIR.C[:,1:-1]
        nEI = self.EI.shape[0]
        nIR = self.IR.shape[0]

        # Error dynamics: e_dot, E_dot, I_dot
        Ac = np.block( [[1, Cf], [np.zeros((nEI+nIR,1)), Af]] )
        Bc = np.vstack([0, Bf])

        # Controllability matrix of combined error & state dynamics
        ctrb = np.matrix(np.zeros((nEI+nIR+1, nEI+nIR+1))) # +1 for error dynamics
        ctrb[:,0] = Bc
        for i in range(nEI+nIR):
            ctrb[:,i+1] = Ac*ctrb[:,i] # columns of controllability matrix

        # Characteristic (monic) polynomial coefficients given pole locations:
        alpha = np.poly(pole_loc*np.ones(ctrb.shape[0])) # All poles at same place?

        alpha_c_F = mp(Ac,nEI+nIR+1)
        for i in range(1, nEI+nIR+1):
            alpha_c_F += alpha[i] * mp(Ac,nEI+nIR+1-i)  # Start at 1
        alpha_c_F += alpha[-1] * np.eye(nEI+nIR+1)

        last_row = np.matrix(np.zeros(nEI+nIR+1))
        last_row[:,-1] = 1

        # Ackermann's formula for pole placement
        assert(ctrb.shape[0] == np.linalg.matrix_rank(ctrb))
        K = np.dot(last_row, np.dot(np.linalg.inv(ctrb), alpha_c_F)) # TODO: Solve
        return K

