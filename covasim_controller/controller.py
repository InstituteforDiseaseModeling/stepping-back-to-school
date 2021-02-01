import numpy as np
from numpy.linalg import matrix_power as mp

__all__ = ['Controller']


class Controller:

    def __init__(self, SEIR, pole_loc=0.7):
        self.SEIR = SEIR
        self.pole_loc = pole_loc
        self._design_controller()

    def _design_controller(self):
        # Dynamics for feedback controller design

        nEI = self.SEIR.EI.shape[0]
        nIR = self.SEIR.IR.shape[0]

        if self.SEIR.nERR > 0:
            # SEIR already has error dynamic, no need to augment!
            # Keep only E, I, and error
            N = self.SEIR.A.shape[0]
            inds = np.r_[1:N-2, N-1]
            Ac = self.SEIR.A[np.ix_(inds,inds)]
            Bc = self.SEIR.B[inds]

        else:
            # Keep only E and I
            Af = self.SEIR.A[1:-1,1:-1]
            Bf = self.SEIR.B[1:-1]
            Cf = self.SEIR.C[:,1:-1]

            # Augment with error dynamics so that x is [E_dot, I_dot, e_dot]
            Ac = np.block( [[Af, np.zeros((nEI+nIR,1))], [Cf, 1]] )
            Bc = np.vstack([Bf, 0])

        # Controllability matrix of combined error & state dynamics
        ctrb = np.matrix(np.zeros_like(Ac))
        ctrb[:,0] = Bc
        for i in range(nEI+nIR):
            ctrb[:,i+1] = Ac*ctrb[:,i] # columns of controllability matrix

        # Characteristic (monic) polynomial coefficients given pole locations:
        alpha = np.poly(self.pole_loc*np.ones(ctrb.shape[0])) # All poles at same place?

        alpha_c_F = mp(Ac,nEI+nIR+1)
        for i in range(1, nEI+nIR+1):
            alpha_c_F += alpha[i] * mp(Ac,nEI+nIR+1-i)  # Start at 1
        alpha_c_F += alpha[-1] * np.eye(nEI+nIR+1)

        last_row = np.matrix(np.zeros(nEI+nIR+1))
        last_row[:,-1] = 1

        # Ackermann's formula for pole placement
        #assert(ctrb.shape[0] == np.linalg.matrix_rank(ctrb))

        self.K = np.dot(last_row, np.dot(np.linalg.inv(ctrb), alpha_c_F)) # TODO: Solve

    def get_control(self, X):
        return np.asarray(-self.K * X)[0][0]
