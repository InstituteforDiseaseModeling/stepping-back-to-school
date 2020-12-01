import numpy as np
from numpy.linalg import matrix_power as mp

__all__ = ['Controller']


class Controller:

    def __init__(self):

        tm = TransitionMatrix()

        self.EI = tm.full([0.54821612, 0.40217725, 0.24984496, 0.03756642, 0.5213049 ,
               0.40917672])

        self.IR = tm.full([ 3.40190364e-01,  3.63341049e-01,  3.35031247e-02,  2.96464203e-01,
                9.65784905e-01,  1.12059492e-02,  4.37587969e-06, -1.21861188e-07,
                9.62334305e-01,  9.75919056e-03, -6.23660031e-10,  1.64779825e-08,
                1.01696020e-02,  9.77812849e-01,  5.15078739e-02,  4.48849543e-10,
                1.53316944e-08,  1.62901373e-02,  1.24279573e-02,  9.48492111e-01,
                3.71067650e-01,  4.71039087e-09,  7.12066962e-04,  3.37529797e-12,
               -5.69699539e-10,  1.97380899e-08,  6.28932347e-01,  5.53031491e-01])

    def design_controller(self, pole_loc):
        # Dynamics for feedback controller design
        Af = self.A[1:-1,1:-1]
        Bf = self.B[1:-1]
        Cf = self.C[:,1:-1]
        nEI = self.nEI
        nIR = self.nIR

        # Error dynamics: e_dot, E_dot, I_dot
        Ac = np.block( [[1, Cf], [np.zeros((nEI+nIR,1)), Af]] )
        Bc = np.vstack([0, Bf])

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


        # self.K = self.design_controller(pole_loc=0.7)

