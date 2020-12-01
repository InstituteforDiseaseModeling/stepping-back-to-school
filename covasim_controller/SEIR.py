import numpy as np

__all__ = ['SEIR']

class SEIR:
    '''
    Simulate a SEIR system.
    '''

    def __init__(self, pop_size, EI, IR, beta=0.185, Ipow=1, C=None, dt=1):
        '''
        Initialize:
        * pop_size is the size of the population
        * EI is the transition matrix from E-->I
        * IR is the transition matrix from I-->R
        * beta is the infectivity
        * Ipow is the power to which I is raised in beta*S*I^(Ipow)/N to represent heterogeneity and lower the final size
        * C is the observation matrix, will default to E+I if None
        * dt is the time step (not really used)
        '''

        self.dt = dt
        self.pop_size = pop_size
        self.EI = EI
        self.IR = IR
        self.beta = beta
        self.Ipow = Ipow
        self.C = C

        self._build()
        self.check_controllability()
        self.check_observability()

    def _build(self):
        '''
        Build system matrics.
        '''

        nEI = self.EI.shape[0]
        nIR = self.IR.shape[0]
        belowEI = np.zeros((nIR, nEI))
        belowEI[0,:] = 1-np.sum(self.EI, axis=0)
        belowIR = 1-np.sum(self.IR, axis=0)

        # Full SEIR Dynamics
        self.A = np.block( [
            [1,                 np.zeros(nEI),     np.zeros(nIR),       0                   ],
            [np.zeros((nEI,1)), self.EI,           np.zeros((nEI,nIR)), np.zeros((nEI,1))   ],
            [np.zeros((nIR,1)), belowEI,           self.IR,             np.zeros((nIR,1))   ],
            [0,                 np.zeros((1,nEI)), belowIR,             1                   ] ])

        self.B = np.matrix(np.zeros((2+nEI+nIR,1)))
        self.B[0] = -1
        self.B[1] = 1

        if self.C is None:
            self.C = np.matrix(np.zeros((1,2+nEI+nIR)))
            self.C[:, 1:-1] = 1

        self.nEI = nEI
        self.nIR = nIR

    def check_observability(self):
        '''
        Check observability of infected (E & I) states only
        '''

        A = self.A[1:-1, 1:-1]
        C = self.C[:, 1:-1]

        nEIR = self.nEI + self.nIR

        Omats = [C]
        for i in range(1, nEIR):
            Omats.append(Omats[-1]*A)
        obs_mat = np.vstack(Omats)
        assert(np.linalg.matrix_rank(obs_mat) == nEIR)

        return obs_mat

    def check_controllability(self):
        '''
        Check controllability of infected (E & I) states only
        '''

        A = self.A[1:-1, 1:-1]
        B = self.B[1:-1, :]

        ctb_mat = np.matrix(np.zeros((self.nEI+self.nIR, self.nEI+self.nIR)))
        ctb_mat[:,0] = B
        for i in range(self.nEI+self.nIR-1):
            ctb_mat[:,i+1] = A*ctb_mat[:,i] # columns of controllability matrix
        assert(np.linalg.matrix_rank(ctb_mat) == self.nEI+self.nIR)


    def step(self, X, u_beta):
        x = np.matrix(X).T # Move away from matrix

        xs = x[0]
        xi = np.sum(x[np.arange(self.nEI+1, self.nEI + self.nIR+1)])
        xi += np.sum(x[np.arange(self.nEI+1, self.nEI+4)]) # Double infectivity early

        if self.Ipow != 1:
            xi = np.power(xi, self.Ipow)
        u = u_beta * xs * xi / self.pop_size
        y = self.C*x

        new_x = self.A*x + self.B*u

        return np.squeeze(np.asarray(new_x)), y


    def run(self, n_seeds, n_days):
        X = np.zeros((self.A.shape[1], n_days+1))
        X[0,0] = self.pop_size - n_seeds
        X[1,0] = n_seeds
        Y = np.zeros(n_days)

        for k in range(n_days):
            X[:,k+1], Y[k] = self.step(X[:,k], self.beta)

        return X

    def plot(self):
        pass
