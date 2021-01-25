import numpy as np

__all__ = ['SEIR', 'err_dyn']

class SEIR:
    '''
    Simulate a SEIR system.
    '''

    def __init__(self, pop_size, EI, IR, ERR=None, beta=0.185, Ipow=1):
        '''
        Initialize:
        * pop_size is the size of the population
        * EI is the transition matrix from E-->I
        * IR is the transition matrix from I-->R
        * ERR is the err dynamics matrix (optional, will be augmented to state)
        * beta is the infectivity
        * Ipow is the power to which I is raised in beta*S*I^(Ipow)/N to represent heterogeneity and lower the final size
        '''

        self.pop_size = pop_size
        self.EI = EI
        self.IR = IR
        self.ERR = np.matrix(ERR)
        self.beta = beta
        self.Ipow = Ipow

        self._build()
        self.check_controllability()
        self.check_observability()

        self.X = None
        self.Y = None
        self.k = 0 # current step


    def _build(self):
        '''
        Build system matrics.
        '''

        nEI = self.EI.shape[0]
        nIR = self.IR.shape[0]
        nERR = 0
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

        self.C = np.matrix(np.zeros((1,2+nEI+nIR)))
        self.C[:, 1:-1] = 1

        self.Bref = None

        if self.ERR is not None:
            # Augment with error dynamics so that x is [E_dot, I_dot, e_dot]
            # TODO: WARNING: Will only work with 1D error (e.g. ERR=1)
            self.A = np.block( [[self.A, np.zeros((2+nEI+nIR,1))], [self.C, self.ERR]] )
            self.B = np.vstack([self.B, 0])
            self.C = np.hstack([self.C, np.matrix(0)])
            self.Bref = np.zeros_like(self.B)
            self.Bref[-1] = -1
            nERR = self.ERR.shape[0]

        self.nEI = nEI
        self.nIR = nIR
        self.nERR = nERR


    def step(self, u=None, ref=None):
        ''' Take a single update step '''
        x = np.matrix(self.X[:,self.k]).T # Move away from matrix

        # Allow user to override beta
        if u is None:
            xs = x[0]
            xi = np.sum(x[np.arange(self.nEI+1, self.nEI + self.nIR+1)])
            xi += np.sum(x[np.arange(self.nEI+1, self.nEI+4)]) # Double infectivity early

            if self.Ipow != 1:
                xi = np.power(xi, self.Ipow)

            u = self.beta * xs * xi / self.pop_size

        new_x = self.A*x + self.B*u
        if ref is not None:
            assert(self.Bref is not None)
            new_x += self.Bref * ref

        self.X[:,self.k+1] = np.squeeze(np.asarray(new_x))
        self.Y[self.k+1] = self.C*x
        self.k += 1


    def reset(self, n_seeds, n_days):
        ''' Reset the state '''
        self.X = np.zeros((self.A.shape[1], n_days+1))
        self.X[0,0] = self.pop_size - n_seeds
        self.X[1,0] = n_seeds
        self.Y = np.zeros(n_days+1)
        self.Y[0] = n_seeds

        self.k = 0


    def finalize(self):
        ''' Package the state for output '''
        results = {
            'S': self.X[0,:],
            'E': np.sum(self.X[1:(self.nEI+1),:], axis=0),
            'I': np.sum(self.X[(self.nEI+1):(self.nEI+self.nIR+1),:], axis=0),
            'R': self.X[-1-self.nERR,:],
            'e': self.X[-self.nERR:,:],
            'X': self.X,
            'Y': self.Y,
        }

        return results


    def run(self, n_seeds, n_days):
        ''' Run the SEIR model from n_seeds seed infections for n_days days '''
        self.reset(n_seeds, n_days)

        for k in range(n_days):
            self.step()

        return self.finalize()


    def check_observability(self):
        '''
        Check observability of infected (E & I) states only
        '''

        nEIR = self.nEI + self.nIR
        A = self.A[1:1+nEIR, 1:1+nEIR]
        C = self.C[:, 1:1+nEIR]

        Omats = [C]
        for i in range(1, nEIR):
            Omats.append(Omats[-1]*A)
        obs_mat = np.vstack(Omats)
        #assert(np.linalg.matrix_rank(obs_mat) == nEIR)

        return obs_mat


    def check_controllability(self):
        '''
        Check controllability of infected (E & I) states only
        '''

        nEIR = self.nEI + self.nIR
        A = self.A[1:1+nEIR, 1:1+nEIR]
        B = self.B[1:1+nEIR, :]

        ctb_mat = np.matrix(np.zeros((self.nEI+self.nIR, self.nEI+self.nIR)))
        ctb_mat[:,0] = B
        for i in range(self.nEI+self.nIR-1):
            ctb_mat[:,i+1] = A*ctb_mat[:,i] # columns of controllability matrix
        #assert(np.linalg.matrix_rank(ctb_mat) == self.nEI+self.nIR)


    def plot(self):
        pass


# TODO: Need [err,E,I] in controller design
# Then u=-K * [err,E,I]
# Seems error dynamics should be integrated
class err_dyn():
    ''' System dynamics for tracking error '''
    def __init__(self, A, ref):
        self.A = np.matrix(A)
        self.ref = ref

    def reset(self, n_seeds, n_days):
        self.X = np.zeros((self.A.shape[1], n_days+1))
        self.k = 0

    def step(self, y):
        x = np.matrix(self.X[:,self.k]).T # Move away from matrix
        new_x = self.A*x + y - self.ref.get(self.k) # err is: C*x - y_ref
        self.X[:,self.k+1] = np.squeeze(np.asarray(new_x))
