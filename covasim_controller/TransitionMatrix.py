import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt

__all__ = ['TransitionMatrix']

class TransitionMatrix:
    '''
    Class to build a transition matrix that results in an exit time distribution that fits data
    Two matrix options:
    1) Coxian, in which x1 --> x2 --> ... --> xN
    2) Full, in which x1 --> (x2, x3, ..., xN), x2 --> (x3, x4, ..., xN), ...
    '''

    def __init__(self, data, n, do_coxian=False):
        self.n = n # Dimension of the resulting matrix will be n x n
        self.do_coxian = do_coxian

        if self.do_coxian:
            self.method = self.coxian
            self.num_pars = self.n
        else:
            self.method = self.full
            self.num_pars = int(self.n*(self.n+1)/2)

        # Build PMF from data
        self.pmf, _ = np.histogram(data, bins=range(int(max(data))))
        self.pmf = self.pmf / np.sum(self.pmf)
        self.cmf = np.cumsum(self.pmf)

        self.opt_result = None
        self.Mopt = None


    def full(self, p):
        '''
        Build a full-type matrix
        '''

        M = np.zeros((self.n, self.n))
        inds = np.ravel_multi_index(np.tril_indices(self.n), M.shape)
        np.put(M, inds, p)
        return M


    def coxian(self, p):
        '''
        Build a coxian-type matrix
        '''

        M = np.zeros((self.n, self.n))
        np.fill_diagonal(M, 1-p)
        inds = self.n+(self.n+1)*np.arange(self.n-1)
        np.put(M, inds, p[:-1])
        return M


    def add_exit_time(self, M):
        '''
        Add a row to the transition matrix, M, to compute the cumulative mass that has exited
        '''

        F = np.block([[M, np.zeros((self.n,1))],
                      [1-M.sum(axis=0), 1]])
        return F


    def get_cum(self, x):
        '''
        Run the system represented by the parameters x.
        A transition matrix M is created from x using coxian or full method
        '''

        M = self.method(x)
        F = self.add_exit_time(M)

        T = len(self.cmf)
        dur = np.matrix(np.zeros((F.shape[0],T)))
        dur[:,0] = 0
        dur[0,0] = 1
        for k in range(T-1):
            dur[:,k+1] = F*dur[:,k]

        return dur[-1,:]


    def err(self, x):
        '''
        Compute the error between the simulated cumulative distribution and the CMF of the data
        Take the max absolute difference as the error metric
        '''

        cum = self.get_cum(x)
        err = np.max(np.abs(cum - self.cmf))
        return err


    def fit(self, guess=0.5):
        '''
        Fit the transition matrix to the data by constrained optimization
        '''

        x0 = guess*np.random.rand(self.num_pars)

        lc = [np.eye(self.num_pars)]
        inds = [0]
        for col in range(self.n-1):
            if col == 0:
                # Build inds
                for i in range(1,self.n):
                    inds.append(inds[-1]+i)
                inds = np.array(inds)
            else:
                inds += 1
                inds = inds[1:]
            constraint = np.zeros(self.num_pars)
            constraint[inds] = 1
            lc.append(constraint)

        A = np.vstack(lc)
        lc = spo.LinearConstraint(A, np.zeros(A.shape[0]), np.ones(A.shape[0]))
        self.opt_result = spo.minimize(self.err, x0, constraints=lc, method='SLSQP')

        # Make sure optimization was successful
        assert(self.opt_result['success'])

        # Store the optimal transition matrix
        self.Mopt = self.method(self.opt_result['x'])

        return self.opt_result


    def plot(self):
        '''
        Plot the result
        '''

        fig, ax = plt.subplots(2,1, figsize=(10,8))

        F = self.add_exit_time(self.Mopt)

        T = len(self.cmf)
        dur = np.matrix(np.zeros((F.shape[0],T+1)))
        dur[0,0] = 1
        for k in range(T):
            dur[:,k+1] = F*dur[:,k]

        for i in range(F.shape[0]-1):
            ax[0].plot(np.squeeze(np.asarray(dur[i,:])), label=f'$x_{{{i}}}$')
        ax[0].set_xlim([0,T])
        ax[0].legend()
        ax[0].set_ylabel('State')

        ax[1].bar(range(len(self.pmf)), np.cumsum(self.pmf))
        ax[1].plot( np.squeeze(np.asarray(dur[-1,:])), marker='o', ms=5, color='r', lw=0.5, mew=4, alpha=0.7)
        ax[1].set_xlim([0,T])

        ax[1].set_xlabel('Step')
        ax[1].set_ylabel('Cumulative distribution')

        fig.tight_layout()

        return fig
