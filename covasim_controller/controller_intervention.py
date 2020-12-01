'''
Main file implementing school-based interventions.  The user interface is handled
by the schools_manager() intervention. This primarily uses the School class, of
which there is one instance per school. SchoolTesting orchestrates testing within
a school, while SchoolStats records results. The remaining functions are contact
managers, which handle different cohorting options (and school days).
'''

import covasim as cv
import numpy as np
import sciris as sc
from .TransitionMatrix import TransitionMatrix as tm


__all__ = ['controller_intervention']

class controller_intervention(cv.Intervention):
    '''
    TODO
    '''

    def __init__(self, targets, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so that intervention can be recreated

        # Store arguments
        self.targets = targets
        self.betat = kwargs['betat'] if 'betat' in kwargs else None
        self.start_day = 15
        self.end_date = '2099-12-02'

        self.integrated_err = 0

        self.EI = tm.full([0.54821612, 0.40217725, 0.24984496, 0.03756642, 0.5213049, 0.40917672])
        self.nEI = self.EI.shape[0]

        self.IR = tm.full([ 3.40190364e-01,  3.63341049e-01,  3.35031247e-02,  2.96464203e-01,
                9.65784905e-01,  1.12059492e-02,  4.37587969e-06, -1.21861188e-07,
                9.62334305e-01,  9.75919056e-03, -6.23660031e-10,  1.64779825e-08,
                1.01696020e-02,  9.77812849e-01,  5.15078739e-02,  4.48849543e-10,
                1.53316944e-08,  1.62901373e-02,  1.24279573e-02,  9.48492111e-01,
                3.71067650e-01,  4.71039087e-09,  7.12066962e-04,  3.37529797e-12,
               -5.69699539e-10,  1.97380899e-08,  6.28932347e-01,  5.53031491e-01])
        self.nIR = self.IR.shape[0]

        # Filtered state estimate
        self.nEIR = self.nEI + self.nIR
        self.EIhat = np.matrix(np.zeros( (self.nEIR,1) )) # TODO: Informed guess
        self.Sigma = 1 * np.eye(self.nEIR) # State estimate covariance # TODO: Better guess

        self.Q = 10 * np.eye(self.nEIR) # Process noise - TODO: better guess
        self.R = 1 * np.eye(2)  # Observation noise (E and I) # TODO

        #self.K = np.array([[0.04388146, 0.27750344, 0.96649746, -0.14699181, 0.92476425, -0.36230294, 0.3816167, 0.25341555, 0.21416596, 0.16789652, 0.0981611]])

        self.build_SEIR()

        return

    def initialize(self, sim):
        self.beta0 = sim.pars['beta']
        if self.betat is None:
            self.betat = sim.pars['beta'] * np.ones(sim.pars['n_days']+1)

        self.EIhat[0,0] = np.sum(sim.people.exposed)

        self.tpi = sim.pars['interventions'][0]
        self.initialized = True

    def Kalman(self, yE, yI, u):
        # remove S and R states, leaving just E and E
        A = self.A[1:-1,1:-1]
        B = self.B[1:-1]
        #C = self.C[:,1:-1]

        Ce = np.hstack([np.ones(self.nEI), np.zeros(self.nIR)])
        Ci = np.hstack([np.zeros(self.nEI), np.ones(self.nIR)])
        C = np.matrix(np.vstack([Ce, Ci]))
        y = np.vstack([yE, yI])

        print('BEFORE:', self.EIhat.T)
        #print(self.Sigma)

        # Predict
        self.EIhat = A * self.EIhat + B * u
        self.Sigma = A * self.Sigma * A.T + self.Q

        # Correct
        L = self.Sigma * C.T * np.linalg.inv(C*self.Sigma*C.T + self.R)
        self.EIhat += L * (y - C*self.EIhat)
        self.Sigma = (np.eye(self.nEIR) - L*C)*self.Sigma

        print('AFTER:', self.EIhat.T)
        print('Corrected error:', (y-C*self.EIhat).T)
        #print(self.Sigma)

    def apply(self, sim):
        print(sim.t, '-'*80)
        #if sim.t == sim.day('2020-11-02'):
        #    import time
        #    r = np.random.rand(int(time.time_ns() % 1e4)) # Pull a random number to mess up the stream

        y = np.sum(sim.people.exposed)#sim.results['n_exposed'][sim.t-1]
        N = sim.scaled_pop_size # don't really care about N vs alive...
        S = N - np.count_nonzero(~np.isnan(sim.people.date_exposed)) #sim.results['cum_infections'][sim.t-1]
        I = np.sum(sim.people.infectious)#sim.results['n_infectious'][sim.t-1]
        E = y - I

        if sim.t < self.start_day:
            u = sim.results['new_infections'][sim.t-1] # S*I/N # np.sum(sim.people.date_exposed == sim.t-1)
            print('NEW EXPOSURES:', u, sim.results['new_infections'][sim.t-1], S*I/N)
            self.Kalman(E, I, u)
            return

        Xu = np.vstack([self.integrated_err, self.EIhat])
        u = np.asarray(-self.K * Xu)[0][0]
        u = np.maximum(u,0)

        #print('WARNING: reducing u')
        #u *= 0.1

        # Should be in conditional
        self.Kalman(E, I, u)

        print(f'Covasim E={E:.0f}, I={I:.0f} | SEIR E={np.sum(self.EIhat[:self.nEI]):.0f}, I={np.sum(self.EIhat[self.nEI:]):.0f} --> U={u:.1f}, beta={sim.pars["beta"]}')

        if S*I > 0:
            #expected_new_infections = S*I/N

            xs = S
            xi = np.sum(self.EIhat[np.arange(self.nEI, self.nEI+self.nIR)])
            xi += np.sum(self.EIhat[np.arange(self.nEI, self.nEI+3)]) # Double infectivity early

            expected_new_infections = xs*xi / N # np.power(xi, 1)
            print(expected_new_infections, S*I/N)

            sim.pars['beta'] = np.maximum(u / expected_new_infections, 0)

        self.integrated_err = self.integrated_err + y - self.targets['EI']

        return
