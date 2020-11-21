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

__all__ = ['controller']

def coxian(p):
    n = len(p)
    M = np.zeros((n,n))
    np.fill_diagonal(M, 1-p)
    np.put(M, n+(n+1)*np.arange(n-1), p[:-1])
    return M

class controller(cv.Intervention):
    '''
    TODO
    '''

    def __init__(self, targets, gain=None, betat=None, **kwargs):
        assert(gain is not None or betat is not None)

        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so that intervention can be recreated

        # Store arguments
        self.targets = targets
        self.gain = gain
        self.betat = betat
        self.start_day = 15
        self.end_date = '2099-12-02'

        self.EI = coxian(np.array([0.4950429 , 0.51759924, 0.5865702 ]))
        self.nEI = self.EI.shape[0]
        self.Bei = np.zeros((self.nEI,1))
        self.Bei[0] = 1

        self.IR = coxian(np.array([0.66154341, 0.61511552, 1.52192331, 0.69897356, 0.6143495, 0.61457423, 0.70117798]))
        self.nIR = self.IR.shape[0]
        self.Bir = np.zeros((self.nIR,1))
        self.Bir[0] = 1

        self.E = np.matrix(np.zeros( (self.nEI,1) ))
        self.I = np.matrix(np.zeros( (self.nIR,1) ))
        self.integrated_err = 0

        self.K = np.array([[0.07376376, 0.67313015, 1.3197574, 0.12207705, 1.89558587, -0.89605874, 2.73307806, 0.45082311, 0.34529195, 0.22522394, 0.10519977]])

        return

    def initialize(self, sim):
        self.beta0 = sim.pars['beta']
        if self.betat is None:
            self.betat = sim.pars['beta'] * np.ones(sim.pars['n_days']+1)

        self.tpi = sim.pars['interventions'][0]
        self.initialized = True

    def apply(self, sim):
        print(sim.t, '-'*80)
        #if sim.t == sim.day('2020-11-02'):
        #    import time
        #    r = np.random.rand(int(time.time_ns() % 1e4)) # Pull a random number to mess up the stream

        #if self.gain is None:
        #    # Playback
        #    sim.pars['beta'] = self.betat[sim.t]
        #    return

        self.E = self.EI * self.E + self.Bei * sim.results['new_infections'][sim.t-1]
        self.I = self.IR * self.I + self.Bir * sim.results['new_infectious'][sim.t-1]

        if sim.t < self.start_day:
            return

        '''
        if sim.t == sim.day(self.end_date):
            self.betat[sim.t:] = self.betat[sim.t-1] + (self.betat[sim.t-1]-self.betat[sim.t-61])/60 * np.arange(sim.pars['n_days']-sim.t+1)
        elif sim.t < sim.day(self.end_date):
            # Cases
            new_diagnoses = 14 * sim.results['new_diagnoses'][sim.t-1]/sim.scaled_pop_size * 100_000 # 14 day rate
            target_new_diagnoses = self.targets['cases']
            didt = 14*100_000/sim.scaled_pop_size * (sim.results['new_diagnoses'][sim.t-1] - sim.results['new_diagnoses'][sim.t-8])/7
            adjustment = (self.gain*target_new_diagnoses + (1-self.gain)*new_diagnoses) / new_diagnoses - 0.25 * didt / target_new_diagnoses

            adjustment *= sim.results['n_exposed'][sim.t-2] / sim.results['n_exposed'][sim.t-1]
            #adjustment *= sim.results['n_infectious'][sim.t-2] / sim.results['n_infectious'][sim.t-1]
            adjustment = np.median([adjustment, 0.33, 3])

            self.betat[sim.t] = sim.pars['beta'] * adjustment
            print(f'{sim.datevec[sim.t]}: New {new_diagnoses:.0f} -- Target {target_new_diagnoses:.0f} -- Adjustment {adjustment:.3f} -- Beta {self.betat[sim.t]:.5f} -- Exposed: {sim.results["n_exposed"][sim.t-1]} -- Infectious: {sim.results["n_infectious"][sim.t-1]}')

            #print(sim.results.keys())
            # Tests
            new_tests = sim.results['new_tests'][sim.t-1]
            target_new_tests = self.targets['tests']/100_000 * sim.scaled_pop_size

            adjustment = (self.gain*target_new_tests + (1-self.gain)*new_tests) / new_tests
            #self.tpi.asymp_prob = self.tpi.asymp_quar_prob = self.tpi.asymp_prob * adjustment

            test_yield = sim.results['new_diagnoses'][sim.t-1]/sim.results['new_tests'][sim.t-1]
            target_yield = self.targets['yield']
            adjustment = (self.gain*target_yield + (1-self.gain)*test_yield) / test_yield

            #self.tpi.symp_prob = self.tpi.symp_quar_prob = self.tpi.symp_prob * adjustment

            print('TESTS:', new_tests, 'TARGET:', target_new_tests)
            #print('SYMP_PROB:', self.tpi.symp_prob, self.tpi.symp_quar_prob)
            #print('ASYMP_PROB:', self.tpi.asymp_prob, self.tpi.asymp_quar_prob)
            print('YIELD:', test_yield, 'TARGET:', target_yield)

        sim.pars['beta'] = self.betat[sim.t]
        '''

        y = np.sum(sim.people.exposed)#sim.results['n_exposed'][sim.t-1]
        N = sim.scaled_pop_size # don't really care about N vs alive...
        S = N - np.count_nonzero(~np.isnan(sim.people.date_exposed)) #sim.results['cum_infections'][sim.t-1]
        I = np.sum(sim.people.infectious)#sim.results['n_infectious'][sim.t-1]
        E = y - I

        Q = 50
        e = np.zeros((self.nEI,Q))
        e[0,0] = 1
        for k in range(Q-1):
            e[:,k+1] = np.dot(self.EI,e[:,k])

        norme = e/e.sum(axis=0)
        evec = np.zeros((Q,1))
        for k in range(Q):
            evec[k,0] = np.sum(
                np.logical_and(
                    np.logical_and(
                        np.logical_and(
                            sim.people.date_exposed == sim.t-k, # Q-1-k
                            np.logical_not(sim.people.infectious)
                        ),
                        np.logical_not(sim.people.recovered)
                    ),
                    ~sim.people.dead
                )
            )
        csE = np.dot(norme,evec)

        zzz = np.sum(np.logical_and(sim.people.exposed, np.logical_not(sim.people.infectious)))
        #print('Echeck:', E, np.sum(csE), zzz)

        #print(sim.people.date_exposed[np.logical_and(sim.people.exposed, ~sim.people.infectious)])
        #for k in range(Q):
        #    print(sim.t-k, np.sum( np.logical_and( np.logical_and( np.logical_and( sim.people.date_exposed == sim.t-k, np.logical_not(sim.people.infectious)), np.logical_not(sim.people.recovered)), ~sim.people.dead)))

        i = np.zeros((self.nIR,Q))
        i[0,0] = 1
        for k in range(Q-1):
            i[:,k+1] = np.dot(self.IR,i[:,k])


        normi = i/i.sum(axis=0)
        ivec = np.zeros((Q,1))
        for k in range(Q):
            ivec[k,0] = np.sum(np.logical_and(sim.people.date_infectious == sim.t-k, sim.people.infectious))
        csI = np.dot(normi,ivec)
        zzz = np.sum(sim.people.infectious)
        #print('Icheck:', I, np.sum(csI), zzz)


        print('\nE: ', np.hstack([csE, self.E]))
        print('\nI: ', np.hstack([csI, self.I]))

        Xu = np.vstack([self.integrated_err, csE, csI])
        u = np.asarray(-self.K * Xu)[0][0]

        Xu_orig = np.vstack([self.integrated_err, self.E, self.I])
        u_orig = np.asarray(-self.K * Xu_orig)[0][0]

        print(f'Covasim E={E:.0f}, I={I:.0f} | SEIR E={np.sum(self.E):.0f}, I={np.sum(self.I):.0f} --> U={u:.1f}, Uorig={u_orig:.1f}, beta={sim.pars["beta"]}')

        if S*I > 0:
            sim.pars['beta'] = np.maximum(0.1 * u_orig * N / (S*I), 0)

        self.integrated_err = self.integrated_err + y - self.targets['EI']

        '''
        if sim.t == 126:
            inds = []
            for k in range(Q):
                inds += list(np.where(np.logical_and(np.logical_and(sim.people.date_exposed == sim.t-k, np.logical_not(sim.people.infectious)), np.logical_not(sim.people.recovered))))#.tolist()
            inds = np.hstack(inds)
            print(inds)
            inds = np.unique(inds)
            print(inds)
            print('N:', len(inds))
            print('Exposed:', np.sum(sim.people.exposed[inds]))
            #print('Infected:', np.sum(sim.people.infected[inds]))
            print('Infectious:', np.sum(sim.people.infectious[inds]))
            print('Recovered:', np.sum(sim.people.recovered[inds]))
            print('Died:', np.sum(sim.people.dead[inds]))
            exit()
        '''


        return
