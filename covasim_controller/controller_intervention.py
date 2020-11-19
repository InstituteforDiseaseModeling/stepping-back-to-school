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

        return

    def initialize(self, sim):
        self.beta0 = sim.pars['beta']
        if self.betat is None:
            self.betat = sim.pars['beta'] * np.ones(sim.pars['n_days']+1)

        self.tpi = sim.pars['interventions'][0]
        self.initialized = True

    def apply(self, sim):
        if sim.t == sim.day('2020-11-02'):
            import time
            r = np.random.rand(int(time.time_ns() % 1e4)) # Pull a random number to mess up the stream

        if self.gain is None:
            # Playback
            sim.pars['beta'] = self.betat[sim.t]
            return

        if sim.t < self.start_day:
            return

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

        return
