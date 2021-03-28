'''
Creates an intervention that controls prevalence to a certain level. This intervention
is typically created automatically by the Manager class in school_tools.
'''

import covasim as cv
import numpy as np
from .controller import Controller as ct
from .Kalman import Kalman as kf


__all__ = ['controller_intervention']

class controller_intervention(cv.Intervention):
    '''
    Control prevalence to a certain value.
    '''

    def __init__(self, SEIR, targets, pole_loc=None, start_day=0, verbose=False, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object

        self.verbose = verbose

        # Store arguments
        self.targets = targets
        self.SEIR = SEIR
        self.u_k =  None

        self.start_day = start_day
        self.end_date = '2099-12-02'

        self.Controller = ct(SEIR, pole_loc=pole_loc)
        self.integrated_err = 0
        self.set_beta_directly = False # Legacy option
        return


    def initialize(self, sim):
        self.beta0 = sim.pars['beta']
        self.beta_layer0 = sim.pars['beta_layer'].copy()
        self.u_k = sim.pars['beta'] * np.ones(sim.pars['n_days']+1)

        initial_exposed_pop = np.sum(sim.people.exposed)
        self.Kalman = kf(initial_exposed_pop, self.SEIR)
        self.initialized = True
        return


    def apply(self, sim):
        if self.verbose: print(sim.t, '-'*80)

        y = sim.results['n_exposed'][sim.t-1]
        N = sim.scaled_pop_size # don't really care about N vs alive...
        S = N - np.count_nonzero(~np.isnan(sim.people.date_exposed))
        I = np.sum(sim.people.infectious)
        E = y - I

        if self.verbose: print('NEW EXPOSURES:', sim.results['new_infections'][sim.t-1])

        if sim.t < self.start_day or S*I==0:
            u = sim.results['new_infections'][sim.t-1] # S*I/N # np.sum(sim.people.date_exposed == sim.t-1)
            self.Kalman.update(E, I, u)

        else:

            Xu = np.vstack([self.Kalman.EIhat, self.integrated_err])
            u = self.Controller.get_control(Xu)
            u = np.maximum(u,0)

            # Should be in conditional
            self.Kalman.update(E, I, u)

            if self.verbose: print(f'Covasim E={E:.0f}, I={I:.0f} | SEIR E={np.sum(self.Kalman.Ehat()):.0f}, I={np.sum(self.Kalman.Ihat()):.0f} --> U={u:.1f}, beta={sim.pars["beta"]}')

            xs = S
            Ihat = self.Kalman.Ihat()

            xi = np.sum(Ihat)
            xi += np.sum(Ihat[:3]) # Double infectivity early

            if xi > 0:
                xi = np.power(xi, self.SEIR.Ipow) # Ipow - do this in SEIR class where Ipow is known?
                SI_by_N = xs*xi / N
                new_beta = np.maximum(u / SI_by_N, 0)

                if self.verbose: print('WARNING: shrinking beta!')
                new_beta /= 18
            else:
                print(f'WARNING: Estimate of infectious population ({xi}) is negative, resetting beta!')
                new_beta = self.beta0

            if self.set_beta_directly: # Set beta directly - includes all layers including inside each school
                sim.pars['beta'] = new_beta
            else: # Set beta_layer for all "traditional" layers, will include individuals schools if using covid_schools
                for lkey in ['h', 's', 'w', 'c', 'l']:
                    sim.pars['beta_layer'][lkey] = self.beta_layer0[lkey] * new_beta / self.beta0
            self.u_k[sim.t] = new_beta

            if self.verbose:
                print(f'CONTROLLER IS ASKING FOR {u} NEW EXPOSURES')
                print(f'New error estimate --- y={y}, n_exp={sim.results["n_exposed"][sim.t-1]}, t={self.targets["infected"]}, err={y - self.targets["infected"]} --> int err now {self.integrated_err}')

            self.integrated_err = self.integrated_err + y - self.targets['infected'] # TODO: use ReferenceTrajectory class

        return
