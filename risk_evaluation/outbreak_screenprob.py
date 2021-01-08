'''
First try at new outbreak analysis
'''

import argparse
from run import Run
import numpy as np
import utils as ut

class OutbreakScreenProb(Run):
    def build_configs(self):
        # NPI / in-school transmissibility
        npi_scens = {x:{'beta_s': 1.5*x} for x in [0.75, 0.75*1.6]}
        self.builder.add_level('In-school transmission multiplier', npi_scens, self.builder.screenpars_func)

        # Sweep over symptom screening
        symp_screens = {p:{'screen_prob': p} for p in np.linspace(0, 1, 10)}
        self.builder.add_level('Screen prob', symp_screens, self.builder.screenpars_func)

        return super().build_configs()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    sweep_pars = {
        'n_reps':       1,
        'n_prev':       0, # No controller
        'schcfg_keys':  ['with_countermeasures'],
        'school_start_date': '2021-02-01',
        'school_seed_date': '2021-02-01',
        'screen_keys':  [ 'None' ],
    }

    sim_pars = {
        'pop_infected': 0,
        'pop_size': 100_000,
        'start_day': '2021-01-31',
        'end_day': '2021-07-31',
        'beta_layer': dict(h=0, s=0, w=0, c=0), # Turn off non-school transmission
    }

    runner = OutbreakScreenProb(sweep_pars=sweep_pars, sim_pars=sim_pars)
    runner.run(args.force)
    analyzer = runner.analyze()

    xvar='Screen prob'
    huevar='In-school transmission multiplier'

    #runner.regplots(xvar=xvar, huevar=huevar)
    analyzer.outbreak_reg(xvar, huevar, order=4)

    analyzer.cum_incidence(colvar=xvar, rowvar=huevar)
    analyzer.outbreak_size_over_time(colvar=xvar, rowvar=huevar)
    analyzer.source_pie()

    runner.tsplots()
