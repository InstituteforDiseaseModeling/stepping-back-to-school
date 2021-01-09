'''
Sweeping in-school transmissibility
'''

import argparse
from run import Run
import numpy as np
import utils as ut

class OutbreakBetaSchool(Run):
    def build_configs(self):
        # Sweep over NPI multipliers
        npi_scens = {x:{'beta_s': 1.5*x} for x in np.linspace(0, 2, 10)}
        self.builder.add_level('In-school transmission multiplier', npi_scens, self.builder.screenpars_func)

        return super().build_configs()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    sweep_pars = {
        'n_reps':       10,
        'n_prev':       0, # No controller
        'screen_keys':  ['None'],
        'schcfg_keys':  ['with_countermeasures'],
        'school_start_date': '2021-02-01',
        'school_seed_date': '2021-02-01',
    }

    sim_pars = {
        'pop_infected': 0,
        'pop_size': 223_000,
        'start_day': '2021-01-31',
        'end_day': '2021-08-31',
        'beta_layer': dict(h=0, s=0, w=0, c=0), # Turn off non-school transmission
    }

    runner = OutbreakBetaSchool(sweep_pars=sweep_pars, sim_pars=sim_pars)
    runner.run(args.force)
    analyzer = runner.analyze()

    analyzer.outbreak_R0()

    xvar='In-school transmission multiplier'
    huevar=None

    #runner.regplots(xvar=xvar, huevar=huevar)
    analyzer.outbreak_reg(xvar, huevar, order=5)

    analyzer.cum_incidence(colvar=xvar)
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()

    runner.tsplots()
