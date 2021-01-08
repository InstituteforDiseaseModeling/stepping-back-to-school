'''
Debugging
'''

import argparse
from run import Run
import numpy as np
import utils as ut

class Debug(Run):
    def build_configs(self):
        # Sweep over NPI multipliers
        npi_scens = {x:{'beta_s': 1.5*x} for x in [0.75, 2]}
        self.builder.add_level('In-school transmission multiplier', npi_scens, self.builder.screenpars_func)

        return super().build_configs()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    sweep_pars = {
        'n_reps':       5,
        'n_prev':       2,
        'screen_keys':  ['None'],
        'schcfg_keys':  ['k5']
    }

    sim_pars = dict(pop_size=223_000, end_day='2021-04-30')
    runner = Debug(sweep_pars=sweep_pars, sim_pars=sim_pars)
    runner.run(args.force)
    analyzer = runner.analyze()

    xvar='In-school transmission multiplier'
    huevar='Prevalence'

    runner.regplots(xvar=xvar, huevar=huevar)

    analyzer.cum_incidence(colvar=xvar)
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()

    runner.tsplots()
