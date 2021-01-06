'''
Dense sweep of beta_s at a few fixed prevalence levels.
'''

import argparse
from run import Run
import numpy as np
import utils as ut

class BetaSchool(Run):
    def build_configs(self):
        # Sweep over NPI multipliers
        npi_scens = {x:{'beta_s': 1.5*x} for x in np.linspace(0, 2, 10)}
        self.builder.add_level('In-school transmission multiplier', npi_scens, self.builder.screenpars_func)

        return super().build_configs()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    runner = BetaSchool(sweep_pars=dict(n_reps=5, n_prev=10), sim_pars=dict(pop_size=223_000))#, run_pars=dict(n_cpus=15))
    runner.run(args.force)
    runner.plot(xvar='In-school transmission multiplier', huevar='Prevalence', ts_plots=True, order=3)
