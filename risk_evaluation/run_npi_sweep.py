'''
Dense sweep of beta_s at a few fixed prevalence levels.
'''

import argparse
from run import Run
import numpy as np

class NPISweep(Run):
    def __init__(self, sim_pars=None, sweep_pars=None, run_pars=None):
        name = self.__class__.__name__
        super().__init__(name, sim_pars, sweep_pars, run_pars)

    def build_configs(self):
        # Sweep over NPI multipliers
        npi_scens = {x:{'beta_s': 1.5*x} for x in np.linspace(0, 1, 10)}
        self.builder.add_level('NPI reduction', npi_scens, self.builder.screenpars_func)

        return super().build_configs()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    runner = NPISweep(sweep_pars=dict(n_reps=10, n_prev=3), sim_pars=dict(pop_size=223_000), run_pars=dict(n_cpus=15))
    runner.run(args.force)
    runner.plot(xvar='NPI reduction', huevar='prev', ts_plots=True, order=3)
