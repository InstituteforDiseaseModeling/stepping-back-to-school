'''
Run a varitey of sceening scenarios at a few prevalence levels
'''

import argparse
from run import Run
import numpy as np

class Baseline(Run):
    def __init__(self, sim_pars=None, sweep_pars=None, run_pars=None):
        name = self.__class__.__name__
        super().__init__(name, sim_pars, sweep_pars, run_pars)

    def build_configs(self):
        # Configure alternate sus
        value_labels = {'Yes' if p else 'No':p for p in [True]}
        self.builder.add_level('AltSus', value_labels, ut.alternate_symptomaticity)

        return super().build_configs()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    sweep_pars = {
        'n_reps':       3,
        'n_prev':       5,
        'screen_keys':  ['None'],
    }

    runner = Baseline(sweep_pars=sweep_pars, sim_pars=dict(pop_size=223_000), run_pars=dict(n_cpus=15))
    runner.run(args.force)
    runner.plot(xvar='Prevalence Target', huevar='Dx Screening', ts_plots=True)
