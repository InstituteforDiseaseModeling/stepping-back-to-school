'''
Run a varitey of sceening scenarios at a few prevalence levels
'''

import argparse
from run import Run
import numpy as np

class Screening(Run):
    def __init__(self, sim_pars=None, sweep_pars=None, run_pars=None):
        name = self.__class__.__name__
        super().__init__(name, sim_pars, sweep_pars, run_pars)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    sweep_pars = {
        'n_reps':       2,
        'n_prev':       5,
        'screen_keys':  ['None', 'PCR every 4w', 'Antigen every 1w teach&staff, PCR f/u', 'Antigen every 4w, PCR f/u', 'Antigen every 2w, no f/u', 'Antigen every 2w, PCR f/u', 'PCR every 2w', 'Antigen every 1w, PCR f/u', 'PCR every 1w'],
    }

    runner = Screening(sweep_pars=sweep_pars, sim_pars=dict(pop_size=100_000), run_pars=dict(n_cpus=15))
    runner.run(args.force)
    runner.plot(xvar='prev_tgt', huevar='dxscrn_lbl', ts_plots=True)
