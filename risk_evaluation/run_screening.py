'''
Run a varitey of sceening scenarios at a few prevalence levels
'''

import argparse
from run import Run
import numpy as np
import utils as ut

class Screening(Run):
    def build_configs(self):
        # Configure alternate sus
        #value_labels = {'Yes' if p else 'No':p for p in [True]}
        #self.builder.add_level('AltSus', value_labels, ut.alternate_symptomaticity)

        return super().build_configs()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    sweep_pars = {
        'n_reps':       1,
        'n_prev':       3,
        'screen_keys':  ['None', 'PCR every 4w', 'Antigen every 1w teach&staff, PCR f/u', 'Antigen every 4w, PCR f/u', 'Antigen every 2w, no f/u', 'Antigen every 2w, PCR f/u', 'PCR every 2w', 'Antigen every 1w, PCR f/u', 'PCR every 1w'],
    }

    runner = Screening(sweep_pars=sweep_pars, sim_pars=dict(pop_size=100_000))#, run_pars=dict(n_cpus=15))
    runner.run(args.force)
    analyzer = runner.analyze()

    runner.regplots(xvar='Prevalence Target', huevar='Dx Screening')

    analyzer.cum_incidence(colvar='Prevalence Target')
    analyzer.introductions_rate_by_stype(xvar='Prevalence Target', colvar=None, huevar='stype', order=3)
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()

    runner.tsplots()
