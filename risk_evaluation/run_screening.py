'''
Run a varitey of sceening scenarios at a few prevalence levels

Example usage, forcing new results and using a 4 different seeds:

    python run_screening.py --force --n_reps=4

'''

import sys
import os
import matplotlib.pyplot as plt
import utils as ut
import config as cfg
from run import Run

alt_sus = False

class Screening(Run):
    def build_configs(self):
        # Configure alternate sus
        if alt_sus:
            value_labels = {'Yes' if p else 'No':p for p in [True]}
            self.builder.add_level('AltSus', value_labels, ut.alternate_symptomaticity)

        return super().build_configs()


if __name__ == '__main__':

    args = cfg.process_inputs(sys.argv)

    # Optional overrides
    sweep_pars = dict(
        # n_reps = 5,
        # n_prev = 20,
        screen_keys =  ['None', 'PCR every 4w', 'Antigen every 1w teach&staff, PCR f/u', 'Antigen every 4w, PCR f/u', 'Antigen every 2w, PCR f/u', 'PCR every 2w', 'Antigen every 1w, PCR f/u', 'PCR every 1w'],
    )
    pop_size = cfg.sim_pars.pop_size

    runner = Screening(sweep_pars=sweep_pars, sim_pars=dict(pop_size=pop_size))
    runner.run(args.force)
    analyzer = runner.analyze()

    runner.regplots(xvar='Prevalence Target', huevar='Dx Screening')

    analyzer.introductions_rate(xvar='Prevalence Target', huevar='Dx Screening', height=5, aspect=2, ext='_wide')


    analyzer.cum_incidence(colvar='Prevalence Target')
    analyzer.introductions_rate_by_stype(xvar='Prevalence Target')
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()

    runner.tsplots()
