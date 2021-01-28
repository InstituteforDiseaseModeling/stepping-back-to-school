'''
Run a varitey of sceening scenarios at a few prevalence levels

Example usage, forcing new results and using a 4 different seeds:

    python run_baseline.py --force --n_reps=4

'''

import sys
import os
import matplotlib.pyplot as plt
import utils as ut
import config as cfg
from run import Run

alt_sus = False


class Scheduling(Run):
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
        schcfg_keys = ['as_normal', 'with_countermeasures', 'all_hybrid', 'k5'],
    )
    pop_size = cfg.sim_pars.pop_size

    runner = Scheduling(sweep_pars=sweep_pars, sim_pars=dict(pop_size=pop_size))
    runner.run(args.force)
    analyzer = runner.analyze()

    runner.regplots(xvar='Prevalence Target', huevar='Scenario')

    analyzer.introductions_rate(xvar='Prevalence Target', huevar='Scenario', height=5, aspect=2, ext='_wide')


    analyzer.cum_incidence(colvar='Prevalence Target')
    analyzer.introductions_rate_by_stype(xvar='Prevalence Target', colvar=None, huevar='stype', order=3)
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()

    runner.tsplots()
