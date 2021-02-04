'''
Dense sweep of screen_prob at a few fixed prevalence levels.
'''

import sys
import os
import matplotlib.pyplot as plt
import utils as ut
import config as cfg
from run import Run
import numpy as np

alt_sus = False

class ScreenProb(Run):
    def build_configs(self):
        # Configure alternate sus
        if alt_sus:
            value_labels = {'Yes' if p else 'No':p for p in [True]}
            self.builder.add_level('AltSus', value_labels, ut.alternate_symptomaticity)

        # Sweep over symptom screening
        symp_screens = {p:{'screen_prob': p} for p in np.linspace(0, 1, 10)}
        self.builder.add_level('Screen prob', symp_screens, self.builder.screenpars_func)

        return super().build_configs()


if __name__ == '__main__':

    args = cfg.process_inputs(sys.argv)

    # Optional overrides
    sweep_pars = dict(
        # n_reps = 5,
        # n_prev = 20,
        prev =  [0.005, 0.01],
    )
    pop_size = cfg.sim_pars.pop_size

    runner = ScreenProb(sweep_pars=sweep_pars, sim_pars=dict(pop_size=pop_size))
    runner.run(args.force)
    analyzer = runner.analyze()

    runner.regplots(xvar='Screen prob', huevar='Prevalence Target')

    #analyzer.introductions_rate(xvar='Screen prob', huevar='Prevalence', height=5, aspect=1.4, ext='_wide')

    analyzer.cum_incidence(colvar='Screen prob')
    analyzer.introductions_rate_by_stype(xvar='Screen prob')
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()

    runner.tsplots()
