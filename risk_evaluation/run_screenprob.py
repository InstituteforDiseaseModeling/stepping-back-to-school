'''
Dense sweep of screen_prob at a few fixed prevalence levels.
'''

import argparse
from run import Run
import numpy as np

class ScreenProb(Run):
    def __init__(self, sim_pars=None, sweep_pars=None, run_pars=None):
        name = self.__class__.__name__
        super().__init__(name, sim_pars, sweep_pars, run_pars)

    def build_configs(self):
        # Sweep over symptom screening
        symp_screens = {p:{'screen_prob': p} for p in np.linspace(0, 1, 10)}
        self.builder.add_level('Screen prob', symp_screens, self.builder.screenpars_func)

        return super().build_configs()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    runner = ScreenProb(sweep_pars=dict(n_reps=3, prev=[0.005]), sim_pars=dict(pop_size=100_000), run_pars=dict(n_cpus=15))
    runner.run(args.force)
    analyzer = runner.analyze()

    runner.regplots(xvar='Screen prob', huevar='Dx Screening')

    analyzer.introductions_rate(xvar='Screen prob', huevar='Prevalence', height=5, aspect=2, ext='_wide')

    analyzer.cum_incidence(colvar='Screen prob')
    analyzer.introductions_rate_by_stype(xvar='Screen prob', colvar=None, huevar='stype', order=3)
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()

    runner.tsplots()
