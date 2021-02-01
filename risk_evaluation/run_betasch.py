'''
Dense sweep of in-school transmissibility (beta_s) at a few fixed prevalence levels.
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

    sim_pars = dict(pop_size=223_000, end_day='2021-04-30')

    runner = BetaSchool(sweep_pars=dict(n_reps=10, prev=[0.002, 0.007, 0.014]), sim_pars=sim_pars)
    runner.run(args.force)
    analyzer = runner.analyze()
    #runner.plot(xvar='In-school transmission multiplier', huevar='Prevalence', ts_plots=True, order=3)

    runner.regplots(xvar='In-school transmission multiplier', huevar=None)

    analyzer.outbreak_reg(xvar='In-school transmission multiplier', huevar=None, height=5, aspect=2, ext='_wide')

    analyzer.cum_incidence(colvar='In-school transmission multiplier')
    analyzer.introductions_rate_by_stype(xvar='In-school transmission multiplier')
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()

    runner.tsplots()
