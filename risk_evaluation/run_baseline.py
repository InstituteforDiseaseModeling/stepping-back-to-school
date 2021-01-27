'''
Introduction analysis sweeping over several prevalence levels
'''

import sys
import os
import argparse
import matplotlib.pyplot as plt
import utils as ut
import config as cfg
from run import Run

alt_sus = False


class Baseline(Run):
    def build_configs(self):
        # Configure alternate sus
        if alt_sus:
            value_labels = {'Yes' if p else 'No':p for p in [True]}
            self.builder.add_level('AltSus', value_labels, ut.alternate_symptomaticity)

        return super().build_configs()


if __name__ == '__main__':

    cfg.process_inputs(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    # Optional overrides
    sweep_pars = dict(
        # n_reps = 5,
        # n_prev = 20,
        # screen_keys = ['None'],
    )
    pop_size = cfg.sim_pars.pop_size

    runner = Baseline(sweep_pars=sweep_pars, sim_pars=dict(pop_size=pop_size))
    runner.run(args.force)
    analyzer = runner.analyze()

    runner.regplots(xvar='Prevalence Target', huevar='Dx Screening')

    ###
    # One-off plot for the introduction rate.
    ext='sm'
    g = analyzer.introductions_rate(xvar='Prevalence Target', huevar='Dx Screening', order=2, height=6, aspect=1, ext=ext)
    g.set_xlabels('Prevalence')
    g._legend.remove()
    fn = 'IntroductionRate.png' if ext is None else f'IntroductionRate_{ext}.png'

    plt.grid()
    print(os.path.join(analyzer.imgdir, fn))
    plt.savefig(os.path.join(analyzer.imgdir, fn), dpi=300)

    analyzer.cum_incidence(colvar='Prevalence Target')
    analyzer.introductions_rate_by_stype(xvar='Prevalence Target', colvar=None, huevar='stype', order=3)
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()
    analyzer.source_dow(figsize=(6.5,5))

    runner.tsplots()
