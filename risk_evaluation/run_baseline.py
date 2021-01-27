'''
Introduction analysis sweeping over several prevalence levels.

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


class Baseline(Run):
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

    try:
        analyzer.cum_incidence(colvar='Prevalence Target')
        analyzer.introductions_rate_by_stype(xvar='Prevalence Target', colvar=None, huevar='stype', order=3)
        analyzer.outbreak_size_over_time()
        analyzer.source_pie()
        analyzer.source_dow(figsize=(6.5,5))
        runner.tsplots()
    except Exception as E:
        print(f'Could not run some plots: {str(E)}')

