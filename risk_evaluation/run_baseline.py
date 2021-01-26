'''
Run a varitey of sceening scenarios at a few prevalence levels
'''

import argparse
from run import Run
import numpy as np
import utils as ut

class Baseline(Run):
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
        'n_reps':       5,
        'n_prev':       20,
        'screen_keys':  ['None'],
    }

    runner = Baseline(sweep_pars=sweep_pars, sim_pars=dict(pop_size=100_000))
    runner.run(args.force)
    analyzer = runner.analyze()

    runner.regplots(xvar='Prevalence Target', huevar='Dx Screening')

    ext='sm'
    g = analyzer.introductions_rate(xvar='Prevalence Target', huevar='Dx Screening', order=2, height=6, aspect=1, ext=ext)
    g.set_xlabels('Prevalence')
    g._legend.remove()
    fn = 'IntroductionRate.png' if ext is None else f'IntroductionRate_{ext}.png'
    import covasim as cv
    import os
    import matplotlib.pyplot as plt
    plt.grid()
    print(os.path.join(analyzer.imgdir, fn))
    cv.savefig(os.path.join(analyzer.imgdir, fn), dpi=300)
    exit()

    analyzer.cum_incidence(colvar='Prevalence Target')
    analyzer.introductions_rate_by_stype(xvar='Prevalence Target', colvar=None, huevar='stype', order=3)
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()
    analyzer.source_dow(figsize=(6.5,5))

    runner.tsplots()
