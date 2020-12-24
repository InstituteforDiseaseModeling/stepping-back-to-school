'''
This is the main script used for commissioning the different testing scenarios
(defined in testing_scenarios.py) for the paper results. Run this script first
(preferably on an HPC!), then run the different plotting scripts. Each sim takes
about 1 minute to run. With full settings, there are about 1300 scripts to run;
the test run uses 8.
'''

import argparse
from run import Run

class ScreenProb(Run):
    def __init__(self, sim_pars=None, sweep_pars=None, run_pars=None):
        name = self.__class__.__name__
        super().__init__(name, sim_pars, sweep_pars, run_pars)

    def build_configs(self):
        # Sweep over symptom screening
        symp_screens = {
            'No symptom screening': { 'screen_prob': 0 },
            '50% daily screening':  { 'screen_prob': 0.5 },
            '100% daily screening': { 'screen_prob': 1 },
        }
        self.builder.add_level('ikey', symp_screens, self.builder.screenpars_func)

        return super().build_configs()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    runner = ScreenProb(sweep_pars=dict(n_reps=3, n_prev=20))#, run_pars=dict(n_cpus=1))
    runner.run(args.force, 'ikey', ts_plots=True)
