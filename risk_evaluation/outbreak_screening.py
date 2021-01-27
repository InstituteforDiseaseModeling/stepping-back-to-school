'''
Outbreak analysis to sweep in-school transbissibility while also exploring several diagnostic screening scenarios.
'''

import argparse
from run import Run
import numpy as np
import utils as ut

class OutbreakScreening(Run):
    def build_configs(self):
        # Sweep over NPI multipliers
        #npi_scens = {x:{'beta_s': 1.5*x} for x in [0.75, 0.75*1.6]}
        npi_scens = {x:{'beta_s': 1.5*x} for x in np.linspace(0.25, 2, 10)}
        self.builder.add_level('In-school transmission multiplier', npi_scens, self.builder.screenpars_func)

        return super().build_configs()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    sweep_pars = {
        'n_reps': 10,
        'n_prev': 0, # No controller
        'schcfg_keys': ['with_countermeasures'],
        'school_start_date': '2021-02-01',
        'school_seed_date': '2021-02-01',
        'screen_keys': [
            'None',
            'Antigen every 1w teach&staff, PCR f/u',
            'PCR every 4w',
            'Antigen every 4w, PCR f/u',
            'PCR every 2w',
            'Antigen every 2w, PCR f/u',
            'PCR every 1w',
            'Antigen every 1w, PCR f/u',
        ],
    }

    sim_pars = {
        'pop_infected': 0,
        'pop_size': 223_000,
        'start_day': '2021-01-31',
        'end_day': '2021-07-31',
        'beta_layer': dict(h=0, s=0, w=0, c=0), # Turn off non-school transmission
    }

    runner = OutbreakScreening(sweep_pars=sweep_pars, sim_pars=sim_pars)
    runner.run(args.force)
    analyzer = runner.analyze()

    xvar='In-school transmission multiplier'
    huevar='Dx Screening'

    #runner.regplots(xvar=xvar, huevar=huevar)

    analyzer.outbreak_reg(xvar, huevar, order=4)

    analyzer.cum_incidence(colvar=xvar, rowvar=huevar)
    analyzer.outbreak_size_over_time(colvar=xvar, rowvar=huevar)
    analyzer.source_pie()

    runner.tsplots()
