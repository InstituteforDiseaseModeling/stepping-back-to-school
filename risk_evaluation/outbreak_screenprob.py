'''
Outbreak analysis to sweep in-school transbissibility and screening probability
'''

import sys
import os
import matplotlib.pyplot as plt
import utils as ut
import config as cfg
from run import Run
import numpy as np


alt_sus = False

class OutbreakScreenProb(Run):
    def build_configs(self):
        # Configure alternate sus
        if alt_sus:
            value_labels = {'Yes' if p else 'No':p for p in [True]}
            self.builder.add_level('AltSus', value_labels, ut.alternate_symptomaticity)

        # NPI / in-school transmissibility
        npi_scens = {x:{'beta_s': 1.5*x} for x in [0.75, 0.75*1.6]}
        self.builder.add_level('In-school transmission multiplier', npi_scens, self.builder.screenpars_func)

        # Sweep over symptom screening
        symp_screens = {p:{'screen_prob': p} for p in np.linspace(0, 1, 10)}
        self.builder.add_level('Screen prob', symp_screens, self.builder.screenpars_func)

        return super().build_configs()


if __name__ == '__main__':
    args = cfg.process_inputs(sys.argv)

    sweep_pars = {
        'n_prev':       0, # No controller
        'school_start_date': '2021-02-01',
        'school_seed_date': '2021-02-01',
        #'schcfg_keys':  ['with_countermeasures'],
        #'screen_keys':  [ 'None' ],
    }

    pop_size = cfg.sim_pars.pop_size
    sim_pars = {
        'pop_infected': 0, # Do not seed
        'pop_size': pop_size,
        'start_day': '2021-01-31',
        'end_day': '2021-08-31',
        'beta_layer': dict(w=0, c=0), # Turn off work and community transmission
    }

    runner = OutbreakScreenProb(sweep_pars=sweep_pars, sim_pars=sim_pars)
    runner.run(args.force)
    analyzer = runner.analyze()

    xvar='Screen prob'
    huevar='In-school transmission multiplier'

    #runner.regplots(xvar=xvar, huevar=huevar)
    analyzer.outbreak_reg(xvar, huevar, order=4)

    analyzer.cum_incidence(colvar=xvar, rowvar=huevar)
    analyzer.outbreak_size_over_time(colvar=xvar, rowvar=huevar)
    analyzer.source_pie()

    runner.tsplots()
