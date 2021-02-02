'''
Outbreak analysis to sweep in-school transmissibility
'''

import sys
import utils as ut
import config as cfg
from run import Run
import numpy as np

alt_sus = False

class OutbreakBetaSchool(Run):
    def build_configs(self):
        # Configure alternate sus
        if alt_sus:
            value_labels = {'Yes' if p else 'No':p for p in [True]}
            self.builder.add_level('AltSus', value_labels, ut.alternate_symptomaticity)

        # Sweep over NPI multipliers
        npi_scens = {x:{'beta_s': 1.5*x} for x in np.linspace(0, 2, 10)}
        self.builder.add_level('In-school transmission multiplier', npi_scens, self.builder.screenpars_func)

        return super().build_configs()


if __name__ == '__main__':

    args = cfg.process_inputs(sys.argv)


    sweep_pars = {
        'n_prev':       0, # No controller
        'school_start_date': '2021-02-01',
        'school_seed_date': '2021-02-01',
        #'screen_keys':  ['None'],
        #'schcfg_keys':  ['with_countermeasures'],
    }

    pop_size = cfg.sim_pars.pop_size
    sim_pars = {
        'pop_infected': 0, # Do not seed
        'pop_size': pop_size,
        'start_day': '2021-01-31',
        'end_day': '2021-08-31',
        'beta_layer': dict(w=0, c=0), # Turn off work and community transmission
    }

    runner = OutbreakBetaSchool(sweep_pars=sweep_pars, sim_pars=sim_pars)
    runner.run(args.force)
    analyzer = runner.analyze()

    analyzer.outbreak_R0()

    xvar='In-school transmission multiplier'
    huevar=None

    #runner.regplots(xvar=xvar, huevar=huevar)
    analyzer.outbreak_reg(xvar, huevar)

    analyzer.cum_incidence(colvar=xvar)
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()

    runner.tsplots()
