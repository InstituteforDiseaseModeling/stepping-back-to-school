'''
Copy of outbreak script to use for testing.
'''

import numpy as np
import sciris as sc
import school_tools as sct

alt_sus = False


class OutbreakBetaSchool(sct.Run):
    def build_configs(self):
        # Configure alternate sus
        if alt_sus:
            value_labels = {'Yes' if p else 'No':p for p in [True]}
            self.builder.add_level('AltSus', value_labels, sct.alternate_symptomaticity)

        # Sweep over NPI multipliers
        npi_scens = {x:{'beta_s': 1.5*x} for x in np.linspace(0, 2, 2)}
        self.builder.add_level('In-school transmission multiplier', npi_scens, self.builder.screenpars_func)

        return super().build_configs()


def test_outbreaks():

    # Minimal example
    sct.config.sweep_pars.n_reps = 2
    sct.config.sweep_pars.n_seeds = 2
    sct.config.sim_pars.pop_size = 20_000
    sct.config.paths.inputs = sc.thisdir(None, 'inputs')
    sct.config.paths.outputs = sc.thisdir(None, 'outputs')
    sct.config.run_pars.parallel = False # Interferes with coverage otherwise

    sweep_pars = {
        'n_prev':  1, # Include controller
        'school_start_date': '2021-02-01',
        'school_seed_date': '2021-02-01',
        #'screen_keys':  ['None'],
        #'schcfg_keys':  ['with_countermeasures'],
    }

    pop_size = sct.config.sim_pars.pop_size
    sim_pars = {
        'pop_infected': 0, # Do not seed
        'pop_size': pop_size,
        'start_day': '2021-01-31',
        'end_day': '2021-03-31',
        'beta_layer': dict(w=0, c=0), # Turn off work and community transmission
    }

    sct.create_pops(cfg=sct.config)

    runner = OutbreakBetaSchool(sweep_pars=sweep_pars, sim_pars=sim_pars, cfg=sct.config)
    runner.run(force=True)
    analyzer = runner.analyze()

    analyzer.outbreak_size_distribution(row='In-school transmission multiplier', col=None)

    analyzer.outbreak_R0()

    xvar='In-school transmission multiplier'
    huevar=None

    #runner.regplots(xvar=xvar, huevar=huevar)
    analyzer.outbreak_reg(xvar, huevar)

    analyzer.cum_incidence(colvar=xvar)
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()

    runner.tsplots()



if __name__ == '__main__':

    test_outbreaks()

