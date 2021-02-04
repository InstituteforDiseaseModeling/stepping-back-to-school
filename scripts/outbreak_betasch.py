'''
Outbreak analysis to sweep in-school transmissibility
'''

import sys
import numpy as np
import school_tools as sct

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

    npi_scens = {x:{'beta_s': 1.5*x} for x in np.linspace(0, 2, 10)}
    levels = [{'keyname':'In-school transmission multiplier', 'level':npi_scens, 'func':'screenpars_func'}]

    runner = OutbreakBetaSchool(sweep_pars=sweep_pars, sim_pars=sim_pars, levels=levels)
    runner.run(args.force)
    analyzer = runner.analyze()

    analyzer.outbreak_size_distribution(row='In-school transmission multiplier', col=None)

    analyzer.outbreak_R0()

    xvar='In-school transmission multiplier'
    huevar=None

    #runner.regplots(xvar=xvar, huevar=huevar)
    analyzer.exports_reg(xvar, huevar)
    analyzer.outbreak_reg(xvar, huevar)
    analyzer.outbreak_size_plot(xvar, scatter=True, loess=False)
    analyzer.outbreak_size_plot(xvar, scatter=False, loess=True) # Included for completeness but looks weird

    analyzer.cum_incidence(colvar=xvar)
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()

    runner.tsplots()
