'''
Outbreak analysis to sweep in-school transmissibility while also exploring several diagnostic screening scenarios.
'''

import sys
import numpy as np
import school_tools as sct

if __name__ == '__main__':

    # Settings

    args = sct.config.process_inputs(sys.argv)

    sweep_pars = {
        'n_prev': 0, # No controller
        'school_start_date': '2021-02-01',
        'school_seed_date': '2021-02-01',
        'screen_keys':  ['None', 'Antigen every 1w teach&staff', 'Antigen every 4w', 'Antigen every 2w', 'Antigen every 1w', 'PCR every 1w'],
    }

    pop_size = sct.config.sim_pars.pop_size
    sim_pars = {
        'pop_infected': 0, # Do not seed
        'pop_size': pop_size,
        'start_day': '2021-01-31',
        'end_day': '2021-08-31',
        'beta_layer': dict(w=0, c=0), # Turn off work and community transmission
    }

    npi_scens = {x:{'beta_s': 1.5*x} for x in np.linspace(0.25, 2, 10)}
    levels = [{'keyname':'In-school transmission multiplier', 'level':npi_scens, 'func':'screenpars_func'}]

    xvar = 'In-school transmission multiplier'
    huevar = 'Dx Screening'

    # Create and run
    mgr = sct.Manager(name='OutbreakScreening', sweep_pars=sweep_pars, sim_pars=sim_pars, levels=levels)
    mgr.run(args.force)
    analyzer = mgr.analyze()

    # Plots
    analyzer.outbreak_size_distribution(row='Dx Screening', col='In-school transmission multiplier')
    analyzer.outbreak_reg(xvar, huevar)
    analyzer.cum_incidence(colvar=xvar, rowvar=huevar)
    analyzer.outbreak_size_over_time(colvar=xvar, rowvar=huevar)
    analyzer.source_pie()
    mgr.tsplots()
